"""Optional, assay-scoped field QSAR orchestration; no docking is run here."""

from __future__ import annotations

import hashlib
import io
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope import provenance
from sarscope.analysis.align3d import AlignmentResult, align_series
from sarscope.analysis.contours import ContourResult, contribution_maps
from sarscope.analysis.domain import DomainResult, pca_bounding_box
from sarscope.analysis.features import fingerprint_matrix
from sarscope.analysis.field_filter import FieldColumnFilter
from sarscope.analysis.fields import FieldResult, GridSpec, compute_fields
from sarscope.analysis.regression import (
    RegressionResult,
    _regression_folds,
    evaluate_regression,
    regression_cv_predictions,
    regression_split,
)
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.curate import curate_table
from sarscope.field_source_quality import require_closed_shell
from sarscope.params import CurationParams, FeatureParams, FieldQsarParams, ModelParams, RunParams
from sarscope.units import to_pactivity
from sarscope.variants import normalize_variant

BENCHMARK_STATUS = (
    "SEH steroid benchmark agreement (training/LOO) demonstrated by SARscope's own "
    "field-to-model implementation. Historical test-mean shifting is diagnostic only. "
    "Native end-to-end validation remains incomplete; this feature is experimental."
)


@dataclass
class FieldDataset:
    table: pd.DataFrame
    molecules: list[Any]
    measurements: pd.DataFrame
    accounting: pd.DataFrame
    source: dict[str, Any]


@dataclass
class PreparedFieldSeries:
    dataset: FieldDataset
    alignment: AlignmentResult
    params: FieldQsarParams
    alignment_settings: dict[str, Any]


@dataclass
class FieldQsarResult:
    prepared: PreparedFieldSeries
    fields: FieldResult
    regression: RegressionResult
    baseline: RegressionResult
    predictions: pd.DataFrame
    statistics: pd.DataFrame
    checks: pd.DataFrame
    permutations: pd.DataFrame
    contours: ContourResult
    domain: DomainResult
    provenance: dict[str, Any]
    status: str
    training_stability: dict[str, Any] | None = None


def _variant(value: Any) -> str:
    if value is None or pd.isna(value):
        return "Unreported"
    annotation = normalize_variant(value, source="chembl_structured")
    return ";".join(annotation.normalized_mutations) or annotation.raw_annotation or "Unreported"


def assay_contexts(evidence: pd.DataFrame) -> pd.DataFrame:
    required = {"molecule_id", "assay_chembl_id", "standard_type", "pactivity"}
    if not required <= set(evidence):
        return pd.DataFrame()
    records = evidence.dropna(subset=["assay_chembl_id", "standard_type"]).copy()
    records["variant_scope"] = [
        _variant(v)
        for v in records.get("assay_variant_mutation", pd.Series(None, index=records.index))
    ]
    if "standard_relation" in records:
        records = records[records["standard_relation"] == "="]
    return (
        records.groupby(["assay_chembl_id", "standard_type", "variant_scope"], sort=True)
        .agg(compounds=("molecule_id", "nunique"), records=("molecule_id", "size"))
        .reset_index()
        .sort_values("compounds", ascending=False, kind="stable")
    )


def dataset_from_assay(
    table: pd.DataFrame,
    evidence: pd.DataFrame,
    assay: str,
    endpoint: str,
    variant: str,
    source: dict[str, Any] | None = None,
) -> FieldDataset:
    """Reaggregate exact records in one assay/endpoint/structured-variant context.

    Target-wide medians are explicitly overwritten. Duplicated copies of the
    same record are removed, not equal-valued independent measurements.
    """
    if assay_contexts(evidence).empty:
        raise ValueError("No exact, assay-linked measurements are available.")
    records = evidence.copy()
    records["variant_scope"] = [
        _variant(v)
        for v in records.get("assay_variant_mutation", pd.Series(None, index=records.index))
    ]
    records = records[
        (records["assay_chembl_id"] == assay)
        & (records["standard_type"] == endpoint)
        & (records["variant_scope"] == variant)
    ].copy()
    if "standard_relation" in records:
        records = records[records["standard_relation"] == "="]
    records["pactivity"] = pd.to_numeric(records["pactivity"], errors="coerce")
    records = records[np.isfinite(records["pactivity"])].copy()
    if "record_id" in records:
        records = records.drop_duplicates(["molecule_id", "record_id", "pactivity"])
    values = records.groupby("molecule_id")["pactivity"].agg(["median", "count", "min", "max"])
    scoped = table[table["molecule_id"].isin(values.index)].copy().reset_index(drop=True)
    scoped["pactivity"] = scoped["molecule_id"].map(values["median"])
    scoped["n_measurements"] = scoped["molecule_id"].map(values["count"])
    scoped["pactivity_range"] = scoped["molecule_id"].map(values["max"] - values["min"])
    molecules = [Chem.MolFromSmiles(smi) for smi in scoped["smiles"]]
    if any(mol is None for mol in molecules):
        raise ValueError("An assay structure could not be parsed.")
    require_closed_shell(molecules, scoped["molecule_id"].astype(str).tolist())
    return FieldDataset(
        scoped,
        molecules,
        records.reset_index(drop=True),
        pd.DataFrame(
            [
                {
                    "step": "selected assay / endpoint / variant",
                    "records": len(records),
                    "molecules": len(scoped),
                }
            ]
        ),
        {
            **(source or {}),
            "kind": "chembl_assay",
            "assay": assay,
            "endpoint": endpoint,
            "variant": variant,
            "aggregation": "median of exact retained measurements within selected context",
        },
    )


def upload_table(data: bytes, suffix: str, max_rows: int = 500) -> tuple[pd.DataFrame, list[Any]]:
    """Read bounded inputs without dropping malformed SDF records or CSV rows."""
    if len(data) > 5 * 1024**2:
        raise ValueError("Upload at most 5 MB.")
    if suffix.lower() in ("sdf", ".sdf"):
        molecules, rows = [], []
        for index, mol in enumerate(Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False)):
            if index >= max_rows:
                raise ValueError(f"Upload at most {max_rows} records.")
            if mol is None:
                raise ValueError(
                    f"SDF record {index + 1} could not be parsed; none were discarded."
                )
            rows.append(
                {
                    **mol.GetPropsAsDict(),
                    "molecule_id": mol.GetProp("_Name")
                    if mol.HasProp("_Name") and mol.GetProp("_Name").strip()
                    else f"compound_{index + 1}",
                    "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                }
            )
            molecules.append(mol)
        return pd.DataFrame(rows), molecules
    table = pd.read_csv(
        io.BytesIO(data),
        sep="\t" if suffix.lower().endswith("tsv") else ",",
        dtype=str,
        nrows=max_rows + 1,
    )
    if len(table) > max_rows:
        raise ValueError(f"Upload at most {max_rows} records.")
    return table, []


def dataset_from_upload(
    raw: pd.DataFrame,
    molecules: list[Any],
    *,
    id_col: str,
    smiles_col: str,
    value_col: str,
    unit: str,
    endpoint: str,
    source_sha256: str,
    preserve_input_states: bool = False,
    split_col: str | None = None,
    train_value: str = "train",
    test_value: str = "held_out",
) -> FieldDataset:
    """Use existing curation; refuse silent geometry-changing standardisation.

    Uploaded endpoints must be homogeneous. Pre-aligned poses survive only
    if curation leaves their isomeric molecular graph unchanged. Multiple
    poses of the same compound are not independent QSAR observations. Explicit
    reviewed-state mode requires one single-component, closed-shell SDF pose
    per unique compound; it does not neutralise, aggregate or infer states.
    """
    required = [id_col, smiles_col, value_col]
    if raw.empty or not set(required) <= set(raw):
        raise ValueError("Choose identifier, SMILES and activity columns with at least one row.")
    if raw[required].isna().any().any():
        raise ValueError(
            "Missing identifiers, structures or activities; fix the input (no rows dropped)."
        )
    for context in ("assay_chembl_id", "standard_type", "assay_variant_mutation"):
        if context in raw and raw[context].fillna("Unreported").nunique() > 1:
            raise ValueError(f"Input mixes {context}; upload one comparable context.")
    if "standard_relation" in raw and not raw["standard_relation"].eq("=").all():
        raise ValueError(
            "Use exact activities only; censored or qualitative calls are not regression labels."
        )
    values = pd.to_numeric(raw[value_col], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Activities must be finite numbers, not screening calls.")
    if unit != "p":
        values = np.array([to_pactivity(float(value), unit) for value in values])
    measurements = pd.DataFrame(
        {
            "molecule_id": raw[id_col].astype(str).str.strip(),
            "smiles": raw[smiles_col].astype(str).str.strip(),
            "pactivity": values,
            "record_id": [f"upload:{i + 1}" for i in range(len(raw))],
        }
    )
    if (measurements["molecule_id"] == "").any():
        raise ValueError("Identifiers cannot be empty.")
    if measurements.groupby("molecule_id")["smiles"].nunique().max() > 1:
        raise ValueError("An identifier names multiple structures; use unambiguous compound IDs.")
    parsed = [Chem.MolFromSmiles(smi) for smi in measurements["smiles"]]
    if any(mol is None for mol in parsed):
        raise ValueError("At least one SMILES is invalid; no rows were discarded.")
    require_closed_shell(parsed, measurements["molecule_id"].tolist())
    if preserve_input_states:
        if len(molecules) != len(parsed):
            raise ValueError("Reviewed-state mode requires one coordinate-bearing SDF per row.")
        if any(len(Chem.GetMolFrags(mol)) != 1 for mol in parsed):
            raise ValueError("Reviewed-state mode needs single-component structures, not salts.")
        graphs = [Chem.MolToSmiles(mol) for mol in parsed]
        if len(set(graphs)) != len(graphs) or measurements.molecule_id.duplicated().any():
            raise ValueError(
                "Reviewed-state mode needs unique structures and identifiers; "
                "adjudicate replicates/poses before upload. No aggregation was performed."
            )
        # Reuse ordinary curation solely as a SHADOW identity check: multiple
        # protonation/tautomer representations are not independent compounds.
        # Never transfer its changed graphs or aggregated labels to the poses.
        identity = curate_table(
            measurements, RunParams(curation=CurationParams(canonical_tautomer=True))
        )
        if not identity.rejected.empty or len(identity.table) != len(measurements):
            raise ValueError(
                "Reviewed-state parent identity needs adjudication: rows may be alternate "
                "states/replicates of one compound, or parent curation failed. "
                "No source graph, coordinate or activity was changed or merged."
            )
        from sarscope.analysis.fields import coordinates
        from sarscope.curate import CurationResult, CurationStep

        for mol in molecules:
            coordinates(mol)  # Require real finite 3D input, not a mode that skips preparation.
        require_closed_shell(molecules, measurements.molecule_id.tolist())
        kept = measurements.drop(columns="record_id").assign(
            smiles=graphs, n_measurements=1, pactivity_range=0.0
        )
        curated = CurationResult(
            kept,
            [
                CurationStep(
                    "reviewed_input_states",
                    len(kept),
                    len(kept),
                    len(kept),
                    "Input protonation/tautomer/graphs and pose retained; "
                    "no state inference or merging",
                )
            ],
        )
    else:
        curated = curate_table(
            measurements, RunParams(curation=CurationParams(canonical_tautomer=False))
        )
    if not curated.rejected.empty:
        raise ValueError(
            "Curation rejected structures; review them in the main curation workflow first."
        )
    by_smiles: dict[str, Any] = {}
    for index, mol in enumerate(parsed):
        key = Chem.MolToSmiles(mol)
        geometry = molecules[index] if molecules else mol
        if Chem.MolToSmiles(Chem.RemoveHs(geometry)) != key:
            raise ValueError(
                "The selected SMILES do not match the SDF coordinate-bearing structure."
            )
        if molecules and key in by_smiles:
            a, b = Chem.RemoveHs(by_smiles[key]), Chem.RemoveHs(geometry)
            if (
                a.GetNumConformers() != b.GetNumConformers()
                or not a.GetNumConformers()
                or not np.allclose(a.GetConformer().GetPositions(), b.GetConformer().GetPositions())
            ):
                raise ValueError(
                    "Repeated structures have different poses; "
                    "supply one reviewed pose per compound."
                )
        by_smiles.setdefault(key, geometry)
    output = []
    for row in curated.table.itertuples():
        if row.smiles not in by_smiles:
            raise ValueError(
                "Standardisation changed a molecular graph. Review parent/charge states first; "
                "coordinates were not transferred to a different structure."
            )
        output.append(Chem.Mol(by_smiles[row.smiles]))
    prescribed = {}
    if split_col is not None:
        if not preserve_input_states:
            raise ValueError("A supplied partition requires unique, reviewed input-state mode.")
        if split_col not in raw or train_value == test_value:
            raise ValueError("Choose a present partition column and distinct train/test values.")
        membership = raw[split_col].astype(str).str.strip()
        if raw[split_col].isna().any() or not membership.isin([train_value, test_value]).all():
            raise ValueError("Every partition value must be explicitly training or held-out.")
        prescribed = {
            "train_ids": measurements.loc[
                membership.eq(train_value).to_numpy(), "molecule_id"
            ].tolist(),
            "test_ids": measurements.loc[
                membership.eq(test_value).to_numpy(), "molecule_id"
            ].tolist(),
            "column": split_col,
            "train_value": train_value,
            "test_value": test_value,
            "policy": "Supplied membership, not independently validated or selected by scores",
        }
        if not prescribed["train_ids"] or not prescribed["test_ids"]:
            raise ValueError("The supplied partition needs both training and held-out compounds.")
    return FieldDataset(
        curated.table,
        output,
        measurements,
        pd.DataFrame([asdict(step) for step in curated.steps]),
        {
            "kind": "upload",
            "sha256": source_sha256,
            "endpoint": endpoint,
            "input_unit": unit,
            "conversion": "unchanged -log10(M)"
            if unit == "p"
            else f"-log10(value in {unit} converted to M)",
            "assay_comparability": "asserted by uploader, not independently verified",
            "canonical_tautomer": False,
            "preserve_input_states": preserve_input_states,
            **({"prescribed_split": prescribed} if prescribed else {}),
        },
    )


def prescribed_split_indices(dataset: FieldDataset) -> tuple[list[int], list[int]]:
    """Map declared IDs, preserving order; refuse silent removal from a frozen partition."""
    membership = dataset.source.get("prescribed_split")
    if not isinstance(membership, dict):
        raise ValueError("This dataset has no supplied training/test partition.")
    train = list(map(str, membership.get("train_ids", [])))
    test = list(map(str, membership.get("test_ids", [])))
    ids = dataset.table.molecule_id.astype(str).tolist()
    if not train or not test or len(set(train + test)) != len(train + test):
        raise ValueError("Supplied partition IDs must be non-empty, unique and non-overlapping.")
    if len(set(ids)) != len(ids) or set(ids) != set(train + test):
        raise ValueError(
            "The supplied partition needs the complete declared compound set. "
            "Restore All compounds/select all, or explicitly choose a new random/scaffold split."
        )
    position = {identifier: index for index, identifier in enumerate(ids)}
    return [position[i] for i in train], [position[i] for i in test]


def subset_dataset(dataset: FieldDataset, identifiers: list[str]) -> FieldDataset:
    if not identifiers:
        raise ValueError("Select a chemical series first.")
    keep = np.flatnonzero(dataset.table["molecule_id"].isin(identifiers).to_numpy())
    return FieldDataset(
        dataset.table.iloc[keep].reset_index(drop=True),
        [dataset.molecules[int(i)] for i in keep],
        dataset.measurements[dataset.measurements["molecule_id"].isin(identifiers)].copy(),
        dataset.accounting.copy(),
        dataset.source.copy(),
    )


def prepare_field_series(dataset: FieldDataset, params: FieldQsarParams) -> PreparedFieldSeries:
    identifiers = dataset.table["molecule_id"].astype(str).tolist()
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Curated molecule identifiers must be unique.")
    aligned = align_series(dataset.molecules, identifiers, params)
    return PreparedFieldSeries(dataset, aligned, params, asdict(params))


def _pooled_q2(truth: np.ndarray, prediction: np.ndarray) -> float:
    valid = np.isfinite(prediction)
    denominator = float(np.sum((truth[valid] - truth.mean()) ** 2))
    return (
        1 - float(np.sum((truth[valid] - prediction[valid]) ** 2)) / denominator
        if denominator
        else float("nan")
    )


def calibration_diagnostics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | bool]:
    """Report origin-regression heuristics, never equate them with validation.

    k fits observed = k*predicted; k_prime reverses that relation. R0 values
    use centred observed SST, not squared uncentred Pearson correlation.
    See Golbraikh & Tropsha, doi:10.1016/S1093-3263(01)00123-1.
    """
    if len(truth) < 3 or np.std(truth) == 0 or np.std(prediction) == 0:
        return {
            "pearson_r2": float("nan"),
            "k": float("nan"),
            "k_prime": float("nan"),
            "r0_squared": float("nan"),
            "r0_prime_squared": float("nan"),
            "origin_calibration_heuristic": False,
        }
    r2 = float(np.corrcoef(truth, prediction)[0, 1] ** 2)
    k = float(truth @ prediction / (prediction @ prediction))
    kp = float(truth @ prediction / (truth @ truth))
    r0 = 1 - float(np.sum((truth - k * prediction) ** 2) / np.sum((truth - truth.mean()) ** 2))
    r0p = 1 - float(
        np.sum((prediction - kp * truth) ** 2) / np.sum((prediction - prediction.mean()) ** 2)
    )
    passes = r2 > 0.6 and (
        ((r2 - r0) / r2 < 0.1 and 0.85 <= k <= 1.15)
        or ((r2 - r0p) / r2 < 0.1 and 0.85 <= kp <= 1.15)
    )
    return {
        "pearson_r2": r2,
        "k": k,
        "k_prime": kp,
        "r0_squared": r0,
        "r0_prime_squared": r0p,
        "origin_calibration_heuristic": passes,
    }


def field_split_plan(
    table: pd.DataFrame,
    model: ModelParams,
    split_indices: tuple[list[int], list[int]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Validate the hold-out and every nested fitting partition before fields/PLS.

    Fold indices in the returned list are relative to the full input table.
    No labels, geometry or test scores are used to choose a different split.
    """
    table = add_scaffolds(table)
    groups = np.array(
        [g if g is not None else f"acyclic_{i}" for i, g in enumerate(table["murcko"])]
    )
    labels = np.full(len(table), "series", dtype=object)
    if model.split == "scaffold":
        count = len(np.unique(groups))
        if count == 1:
            raise ValueError("All compounds share one scaffold. Use a random split.")
        holdout_folds = max(2, round(1 / model.test_fraction))
        if split_indices is None and count < holdout_folds:
            raise ValueError(
                f"The {model.test_fraction:.0%} held-out scaffold split needs at least "
                f"{holdout_folds} scaffolds; found {count}. This is separate from the "
                f"{model.cv_folds} CV folds. Use a random split or a larger multi-scaffold series."
            )
    train, test = regression_split(
        np.zeros((len(table), 1)), labels, groups, model, split_indices=split_indices
    )
    if len(train) < 15 or len(test) < 3:
        raise ValueError(
            f"This split leaves {len(train)} training and {len(test)} held-out compounds. "
            "At least 15 training and 3 held-out compounds are needed. "
            "Reduce the held-out fraction to keep more training compounds, "
            "add comparable compounds, or use a random split for an uneven scaffold series."
        )

    def folds(indices: np.ndarray, context: str) -> list[tuple[np.ndarray, np.ndarray]]:
        n = len(np.unique(groups[indices])) if model.split == "scaffold" else len(indices)
        if model.regression_cv != "loo" and n < model.cv_folds:
            noun = "scaffolds" if model.split == "scaffold" else "compounds"
            raise ValueError(
                f"{context} has {n} {noun}, fewer than the {model.cv_folds} CV folds. "
                "Use fewer CV folds, a random split, or a larger comparable series."
            )
        return [
            (indices[a], indices[b])
            for a, b in _regression_folds(labels[indices], groups[indices], model)
        ]

    partitions = [(train, test)]
    outer = folds(train, "The training partition")
    partitions.extend(outer)
    # Final component selection uses these same folds of the full training set.
    for number, (fitting, _) in enumerate(outer, 1):
        inner = folds(fitting, f"Outer CV fold {number}'s component-selection partition")
        if any(len(a) < 5 for a, _ in inner):
            raise ValueError(
                "Nested CV leaves fewer than 5 fitting compounds in an inner fold. "
                "Add comparable compounds or reduce the held-out fraction; "
                "the n/5 PLS component cap cannot be bypassed."
            )
        partitions.extend(inner)
    return train, test, partitions


def _check_field_partitions(
    matrix: np.ndarray,
    partitions: list[tuple[np.ndarray, np.ndarray]],
    params: FieldQsarParams,
) -> None:
    """Fail before fitting any model, using each fitting partition's own mask."""
    if params.method != "comfa" or params.missing_field_policy != "refuse":
        return
    for number, (fit, validation) in enumerate(partitions):
        filt = FieldColumnFilter(params).fit(matrix[fit])
        try:
            filt.transform(matrix[validation])
        except ValueError as exc:
            context = "held-out partition" if number == 0 else "nested CV partition"
            raise ValueError(f"Early CoMFA check failed in the {context}: {exc}") from exc


def run_field_qsar(
    prepared: PreparedFieldSeries,
    model: ModelParams,
    *,
    split_indices: tuple[list[int], list[int]] | None = None,
) -> FieldQsarResult:
    """Use SARscope's regression engine for fields AND an identical-split 2D baseline.

    Grid bounds and undefined-column filters are training derived. The random
    series split measures interpolation, not a prospective or new-scaffold
    prediction. Repeatedly tuning to this test set invalidates its hold-out role.
    """
    params = prepared.params
    if params.component_sample_ratio is None or params.component_sample_ratio < 5:
        raise ValueError(
            "Native field validation requires the n/5 component cap. "
            "Uncapped historical reproduction is benchmark-only, not a validated app model."
        )
    source_quality = require_closed_shell(
        prepared.alignment.molecules,
        prepared.dataset.table["molecule_id"].astype(str).tolist(),
    )
    table = add_scaffolds(prepared.dataset.table)
    if len(table) < 20 or len(table) > params.max_molecules:
        raise ValueError(f"Use 20–{params.max_molecules} unique compounds for nested field QSAR.")
    y = table["pactivity"].to_numpy(dtype=float)
    if not np.isfinite(y).all() or np.ptp(y) < 1e-8:
        raise ValueError("Continuous, varying activities are required.")
    if model.split not in ("random", "scaffold"):
        raise ValueError(
            "This experimental field workflow supports random or scaffold splits only."
        )
    model = replace(
        model,
        field_qsar=params,
        regression_algorithms=("field_pls",),
        oversample=False,
        leakage_audit=False,
    )
    labels = np.full(len(table), "series", dtype=object)
    groups = np.array(
        [g if g is not None else f"acyclic_{i}" for i, g in enumerate(table["murcko"])]
    )
    train, test, partitions = field_split_plan(table, model, split_indices)
    grid = GridSpec.around([prepared.alignment.molecules[int(i)] for i in train], params)
    fields = compute_fields(prepared.alignment.molecules, grid, params)
    _check_field_partitions(fields.matrix, partitions, params)
    for index, charge in enumerate(fields.charges):
        charge["molecule_id"] = str(table.iloc[index].molecule_id)
    result = evaluate_regression(
        fields.matrix, y, groups.tolist(), labels.tolist(), model, split_indices=split_indices
    )
    baseline_params = replace(
        model,
        field_qsar=None,
        features=FeatureParams(variance_threshold=0.0),
        regression_algorithms=("random_forest",),
    )
    fingerprint = fingerprint_matrix(table["smiles"].tolist(), "ecfp4")
    baseline = evaluate_regression(
        fingerprint,
        y,
        groups.tolist(),
        labels.tolist(),
        baseline_params,
        split_indices=split_indices,
    )
    if not np.array_equal(result.train_index, baseline.train_index) or not np.array_equal(
        result.test_index, baseline.test_index
    ):
        raise RuntimeError("2D and 3D splits differ; comparison refused.")
    assert result.cv_predictions is not None and baseline.cv_predictions is not None
    q2 = _pooled_q2(y[train], result.cv_predictions)
    rng = np.random.default_rng(params.seed)
    scrambled = []
    permutation_deadline = time.monotonic() + params.permutation_timeout
    for number in range(params.permutations):
        if time.monotonic() > permutation_deadline:
            raise ValueError(
                "Permutation time budget exceeded; reduce permutations or the series size."
            )
        shuffled = rng.permutation(y[train])
        prediction, _, _ = regression_cv_predictions(
            "field_pls", fields.matrix[train], shuffled, labels[train], groups[train], model
        )
        scrambled.append({"permutation": number + 1, "q2": _pooled_q2(shuffled, prediction)})
    pvalue = (
        (1 + sum(row["q2"] >= q2 for row in scrambled)) / (1 + len(scrambled))
        if scrambled
        else float("nan")
    )
    filt = result.fitted_filter
    domain = pca_bounding_box(
        filt.transform(fields.matrix[train]), filt.transform(fields.matrix[test])
    )
    predictions = table[["molecule_id", "smiles", "pactivity"]].copy()
    predictions["split"] = "train"
    predictions.loc[test, "split"] = "held_out"
    predictions["prediction"] = np.asarray(
        result.fitted_model.predict(filt.transform(fields.matrix))
    ).reshape(-1)
    predictions["residual"] = predictions["pactivity"] - predictions["prediction"]
    predictions["missing_retained_fields"] = filt.missing_counts(fields.matrix)
    predictions["mean_filled_fraction"] = (
        predictions["missing_retained_fields"] / int(filt.support_.sum())
        if params.missing_field_policy == "training_mean"
        else 0.0
    )
    electro_columns = (
        np.array([label.startswith("electrostatic_") for label in fields.labels]) & filt.support_
    )
    predictions["missing_electrostatic_fraction"] = (
        np.isnan(fields.matrix[:, electro_columns]).mean(axis=1) if electro_columns.any() else 0.0
    )
    predictions["mean_fill_within_limit"] = (
        predictions["mean_filled_fraction"] <= params.max_imputed_fraction
    ) & (predictions["missing_electrostatic_fraction"] <= params.max_imputed_fraction)
    predictions["training_oof_prediction"] = np.nan
    predictions.loc[train, "training_oof_prediction"] = result.cv_predictions
    predictions["baseline_prediction"] = np.nan
    predictions.loc[test, "baseline_prediction"] = baseline.test_predictions
    predictions["pca_box_in_domain"] = pd.Series(pd.NA, index=predictions.index, dtype="boolean")
    predictions.loc[test, "pca_box_in_domain"] = domain.in_domain
    grid_low = np.array(grid.origin)
    grid_high = grid_low + (np.array(grid.shape) - 1) * grid.spacing
    predictions["outside_training_grid"] = [
        bool(
            np.any(Chem.RemoveHs(mol).GetConformer().GetPositions() < grid_low - 1e-8)
            or np.any(Chem.RemoveHs(mol).GetConformer().GetPositions() > grid_high + 1e-8)
        )
        for mol in prepared.alignment.molecules
    ]
    diagnostics = calibration_diagnostics(y[test], result.test_predictions)
    field_stats = result.scores[result.scores["algorithm"] == "field_pls"].copy()
    field_stats["cv_pooled_q2"] = q2
    field_stats["permutation_pvalue"] = pvalue
    field_stats["selected_components"] = result.fitted_model.selected_components_
    for key, value in diagnostics.items():
        field_stats[key] = value
    base_stats = baseline.scores.copy()
    base_stats["cv_pooled_q2"] = [
        _pooled_q2(y[train], baseline.cv_predictions) if name == baseline.best_algorithm else np.nan
        for name in base_stats["algorithm"]
    ]
    checks = pd.DataFrame(
        [
            {"check": "Nested training CV q² > 0.5", "passed": bool(q2 > 0.5), "value": q2},
            {
                "check": "Held-out predictive R² > 0.6",
                "passed": bool(field_stats.iloc[0].test_r2 > 0.6),
                "value": float(field_stats.iloc[0].test_r2),
            },
            {
                "check": "Origin calibration (GT-style heuristic; value = Pearson r²)",
                "passed": bool(diagnostics["origin_calibration_heuristic"]),
                "value": diagnostics["pearson_r2"],
                "criterion": (
                    "Pearson r² > 0.6 AND an origin fit: relative gap < 0.1, slope 0.85–1.15"
                ),
                "details": (
                    f"k={diagnostics['k']:.3g}; k′={diagnostics['k_prime']:.3g}; "
                    f"R₀²={diagnostics['r0_squared']:.3g}; "
                    f"R₀′²={diagnostics['r0_prime_squared']:.3g}. "
                    "Pearson r² is not predictive R²."
                ),
            },
            {
                "check": "Full-pipeline permutation test",
                "passed": bool(pvalue <= params.permutation_alpha),
                "value": pvalue,
                "criterion": f"Training-label permutation p ≤ {params.permutation_alpha:g}",
            },
            {
                "check": "At least five held-out compounds",
                "passed": len(test) >= 5,
                "value": len(test),
            },
            {
                "check": "Alignment thresholds met",
                "passed": not prepared.alignment.report["warning"].astype(bool).any()
                and prepared.alignment.common_core_coverage >= params.min_core_coverage,
                "value": prepared.alignment.common_core_coverage,
                "criterion": (
                    f"Common-core coverage ≥ {params.min_core_coverage:.0%}; "
                    f"core RMSD ≤ {params.max_alignment_rmsd:g} Å; no alignment warnings"
                ),
                "details": "Value is the whole-series common heavy-atom core coverage.",
            },
            {
                "check": "Held-out compounds with heavy atoms outside field grid",
                "passed": not predictions.loc[test, "outside_training_grid"].any(),
                "value": int(predictions.loc[test, "outside_training_grid"].sum()),
                "criterion": "0 compounds outside (zero is a pass)",
            },
            {
                "check": "Native end-to-end 3D-QSAR validation",
                "passed": False,
                "value": BENCHMARK_STATUS,
            },
            {
                "check": "Mean-filled held-out fields within configured reliability limit",
                "passed": bool(predictions.loc[test, "mean_fill_within_limit"].all()),
                "value": float(
                    predictions.loc[
                        test, ["mean_filled_fraction", "missing_electrostatic_fraction"]
                    ]
                    .max()
                    .max()
                ),
                "criterion": f"Both missing-field fractions ≤ {params.max_imputed_fraction:.0%}",
            },
        ]
    )
    checks["criterion"] = checks["criterion"].fillna("See check label/configured settings")
    checks["details"] = checks["details"].fillna("")
    failures = checks.loc[~checks["passed"], "check"].tolist()
    status = "Experimental — not validated. Unmet checks: " + "; ".join(failures)
    record = provenance.collect(RunParams(model=model), prepared.dataset.source)
    record.update(
        {
            "field_qsar": asdict(params),
            "source_graph_audit": source_quality.to_dict(orient="records"),
            "model": asdict(model),
            "grid": asdict(grid),
            "field_convention": fields.convention,
            "alignment_assumption": prepared.alignment.assumption,
            "alignment_settings": prepared.alignment_settings,
            "common_core_smarts": prepared.alignment.common_core_smarts,
            "benchmark_status": BENCHMARK_STATUS,
            "feature_mask_policy": (
                "training-only; NaN CoMFA terms replaced with fitting-column means, "
                "all-missing fitting columns dropped; per-row counts/fractions reported"
                if params.missing_field_policy == "training_mean"
                else "training-only; undefined held-out CoMFA fields refuse scoring"
            ),
            "vendor_compatibility": (
                "Independent MMFF94 MIF implementation from public equations/RDKit parameters; "
                "not a port of Open3DQSAR; full external workflow parity not claimed"
                if params.field_definition == "mmff94_mif"
                else "Public method components only; full SYBYL parity not verified"
            ),
            "cv": "nested component selection; pooled q2 = 1 - PRESS / training SST",
            "component_selection_audit": {
                "requested_cap": params.component_cap,
                "effective_cap": result.fitted_model.effective_component_cap_,
                "selected": result.fitted_model.selected_components_,
                "stop_reason": result.fitted_model.selection_stop_reason_,
            },
            "split_interpretation": "supplied/prespecified partition; independence not established"
            if split_indices
            else "within-series interpolation"
            if model.split == "random"
            else "held-out Murcko scaffold groups",
            "split_indices": {"train": train.tolist(), "test": test.tolist()},
            "grid_policy": "Fixed from the complete training cohort; not regenerated per CV "
            "fold. Masks/scales/component selection refitted inside fitting partitions.",
            "permutation_test": "training labels only; repeats nested selection; (1+C)/(1+B)",
            "input_measurements_sha256": hashlib.sha256(
                prepared.dataset.measurements.to_csv(index=False).encode()
            ).hexdigest(),
            "implementation_sha256": {
                str(path.relative_to(Path(__file__).parent)): provenance.file_sha256(path)
                for path in (
                    Path(__file__),
                    Path(__file__).parent / "analysis" / "fields.py",
                    Path(__file__).parent / "analysis" / "align3d.py",
                    Path(__file__).parent / "analysis" / "contours.py",
                    Path(__file__).parent / "analysis" / "field_filter.py",
                    Path(__file__).parent / "analysis" / "regression.py",
                    Path(__file__).parent / "analysis" / "tripos_reference.py",
                    Path(__file__).parent / "params.py",
                )
            },
        }
    )
    statistics = pd.concat([field_stats, base_stats], ignore_index=True)
    statistics["cv_fold_mean_r2"] = statistics["cv_r2"]
    statistics["cv_fold_sd_r2"] = statistics["cv_r2_sd"]
    return FieldQsarResult(
        prepared,
        fields,
        result,
        baseline,
        predictions,
        statistics,
        checks,
        pd.DataFrame(scrambled, columns=["permutation", "q2"]),
        contribution_maps(fields, filt, result.fitted_model, params),
        domain,
        record,
        status,
    )

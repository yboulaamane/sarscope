"""Assay-scoped one-position analogue proposals, never claims of optimisation.

Reuse R-group reconstruction, controlled SAR pairs, descriptors, cliff detection
and the existing 2D prediction/CV engine. No receptor or 3D model is involved.
Observed substitution effects are not assigned as new-compound potencies.
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.descriptors import compute_descriptors
from sarscope.analysis.features import bit_vectors, fingerprint_matrix
from sarscope.analysis.landscape import sas_map
from sarscope.analysis.regression import (
    RegressionResult,
    evaluate_regression,
    fit_deployment_model,
    regression_metrics,
)
from sarscope.analysis.representations import model_feature_names, model_matrix, resolved_features
from sarscope.analysis.rgroups import ScaffoldSar, select_core_state
from sarscope.analysis.sar_map import _disjoint_count, build_sar_map, filter_series
from sarscope.analysis.sar_stress import assemble_analogue
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_qsar import assay_contexts
from sarscope.params import (
    AnalogueDesignParams,
    FeatureParams,
    LandscapeParams,
    ModelParams,
    RunParams,
)
from sarscope.predict import PredictionBundle, _row_nearest_tanimoto, similarity_domain_threshold
from sarscope.prioritise import predict_candidates

CANDIDATE_COLUMNS = [
    "molecule_id",
    "smiles",
    "parent_id",
    "position",
    "fragment_from",
    "fragment_to",
    "supporting_pairs",
    "disjoint_pairs",
    "observed_median_delta",
    "observed_q25_delta",
    "observed_q75_delta",
    "direction_agreement",
    "contradictory_effects",
    "known_ids",
    "observed_pactivity",
    "unresolved_stereo",
    "structural_alerts",
    "prediction_status",
]


@dataclass
class AnalogueDesign:
    parent: pd.DataFrame
    candidates: pd.DataFrame
    evidence_pairs: pd.DataFrame
    accounting: pd.DataFrame
    provenance: dict[str, Any]
    measurements: pd.DataFrame = field(default_factory=pd.DataFrame)
    scope_table: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class DesignModel:
    bundle: PredictionBundle
    regression: RegressionResult
    scope_signature: str
    checks: pd.DataFrame
    settings: dict[str, Any]


def scope_signature(table: pd.DataFrame, evidence: pd.DataFrame) -> str:
    """Bind structures, activities and retained assay records, not only IDs."""
    records = evidence.copy()
    columns = sorted(
        set(records.columns)
        & {
            "molecule_id",
            "assay_chembl_id",
            "standard_type",
            "assay_variant_mutation",
            "record_id",
            "standard_relation",
            "pactivity",
        }
    )
    digest = hashlib.sha256()
    for frame in (table[["molecule_id", "smiles", "pactivity"]], records[columns]):
        frame = frame.fillna("").astype(str).drop_duplicates()
        digest.update(frame.sort_values(list(frame.columns)).to_csv(index=False).encode())
    return digest.hexdigest()


def _check_scope(table: pd.DataFrame, evidence: pd.DataFrame, params: AnalogueDesignParams) -> None:
    if not 1 <= len(table) <= params.max_scope_compounds:
        raise ValueError(f"Choose 1–{params.max_scope_compounds} assay-scoped compounds.")
    if table.molecule_id.duplicated().any() or len(assay_contexts(evidence)) != 1:
        raise ValueError("Use unique compound IDs and exactly one assay/endpoint/variant context.")
    for name in ("assay_chembl_id", "standard_type", "record_id", "standard_relation"):
        if name not in evidence or evidence[name].isna().any() or evidence[name].eq("").any():
            raise ValueError("Retain complete assay/endpoint/relation/record identifiers.")
    if not np.isfinite(pd.to_numeric(table.pactivity, errors="raise")).all():
        raise ValueError("Finite measured activities are required.")
    if "standard_relation" in evidence and not evidence.standard_relation.eq("=").all():
        raise ValueError("Censored measurements cannot support analogue potency effects.")
    if set(table.molecule_id) != set(evidence.molecule_id):
        raise ValueError("Scope table and retained evidence must identify the same compounds.")
    if table.smiles.map(_canonical).duplicated().any():
        raise ValueError("Merge duplicate isomeric structures and their evidence before designing.")
    records = evidence.copy()
    records["pactivity"] = pd.to_numeric(records.pactivity, errors="raise")
    if not np.isfinite(records.pactivity).all():
        raise ValueError("Retained evidence must contain finite exact activities.")
    if "record_id" in records:
        if records.groupby("record_id")[["molecule_id", "pactivity"]].nunique().gt(1).any().any():
            raise ValueError(
                "A retained record identifier has conflicting compound/activity values."
            )
        records = records.drop_duplicates(["molecule_id", "record_id", "pactivity"])
    medians = records.groupby("molecule_id").pactivity.median()
    if not np.allclose(table.pactivity, table.molecule_id.map(medians), atol=1e-9, rtol=0):
        raise ValueError("Reaggregate activities within the selected assay before designing.")


def _canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid molecular structure in the design scope.")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


@lru_cache(maxsize=1)
def _alert_catalog() -> Any:
    from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

    settings = FilterCatalogParams()
    settings.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
    settings.AddCatalog(FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog(settings)


def _properties(smiles: str) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return {
        **compute_descriptors(mol),
        "structural_alerts": "; ".join(
            sorted({m.GetDescription() for m in _alert_catalog().GetMatches(mol)})
        ),
        "unresolved_stereo": any(
            item.specified != Chem.StereoSpecified.Specified
            for item in Chem.FindPotentialStereo(mol)
        ),
    }


def generate_analogues(
    series: ScaffoldSar,
    table: pd.DataFrame,
    evidence: pd.DataFrame,
    parent_id: str,
    positions: list[str],
    params: AnalogueDesignParams,
    *,
    source: dict[str, Any] | None = None,
) -> AnalogueDesign:
    """Observed-fragment recombination at ONE selected R-position per candidate.

    The mapped core state and every other fragment remain unchanged. Attempts,
    duplicates, unsupported reconstructions and truncation are accounted for.
    """
    _check_scope(table, evidence, params)
    selected = series.members[series.members.molecule_id.eq(parent_id)]
    if len(selected) != 1 or parent_id not in set(table.molecule_id):
        raise ValueError("Choose one parent present in the selected assay and series.")
    if not positions or not set(positions) <= set(series.positions):
        raise ValueError("Choose one or more existing editable R-positions.")
    parent = selected.iloc[0]
    required = {"core_smiles", "rgroup_fragments", "molecule_smiles"}
    if not required <= set(parent.index):
        raise ValueError("Rebuild R-group decomposition to retain exact reconstruction metadata.")
    state = select_core_state(series, str(parent.core_smiles))
    state, retained = filter_series(state, evidence, {})
    structures = table.set_index("molecule_id").smiles.map(_canonical)
    if any(
        _canonical(row.molecule_smiles) != structures.loc[row.molecule_id]
        for row in state.members.itertuples()
    ):
        raise ValueError(
            "Prepared series structures differ from scoped assay records; rebuild SAR."
        )
    mapping = build_sar_map(state, retained, max_pairs=params.max_pairs)
    pairs = mapping.pairs[mapping.pairs.evidence_tier.eq("shared assay")].copy()
    original = _canonical(str(parent.molecule_smiles))
    scoped_parent = table.set_index("molecule_id").loc[parent_id]
    if original != _canonical(str(scoped_parent.smiles)):
        raise ValueError("The parent structure differs from its scoped assay record; rebuild SAR.")
    properties = _properties(original)
    measured_parent = float(table.set_index("molecule_id").loc[parent_id, "pactivity"])
    parent_table = pd.DataFrame(
        [
            {
                "molecule_id": parent_id,
                "smiles": original,
                "observed_pactivity": measured_parent,
                **properties,
            }
        ]
    )
    known: dict[str, list[Any]] = {}
    for record in table.itertuples():
        known.setdefault(_canonical(record.smiles), []).append(record)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    supporting: list[pd.DataFrame] = []
    seen = {original}
    attempted = 0
    truncated = False
    for position in sorted(set(positions)):
        before = str(parent[position])
        direct = pairs[(pairs.position == position) & (pairs.fragment_from == before)].copy()
        reverse = pairs[(pairs.position == position) & (pairs.fragment_to == before)].copy()
        if not reverse.empty:
            for left, right in (
                ("fragment_from", "fragment_to"),
                ("id_from", "id_to"),
                ("pactivity_from", "pactivity_to"),
                ("record_ids_from", "record_ids_to"),
            ):
                reverse[[left, right]] = reverse[[right, left]].to_numpy()
            reverse["delta_pactivity"] = -reverse.delta_pactivity
        directed = pd.concat([direct, reverse], ignore_index=True)
        for fragment, block in directed.groupby("fragment_to", sort=True):
            if attempted >= params.max_attempts or len(rows) >= params.max_candidates:
                truncated = True
                break
            attempted += 1
            entry = {"position": position, "fragment_from": before, "fragment_to": fragment}
            parts = [Chem.MolFromSmiles(f) for f in (before, fragment) if f != "H"]
            if any(
                m is None
                or sum(a.GetAtomicNum() > 1 for a in m.GetAtoms()) > params.max_fragment_heavy_atoms
                or sum(a.GetAtomicNum() == 0 for a in m.GetAtoms()) != 1
                for m in parts
            ):
                audits.append({**entry, "status": "outside single-attachment fragment budget"})
                continue
            smiles = assemble_analogue(parent, position, str(fragment))
            if not smiles:
                audits.append(
                    {**entry, "status": "reconstruction/round-trip failed; no guessed structure"}
                )
                continue
            mol = Chem.MolFromSmiles(smiles)
            if (
                mol is None
                or len(Chem.GetMolFrags(mol)) != 1
                or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
            ):
                audits.append({**entry, "status": "invalid/disconnected/radical structure"})
                continue
            smiles = _canonical(smiles)
            if smiles in seen:
                audits.append({**entry, "status": "duplicate or unchanged structure"})
                continue
            seen.add(smiles)
            identifier = "AN_" + hashlib.sha256(smiles.encode()).hexdigest()[:12]
            effects = block.groupby(["id_from", "id_to"]).delta_pactivity.median()
            hits = known.get(smiles, [])
            prop = _properties(smiles)
            signs = effects.to_numpy()
            rows.append(
                {
                    "molecule_id": identifier,
                    "smiles": smiles,
                    "parent_id": parent_id,
                    **entry,
                    "supporting_pairs": len(effects),
                    "disjoint_pairs": _disjoint_count(block),
                    "observed_median_delta": float(effects.median()),
                    "observed_q25_delta": float(effects.quantile(0.25)),
                    "observed_q75_delta": float(effects.quantile(0.75)),
                    "direction_agreement": float(
                        max((signs > 0).sum(), (signs < 0).sum()) / len(signs)
                    ),
                    "contradictory_effects": bool((signs > 0).any() and (signs < 0).any()),
                    "known_ids": ";".join(str(hit.molecule_id) for hit in hits),
                    "observed_pactivity": float(np.median([hit.pactivity for hit in hits]))
                    if hits
                    else np.nan,
                    **prop,
                    **{
                        f"delta_{name}": prop[name] - properties[name]
                        for name in compute_descriptors(mol)
                    },
                    "prediction_status": "not scored; experimental suggestion only",
                }
            )
            supporting.append(block.assign(candidate_id=identifier))
            audits.append({**entry, "candidate_id": identifier, "status": "generated"})
        if truncated:
            break
    candidates = pd.DataFrame(rows) if rows else pd.DataFrame(columns=CANDIDATE_COLUMNS)
    if not candidates.empty:
        reference_X = fingerprint_matrix(table.smiles.tolist(), "ecfp4")
        query_X = fingerprint_matrix(candidates.smiles.tolist(), "ecfp4")
        similarity, nearest = _row_nearest_tanimoto(query_X, reference_X)
        candidates["nearest_measured_id"] = table.iloc[nearest].molecule_id.to_numpy()
        candidates["nearest_measured_pactivity"] = table.iloc[nearest].pactivity.to_numpy()
        candidates["nearest_measured_similarity"] = similarity
        landscape = sas_map(
            bit_vectors(table.smiles.tolist(), "ecfp4"),
            table.pactivity.tolist(),
            table.molecule_id.tolist(),
            LandscapeParams(
                fingerprints=("ecfp4",),
                similarity_threshold=params.cliff_similarity,
                activity_threshold=params.cliff_activity,
            ),
            "ecfp4",
        )
        cliff_pairs = pd.concat(
            [
                landscape.cliffs,
                landscape.identical_pairs[landscape.identical_pairs.delta > params.cliff_activity],
            ]
        )
        cliff_ids = set(cliff_pairs.id_a) | set(cliff_pairs.id_b)
        cliff_reference = table[table.molecule_id.isin(cliff_ids)]
        candidates["near_activity_cliff"] = False
        candidates["nearest_cliff_id"] = ""
        if not cliff_reference.empty:
            sims, neighbours = _row_nearest_tanimoto(
                query_X, reference_X[table.molecule_id.isin(cliff_ids)]
            )
            candidates["near_activity_cliff"] = sims >= params.cliff_similarity
            candidates["nearest_cliff_id"] = cliff_reference.iloc[neighbours].molecule_id.to_numpy()
    record = provenance.collect(RunParams(), source or {})
    # Version metadata is shared; unrelated default pipeline settings are not a run record.
    record["params"] = {"analogue_design": asdict(params)}
    record.update(
        {
            "scope_signature": scope_signature(table, evidence),
            "context": assay_contexts(evidence).to_dict("records"),
            "settings": asdict(params),
            "parent_id": parent_id,
            "editable_positions": positions,
            "core_smiles": str(parent.core_smiles),
            "one_position_per_candidate": True,
            "attempted": attempted,
            "generated": len(candidates),
            "truncated": truncated,
            "pair_limit_reached": mapping.pair_limit_reached,
            "interpretation": (
                "Observed pair effects are descriptive evidence, not candidate potency "
                "predictions. Not a validated lead optimiser. Known IDs refer only to the "
                "supplied assay corpus, not global novelty."
            ),
            "synthetic_feasibility": (
                "Not established. Valid valence and PAINS/Brenk alerts do not establish "
                "synthesizability, ADME, safety or biological activity."
            ),
            "implementation_sha256": provenance.file_sha256(Path(__file__)),
        }
    )
    return AnalogueDesign(
        parent_table,
        candidates,
        pd.concat(supporting, ignore_index=True)
        if supporting
        else pairs.iloc[0:0].assign(candidate_id=""),
        pd.DataFrame(
            audits, columns=["position", "fragment_from", "fragment_to", "candidate_id", "status"]
        ),
        record,
        evidence.copy(),
        table.copy(),
    )


def model_checks(result: RegressionResult, params: AnalogueDesignParams) -> pd.DataFrame:
    """Conservative release-screen gates, NOT a calibrated validation certificate."""
    score = result.scores[result.scores.algorithm.eq(result.best_algorithm)].iloc[0]
    baseline = result.baseline_test_predictions
    baseline_rmse = (
        float(np.sqrt(np.mean((result.test_truth - baseline) ** 2)))
        if baseline is not None
        else np.nan
    )
    cv_q2 = np.nan
    # CV truth is retained by the caller in the selected score's pooled field below.
    if "design_pooled_cv_q2" in score:
        cv_q2 = float(score.design_pooled_cv_q2)
    values = [
        (
            "Held-out compounds",
            len(result.test_truth),
            len(result.test_truth) >= params.minimum_test_compounds,
        ),
        ("Held-out R²", float(score.test_r2), score.test_r2 >= params.minimum_test_r2),
        ("Pooled training-CV q²", cv_q2, cv_q2 >= params.minimum_cv_q2),
        (
            "Test RMSE below training-mean baseline",
            float(score.test_rmse),
            bool(score.test_rmse < baseline_rmse),
        ),
        (
            "Finite positive training-CV residual band",
            result.empirical_half_width,
            bool(np.isfinite(result.empirical_half_width) and result.empirical_half_width > 0),
        ),
        (
            "Selected estimator is not the mean baseline",
            result.best_algorithm,
            result.best_algorithm != "mean_baseline",
        ),
    ]
    return pd.DataFrame(values, columns=["check", "value", "passed"])


def fit_design_model(
    table: pd.DataFrame,
    evidence: pd.DataFrame,
    params: AnalogueDesignParams,
    model: ModelParams,
    *,
    canonical_tautomer: bool = False,
) -> DesignModel:
    _check_scope(table, evidence, params)
    if len(table) < 30 or model.field_qsar is not None or model.split not in ("random", "scaffold"):
        raise ValueError(
            "Use 30–500 compounds and a 2D random/scaffold model; no 3D model is scored here."
        )
    model = replace(
        model, features=FeatureParams(fingerprint="ecfp4"), oversample=False, leakage_audit=False
    )
    table = add_scaffolds(table)
    X = model_matrix(table.smiles.tolist(), model.features)
    with threadpool_limits(limits=1):
        result = evaluate_regression(
            X, table.pactivity.tolist(), table.murcko.tolist(), table.activity_class.tolist(), model
        )
        if result.cv_predictions is not None:
            result.scores["design_pooled_cv_q2"] = regression_metrics(
                table.iloc[result.train_index].pactivity.to_numpy(),
                result.cv_predictions,
            )[0]
        filt, estimator = fit_deployment_model(
            X, table.pactivity.tolist(), result.best_algorithm, model
        )
    bundle = PredictionBundle(
        result.best_algorithm,
        resolved_features(model.features),
        filt,
        estimator,
        X,
        similarity_domain_threshold(X),
        train_ids=table.molecule_id.tolist(),
        train_smiles=table.smiles.tolist(),
        train_pactivity=table.pactivity.tolist(),
        canonical_tautomer=canonical_tautomer,
        empirical_half_width=result.empirical_half_width,
        feature_names=model_feature_names(model.features),
    )
    return DesignModel(
        bundle,
        result,
        scope_signature(table, evidence),
        model_checks(result, params),
        {**asdict(model), "canonical_tautomer": canonical_tautomer},
    )


def score_analogues(design: AnalogueDesign, model: DesignModel) -> AnalogueDesign:
    """Withhold potency for weak models, changed graph states and out-of-domain rows."""
    if design.provenance["scope_signature"] != model.scope_signature:
        raise ValueError(
            "Model and proposal assay/structure/measurement scopes differ; refit explicitly."
        )
    result = replace(design, candidates=design.candidates.copy(), provenance={**design.provenance})
    for name in (
        "predicted_pactivity",
        "predicted_delta_vs_parent",
        "empirical_lower_pactivity",
        "empirical_upper_pactivity",
        "model_parent_prediction",
    ):
        result.candidates[name] = np.nan
    result.candidates["in_applicability_domain"] = False
    result.provenance["model_checks"] = model.checks.assign(
        value=model.checks.value.astype(str)
    ).to_dict("records")
    result.provenance["model_settings"] = model.settings
    result.provenance["prediction_interpretation"] = (
        "Predicted delta = candidate prediction minus the SAME model's parent prediction, "
        "never minus measured parent potency. Residual bands are approximate level bands, "
        "not gain intervals or calibrated probabilities. All-data deployment refit follows "
        "the unchanged held-out evaluation."
    )
    if not model.checks.passed.all():
        result.candidates["prediction_status"] = "withheld: model failed qualification checks"
        return result
    if result.candidates.empty:
        return result
    source = pd.concat(
        [design.parent[["molecule_id", "smiles"]], result.candidates[["molecule_id", "smiles"]]],
        ignore_index=True,
    )
    scored = predict_candidates(model.bundle, source)
    parent = scored.iloc[0]
    parent_state = _canonical(design.parent.iloc[0].smiles)
    if (
        parent.status != "predicted"
        or parent.canonical_smiles != parent_state
        or not bool(parent.in_applicability_domain)
        or bool(design.parent.iloc[0].unresolved_stereo)
    ):
        result.candidates["prediction_status"] = (
            "withheld: parent state changed, ambiguous stereo or outside model domain"
        )
        return result
    for offset, (index, candidate) in enumerate(result.candidates.iterrows(), start=1):
        row = scored.iloc[offset]
        status = "scored: hypothesis, not a confirmed improvement"
        if row.status != "predicted" or row.canonical_smiles != _canonical(candidate.smiles):
            status = "withheld: standardisation changed the proposed structure"
        elif bool(candidate.unresolved_stereo):
            status = "withheld: unresolved stereochemistry"
        elif not bool(row.in_applicability_domain):
            status = "withheld: outside applicability domain"
        elif not np.isfinite([row.predicted_pactivity, parent.predicted_pactivity]).all():
            status = "withheld: non-finite model prediction"
        else:
            result.candidates.loc[index, "predicted_pactivity"] = float(row.predicted_pactivity)
            result.candidates.loc[index, "model_parent_prediction"] = float(
                parent.predicted_pactivity
            )
            result.candidates.loc[index, "predicted_delta_vs_parent"] = float(
                row.predicted_pactivity - parent.predicted_pactivity
            )
            for name in ("empirical_lower_pactivity", "empirical_upper_pactivity"):
                if name in row:
                    result.candidates.loc[index, name] = row[name]
        result.candidates.loc[index, "prediction_status"] = status
        if row.status == "predicted":
            result.candidates.loc[index, "in_applicability_domain"] = bool(
                row.in_applicability_domain
            )
    return result


def design_archive(design: AnalogueDesign, model: DesignModel | None = None) -> bytes:
    files = {
        "candidates.csv": design.candidates.to_csv(index=False).encode(),
        "parent.csv": design.parent.to_csv(index=False).encode(),
        "supporting_pairs.csv": design.evidence_pairs.to_csv(index=False).encode(),
        "candidate_accounting.csv": design.accounting.to_csv(index=False).encode(),
        "retained_measurements.csv": design.measurements.to_csv(index=False).encode(),
        "scoped_compounds.csv": design.scope_table.to_csv(index=False).encode(),
        "provenance.json": json.dumps(design.provenance, indent=2, allow_nan=False).encode(),
    }
    if model is not None:
        files["model_checks.csv"] = model.checks.to_csv(index=False).encode()
        files["model_scores.csv"] = model.regression.scores.to_csv(index=False).encode()
    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    for row in design.candidates.to_dict("records"):
        mol = Chem.MolFromSmiles(row["smiles"])
        mol.SetProp("_Name", row["molecule_id"])
        for name, value in row.items():
            if value is not None and pd.notna(value):
                mol.SetProp(name, str(value))
        writer.write(mol)
    writer.close()
    files["analogues.sdf"] = stream.getvalue().encode()
    files["README.txt"] = (
        b"Experimental design hypotheses, not validated optimised leads. "
        b"Observed pair deltas are NOT candidate potency predictions. "
        b"Known IDs refer to this supplied assay corpus only. "
        b"SDF contains structures, not validated aligned poses. "
        b"PAINS/Brenk alerts are heuristics, not safety or synthesis assessments."
    )
    files["manifest.sha256.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}, indent=2
    ).encode()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()

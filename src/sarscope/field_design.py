"""Predict analogue hypotheses with the displayed, unchanged 3D field-PLS model."""

from __future__ import annotations

import hashlib
import io
import json
import time
import zipfile
from contextlib import ExitStack
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope.analogue_design import AnalogueDesign, design_archive, generate_analogues
from sarscope.analysis.align3d import _AlignmentGenerator, best_mapping, preparation_available
from sarscope.analysis.fields import compute_fields, coordinates
from sarscope.analysis.rgroups import ScaffoldSar
from sarscope.conformers import validate_conformer_stereo
from sarscope.field_qsar import FieldQsarResult
from sarscope.params import AnalogueDesignParams


@dataclass
class FieldDesignResult:
    design: AnalogueDesign
    aligned: list[Any]
    contributions: pd.DataFrame
    charges: pd.DataFrame
    audit: dict[str, Any]


def field_model_signature(result: FieldQsarResult) -> str:
    """Bind model, split, field mask/scaling, raw fields and reference-frame geometry."""
    if getattr(result.domain, "fitted_pca", None) is None:
        raise ValueError("Rerun the 3D model to retain its fitted applicability-domain transform.")
    digest = hashlib.sha256(
        json.dumps(
            {
                "params": asdict(result.prepared.params),
                "grid": asdict(result.fields.grid),
                "labels": result.fields.labels,
                "table": result.prepared.dataset.table[
                    ["molecule_id", "smiles", "pactivity"]
                ].to_csv(index=False),
                "measurements": result.prepared.dataset.measurements.to_csv(index=False),
                "checks": result.checks.to_csv(index=False),
                "status": result.status,
                "residual_half_width": str(result.regression.empirical_half_width),
            },
            sort_keys=True,
        ).encode()
    )
    filt, model = result.regression.fitted_filter, result.regression.fitted_model
    for array in (
        result.fields.matrix,
        result.regression.train_index,
        result.regression.test_index,
        filt.support_,
        filt.column_scale_,
        filt.column_mean_,
        model.coef_,
        model.intercept_,
        getattr(model, "_x_mean", []),
    ):
        digest.update(np.asarray(array).tobytes())
    for mol in result.prepared.alignment.molecules:
        digest.update(Chem.MolToSmiles(Chem.RemoveHs(mol)).encode())
        digest.update(coordinates(mol).tobytes())
    if result.domain.fitted_pca is not None:
        for value in (
            result.domain.fitted_pca.components_,
            result.domain.fitted_pca.mean_,
            result.domain.lower,
            result.domain.upper,
        ):
            digest.update(np.asarray(value).tobytes())
    return digest.hexdigest()


def field_design_inputs(result: FieldQsarResult) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uploads have user-asserted context; never manufacture a ChEMBL assay identity."""
    table = result.prepared.dataset.table.copy()
    if "activity_class" not in table:
        table["activity_class"] = "series"
    evidence = result.prepared.dataset.measurements.copy()
    defaults = {
        "assay_chembl_id": "USER_DECLARED_3D_SERIES",
        "standard_type": result.prepared.dataset.source.get("endpoint", "pActivity"),
        "standard_relation": "=",
        "assay_variant_mutation": None,
    }
    for name, value in defaults.items():
        if name not in evidence:
            evidence[name] = value
    if "record_id" not in evidence:
        evidence["record_id"] = [f"user_3d:{i + 1}" for i in range(len(evidence))]
    return table, evidence


def generate_field_analogues(
    result: FieldQsarResult,
    series: ScaffoldSar,
    parent_id: str,
    positions: list[str],
    *,
    max_candidates: int = 5,
) -> AnalogueDesign:
    if not 1 <= max_candidates <= 10:
        raise ValueError("Generate at most 10 analogues per 3D scoring batch.")
    table, evidence = field_design_inputs(result)
    design = generate_analogues(
        series,
        table,
        evidence,
        parent_id,
        positions,
        AnalogueDesignParams(max_candidates=max_candidates),
        source=result.prepared.dataset.source,
    )
    design.provenance["field_model_signature"] = field_model_signature(result)
    design.provenance["scoring_model"] = (
        "Same displayed training-fitted 3D field PLS; no refit or 2D fallback"
    )
    return design


def predict_field_analogues(
    result: FieldQsarResult,
    design: AnalogueDesign,
    *,
    route: Literal["rigid_mcs", "constrained"] | None = None,
    allow_exploratory: bool = False,
) -> FieldDesignResult:
    """Never refit PLS/filter, rebuild grid or use RF predictions.

    Weak/incompletely validated models require explicit exploratory consent.
    Stereo, alignment, training-grid, missing-field and training-PCA checks
    cannot be bypassed. Failed rows stay in the result with no numeric potency.
    """
    signature = field_model_signature(result)
    if design.provenance.get("field_model_signature") != signature:
        raise ValueError("This design belongs to a different 3D model/alignment; regenerate it.")
    if len(design.candidates) > 10:
        raise ValueError("Use at most 10 candidates per bounded 3D batch.")
    params = result.prepared.params
    if route is None:
        if params.alignment == "prealigned":
            raise ValueError(
                "Choose an explicit candidate alignment route for a pre-aligned model."
            )
        route = params.alignment
    if route not in ("rigid_mcs", "constrained"):
        raise ValueError("Choose rigid_mcs or constrained candidate alignment.")
    settings = replace(
        params,
        alignment=route,
        reference_index=0,
        alignment_timeout=min(120, params.alignment_timeout),
    )
    frame = design.candidates.copy()
    for name in (
        "predicted_pactivity",
        "predicted_delta_vs_parent",
        "model_parent_prediction",
        "empirical_lower_pactivity",
        "empirical_upper_pactivity",
        "core_rmsd_A",
        "core_coverage",
        "mean_filled_fraction",
        "missing_electrostatic_fraction",
    ):
        frame[name] = np.nan
    frame["in_applicability_domain"] = False
    frame["prediction_status"] = "not attempted"
    scored_design = replace(design, candidates=frame, provenance={**design.provenance})
    audit = {
        "model_signature": signature,
        "prediction_model": "3D field PLS",
        "refit": False,
        "2d_fallback": False,
        "grid": asdict(result.fields.grid),
        "fields": asdict(params),
        "candidate_alignment": asdict(settings),
        "model_checks": json.loads(result.checks.to_json(orient="records")),
        "model_status": result.status,
        "exploratory_consent": allow_exploratory,
        "reference_id": result.prepared.dataset.table.iloc[
            result.prepared.alignment.reference_index
        ].molecule_id,
        "interpretation": (
            "Experimental model associations, not validated optimisation or binding interactions. "
            "Delta uses this same model's parent prediction, not observed parent potency. "
            "Residual bands are approximate activity-level bands, not gain intervals."
        ),
    }
    output = FieldDesignResult(scored_design, [], pd.DataFrame(), pd.DataFrame(), audit)
    scored_design.provenance["3d_prediction"] = audit
    if not result.checks.passed.all() and not allow_exploratory:
        frame["prediction_status"] = (
            "withheld: 3D validation checks unmet; exploratory consent required"
        )
        return output
    table = result.prepared.dataset.table
    ids = table.molecule_id.tolist()
    parent_id = design.parent.iloc[0].molecule_id
    if parent_id not in ids:
        raise ValueError("The parent is absent from this fitted 3D series.")
    parent_index = ids.index(parent_id)

    def canonical(smi: str) -> str:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError("Invalid candidate or parent graph")
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    if canonical(design.parent.iloc[0].smiles) != canonical(table.iloc[parent_index].smiles):
        raise ValueError("The proposed parent differs from the fitted molecular graph.")
    filt, estimator = result.regression.fitted_filter, result.regression.fitted_model
    parent_raw = result.fields.matrix[[parent_index]]
    electro_mask = (
        np.array([label.startswith("electrostatic_") for label in result.fields.labels])
        & filt.support_
    )
    parent_missing = float(filt.missing_counts(parent_raw)[0] / int(filt.support_.sum()))
    parent_electro = (
        float(np.isnan(parent_raw[:, electro_mask]).mean()) if electro_mask.any() else 0.0
    )
    if max(parent_missing, parent_electro) > params.max_imputed_fraction:
        frame["prediction_status"] = "withheld: parent missing-field reliability checks failed"
        return output
    try:
        parent_X = filt.transform(result.fields.matrix[[parent_index]])
    except ValueError as exc:
        frame["prediction_status"] = f"withheld: parent field contract failed: {exc}"
        return output
    parent_prediction = float(np.asarray(estimator.predict(parent_X)).reshape(-1)[0])
    if not result.domain.classify(parent_X).in_domain[0]:
        frame["prediction_status"] = "withheld: parent outside training-field PCA box"
        return output
    reference = result.prepared.alignment.molecules[result.prepared.alignment.reference_index]
    known = {canonical(row.smiles): i for i, row in enumerate(table.itertuples())}
    low = np.asarray(result.fields.grid.origin)
    high = low + (np.asarray(result.fields.grid.shape) - 1) * result.fields.grid.spacing
    parent_xyz = coordinates(Chem.RemoveHs(result.prepared.alignment.molecules[parent_index]))
    parent_warning = result.prepared.alignment.report.iloc[parent_index].warning
    if np.any(parent_xyz < low - 1e-8) or np.any(parent_xyz > high + 1e-8) or bool(parent_warning):
        frame["prediction_status"] = "withheld: parent grid/alignment checks failed"
        return output
    coefficient = np.asarray(estimator.coef_).reshape(-1)
    retained_names = np.asarray([label.rsplit("_", 1)[0] for label in result.fields.labels])[
        filt.support_
    ]
    contributions: list[dict[str, Any]] = []
    charge_rows: list[dict[str, Any]] = []
    deadline = time.monotonic() + settings.alignment_timeout
    with ExitStack() as stack:
        worker = None
        for index, row in frame.iterrows():
            try:
                if time.monotonic() >= deadline:
                    raise ValueError("Total candidate alignment/scoring time budget exceeded")
                mol = Chem.MolFromSmiles(row.smiles)
                if mol is None or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                    raise ValueError("Invalid/radical candidate graph")
                if any(
                    info.specified != Chem.StereoSpecified.Specified
                    for info in Chem.FindPotentialStereo(mol)
                ):
                    raise ValueError("Resolve unspecified R/S or E/Z stereochemistry")
                known_index = known.get(canonical(row.smiles))
                if known_index is not None:
                    aligned = Chem.Mol(result.prepared.alignment.molecules[known_index])
                    fields = result.fields.matrix[[known_index]]
                    charge = result.fields.charges[known_index]
                    frame.loc[index, "pose_source"] = "Exact stored pose/fields reused"
                else:
                    if not preparation_available():
                        raise ValueError("New 3D poses need sarscope[docking]; no 2D fallback")
                    if worker is None:
                        worker = stack.enter_context(_AlignmentGenerator(settings, deadline))
                    aligned, _ = worker.generate(row.smiles, reference)
                    computed = compute_fields([aligned], result.fields.grid, params)
                    if (
                        computed.labels != result.fields.labels
                        or computed.convention != result.fields.convention
                    ):
                        raise ValueError(
                            "Candidate field columns/convention differ from the fitted model"
                        )
                    fields, charge = computed.matrix, computed.charges[0]
                    frame.loc[index, "pose_source"] = (
                        f"Generated {route}; not a validated binding pose"
                    )
                validate_conformer_stereo(aligned, row.smiles)
                mapping, _, _ = best_mapping(
                    Chem.RemoveHs(aligned), Chem.RemoveHs(reference), settings, fixed_frame=True
                )
                xyz, ref_xyz = (
                    coordinates(Chem.RemoveHs(aligned)),
                    coordinates(Chem.RemoveHs(reference)),
                )
                rmsd = float(
                    np.sqrt(np.mean([np.sum((xyz[i] - ref_xyz[j]) ** 2) for i, j in mapping]))
                )
                coverage = len(mapping) / mol.GetNumHeavyAtoms()
                frame.loc[index, ["core_rmsd_A", "core_coverage"]] = [rmsd, coverage]
                if rmsd > params.max_alignment_rmsd or coverage < params.min_core_coverage:
                    raise ValueError(
                        "Generated core alignment exceeds fitted RMSD/coverage thresholds"
                    )
                if np.any(xyz < low - 1e-8) or np.any(xyz > high + 1e-8):
                    raise ValueError("Heavy atoms extend outside the fitted training grid")
                filled = float(filt.missing_counts(fields)[0] / int(filt.support_.sum()))
                electro = (
                    np.array([label.startswith("electrostatic_") for label in result.fields.labels])
                    & filt.support_
                )
                missing_electro = (
                    float(np.isnan(fields[:, electro]).mean()) if electro.any() else 0.0
                )
                frame.loc[index, ["mean_filled_fraction", "missing_electrostatic_fraction"]] = [
                    filled,
                    missing_electro,
                ]
                if max(filled, missing_electro) > params.max_imputed_fraction:
                    raise ValueError("Missing retained fields exceed the fitted reliability limit")
                X = filt.transform(fields)
                if not result.domain.classify(X).in_domain[0]:
                    raise ValueError("Outside training-field PCA bounding box")
                prediction = float(np.asarray(estimator.predict(X)).reshape(-1)[0])
                if not np.isfinite(prediction):
                    raise ValueError("Non-finite 3D prediction")
                changes = (X[0] - parent_X[0]) * coefficient
                if not np.isclose(
                    changes.sum(), prediction - parent_prediction, atol=1e-7, rtol=1e-6
                ):
                    raise ValueError(
                        "Per-field changes do not reconstruct this model's prediction delta"
                    )
                frame.loc[
                    index,
                    ["predicted_pactivity", "predicted_delta_vs_parent", "model_parent_prediction"],
                ] = [prediction, prediction - parent_prediction, parent_prediction]
                half_width = result.regression.empirical_half_width
                if np.isfinite(half_width) and half_width > 0:
                    frame.loc[index, ["empirical_lower_pactivity", "empirical_upper_pactivity"]] = [
                        prediction - half_width,
                        prediction + half_width,
                    ]
                frame.loc[index, "in_applicability_domain"] = True
                frame.loc[index, "prediction_status"] = (
                    "Exploratory 3D field-PLS prediction; not a confirmed improvement"
                )
                aligned.SetProp("_Name", row.molecule_id)
                output.aligned.append(aligned)
                contributions.extend(
                    {
                        "molecule_id": row.molecule_id,
                        "field": name,
                        "delta_pactivity_contribution": float(
                            changes[retained_names == name].sum()
                        ),
                    }
                    for name in dict.fromkeys(retained_names)
                )
                charge_rows.append({"molecule_id": row.molecule_id, **charge})
            except Exception as exc:
                for name in (
                    "predicted_pactivity",
                    "predicted_delta_vs_parent",
                    "model_parent_prediction",
                    "empirical_lower_pactivity",
                    "empirical_upper_pactivity",
                ):
                    frame.loc[index, name] = np.nan
                frame.loc[index, "in_applicability_domain"] = False
                frame.loc[index, "prediction_status"] = f"withheld: {exc}"
    output.contributions = pd.DataFrame(contributions)
    output.charges = pd.DataFrame(charge_rows)
    return output


def field_design_archive(prediction: FieldDesignResult, result: FieldQsarResult) -> bytes:
    if prediction.audit["model_signature"] != field_model_signature(result):
        raise ValueError("The fitted 3D model changed; old predictions cannot be exported with it.")
    with zipfile.ZipFile(io.BytesIO(design_archive(prediction.design))) as original:
        files = {
            name: original.read(name)
            for name in original.namelist()
            if name != "manifest.sha256.json"
        }
    sdf = io.StringIO()
    writer = Chem.SDWriter(sdf)
    for mol in prediction.aligned:
        writer.write(mol)
    writer.close()
    files["aligned_3d_candidates.sdf"] = sdf.getvalue().encode()
    files["3d_field_delta_contributions.csv"] = prediction.contributions.to_csv(
        index=False
    ).encode()
    files["3d_candidate_charges.csv"] = prediction.charges.to_csv(index=False).encode()
    files["3d_scoring_audit.json"] = json.dumps(
        prediction.audit, indent=2, allow_nan=False
    ).encode()
    from sarscope.field_qsar_report import aligned_sdf

    files["fitted_reference.sdf"] = aligned_sdf(result.prepared, reference_only=True).encode()
    files["manifest.sha256.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in files.items()}, indent=2
    ).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return stream.getvalue()

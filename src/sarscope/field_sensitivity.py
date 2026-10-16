"""Training-only Gaussian-field stability, not repeated external-set optimisation.

Reuse the real nested regression engine. Held-out rows and measurements are
removed before grid, charge, pose perturbation or model selection. Relative
rigid perturbations are pose-uncertainty stress tests, NOT alternate verified
binding modes or generated-conformer validation. No docking is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.spatial.transform import Rotation
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.contours import contribution_maps
from sarscope.analysis.fields import GridSpec, compute_fields
from sarscope.analysis.regression import (
    _fit_field_pls,
    regression_cv_predictions,
    regression_metrics,
)
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_benchmark import verify_source
from sarscope.field_campaign import _json, _manifest
from sarscope.field_qsar import (
    FieldDataset,
    PreparedFieldSeries,
    _pooled_q2,
    prepare_field_series,
    upload_table,
)
from sarscope.params import FieldQsarParams, ModelParams


@dataclass
class FieldSensitivityResult:
    summary: pd.DataFrame
    predictions: pd.DataFrame
    contributions: pd.DataFrame
    provenance: dict[str, Any]


def training_dataset(prepared: PreparedFieldSeries, indices: list[int]) -> FieldDataset:
    """Copy fitting rows in their original fitting order; no held-out labels or geometry."""
    selected = np.asarray(indices)
    if (
        selected.ndim != 1
        or not np.issubdtype(selected.dtype, np.integer)
        or len(selected) < 15
        or len(set(indices)) != len(indices)
        or selected.min() < 0
        or selected.max() >= len(prepared.dataset.table)
    ):
        raise ValueError("Use at least 15 unique, in-range training indices.")
    table = prepared.dataset.table.iloc[selected].reset_index(drop=True).copy()
    measurements = prepared.dataset.measurements[
        prepared.dataset.measurements.molecule_id.isin(table.molecule_id)
    ].copy()
    return FieldDataset(
        table,
        [Chem.Mol(prepared.alignment.molecules[int(i)]) for i in selected],
        measurements,
        prepared.dataset.accounting.copy(),
        {
            "kind": "training-only sensitivity",
            "scope": "training_only",
            "training_rows": indices,
            "source": prepared.dataset.source,
        },
    )


def perturb_poses(
    dataset: FieldDataset,
    *,
    seed: int,
    translation: float,
    rotation_degrees: float,
    common_translation: bool = False,
    common_step: float = 2.0,
) -> FieldDataset:
    """Proper rigid transforms preserve bonds/stereo; fix the first fitting reference."""
    if not 0 <= translation <= 2 or not 0 <= rotation_degrees <= 30:
        raise ValueError("Bounded pose perturbations require 0–2 Å and 0–30 degrees.")
    molecules = []
    for index, (identifier, original) in enumerate(
        zip(dataset.table.molecule_id, dataset.molecules, strict=True)
    ):
        mol = Chem.Mol(original)
        xyz = mol.GetConformer().GetPositions()
        if common_translation:
            moved = xyz + common_step * np.array([2.0, 3.0, -1.0])
        elif index == 0:
            moved = xyz
        else:
            local_seed = int.from_bytes(
                hashlib.sha256(f"{seed}:{identifier}".encode()).digest()[:8], "little"
            )
            rng = np.random.default_rng(local_seed)
            axis, shift = rng.normal(size=(2, 3))
            axis /= np.linalg.norm(axis)
            shift = translation * shift / np.linalg.norm(shift)
            rotation = Rotation.from_rotvec(axis * np.deg2rad(rotation_degrees)).as_matrix()
            centre = xyz.mean(0)
            moved = (xyz - centre) @ rotation.T + centre + shift
        for atom, point in enumerate(moved):
            mol.GetConformer().SetAtomPosition(atom, point)
        molecules.append(mol)
    return replace(dataset, molecules=molecules)


def training_fit(
    dataset: FieldDataset, settings: FieldQsarParams, model: ModelParams
) -> dict[str, Any]:
    """Thin field adapter to shared nested CV/PLS; never introduce another solver."""
    prepared = prepare_field_series(dataset, settings)
    grid = GridSpec.around(prepared.alignment.molecules, settings)
    fields = compute_fields(prepared.alignment.molecules, grid, settings)
    y = dataset.table.pactivity.to_numpy(dtype=float)
    labels = np.full(len(y), "series", dtype=object)
    groups = add_scaffolds(dataset.table).murcko.fillna("").to_numpy()
    model = replace(model, field_qsar=settings, regression_algorithms=("field_pls",))
    with threadpool_limits(limits=1):
        prediction, _, _ = regression_cv_predictions(
            "field_pls", fields.matrix, y, labels, groups, model
        )
        filt, fitted = _fit_field_pls(fields.matrix, y, model, groups, labels, None)
    if not np.isfinite(prediction).all():
        raise ValueError("Every training row must have a finite out-of-fold prediction.")
    _, rmse, rho = regression_metrics(y, prediction)
    return {
        "q2": _pooled_q2(y, prediction),
        "rmse": rmse,
        "spearman": rho,
        "prediction": prediction,
        "components": fitted.selected_components_,
        "retained_columns": int(filt.support_.sum()),
        # The fitted filter, so callers can report retention per field block
        # without recomputing the fields.
        "filter": filt,
        "grid_points": grid.n_points,
        "contributions": contribution_maps(fields, filt, fitted, settings).summary,
        "charges": fields.charges,
    }


def sensitivity_variants(charge_model: str = "gasteiger") -> list[dict[str, Any]]:
    """Fixed diagnostics, not a grid search for the highest external R²."""
    alternative = "gasteiger" if charge_model == "mmff94" else "mmff94"
    return [
        {"name": "baseline", "settings": {}, "kind": "baseline"},
        {"name": alternative, "settings": {"charge_model": alternative}, "kind": "charge"},
        *[
            {"name": f"phase_{axis}", "settings": {"grid_phase": phase}, "kind": "grid_phase"}
            for axis, phase in [
                ("x", (0.5, 0, 0)),
                ("y", (0, 0.5, 0)),
                ("z", (0, 0, 0.5)),
                ("xyz", (0.5, 0.5, 0.5)),
            ]
        ],
        *[
            {"name": f"pose_{seed}", "settings": {}, "kind": "relative_pose", "seed": seed}
            for seed in (1101, 1102, 1103)
        ],
        {"name": "common_translation", "settings": {}, "kind": "numerical_control"},
    ]


def run_sensitivity(
    dataset: FieldDataset,
    settings: FieldQsarParams,
    model: ModelParams,
    *,
    variants: list[dict[str, Any]] | None = None,
    max_seconds: float = 180,
) -> FieldSensitivityResult:
    """Input must already be training-only; UI must use training_dataset first.

    Thresholds are engineering flags, not universal QSAR validity criteria.
    Charges may legitimately change conclusions. We do not pick a winner by
    q² or choose a charge model from the diagnostic test population.
    """
    if settings.method != "comsia" or model.split != "random":
        raise ValueError("This diagnostic profile supports Gaussian fields and random nested CV.")
    if dataset.source.get("scope") != "training_only":
        raise ValueError("Explicitly select training-only data before stability analysis.")
    settings = replace(
        settings,
        alignment="prealigned",
        assess_prealigned_core=False,
        preserve_input_hydrogens=True,
        reference_index=0,
        permutations=0,
    )
    variants = sensitivity_variants(settings.charge_model) if variants is None else variants
    if not variants or variants[0]["name"] != "baseline" or len(variants) > 12:
        raise ValueError("Use baseline first and at most twelve declared diagnostic variants.")
    start = time.monotonic()
    baseline, rows, predictions, contributions = None, [], [], []
    for variant in variants:
        name, kind = variant["name"], variant["kind"]
        try:
            if time.monotonic() - start > max_seconds:
                raise ValueError("Training sensitivity time budget exceeded.")
            current = dataset
            if kind in ("relative_pose", "numerical_control"):
                current = perturb_poses(
                    dataset,
                    seed=variant.get("seed", 0),
                    translation=0.25,
                    rotation_degrees=5,
                    common_translation=kind == "numerical_control",
                    common_step=settings.spacing,
                )
            fitted = training_fit(current, replace(settings, **variant["settings"]), model)
            if baseline is None:
                baseline = fitted
            difference = np.abs(fitted["prediction"] - baseline["prediction"])
            p95, maximum = float(np.quantile(difference, 0.95)), float(difference.max())
            numerical = kind == "numerical_control"
            passes = maximum <= 1e-8 if numerical else p95 <= 0.5 and maximum <= 1.0
            rows.append(
                {
                    "variant": name,
                    "kind": kind,
                    "completed": True,
                    "training_oof_q2": fitted["q2"],
                    "training_oof_rmse": fitted["rmse"],
                    "oof_change_p95": p95,
                    "oof_change_max": maximum,
                    "engineering_stability_flag": passes,
                    "components": fitted["components"],
                    "retained_columns": fitted["retained_columns"],
                    "grid_points": fitted["grid_points"],
                    "charge_fallbacks": sum(bool(row["fallback"]) for row in fitted["charges"]),
                }
            )
            for identifier, observed, prediction, delta in zip(
                dataset.table.molecule_id,
                dataset.table.pactivity,
                fitted["prediction"],
                difference,
                strict=True,
            ):
                predictions.append(
                    {
                        "variant": name,
                        "molecule_id": identifier,
                        "training_activity": observed,
                        "oof_prediction": prediction,
                        "absolute_change": delta,
                    }
                )
            contributions.append(fitted["contributions"].assign(variant=name))
        except Exception as exc:
            rows.append(
                {
                    "variant": name,
                    "kind": kind,
                    "completed": False,
                    "engineering_stability_flag": False,
                    "error": str(exc),
                }
            )
            if name == "baseline":
                rows.extend(
                    {
                        "variant": pending["name"],
                        "kind": pending["kind"],
                        "completed": False,
                        "engineering_stability_flag": False,
                        "error": "Blocked by baseline failure",
                    }
                    for pending in variants[1:]
                )
                break
    return FieldSensitivityResult(
        pd.DataFrame(rows),
        pd.DataFrame(predictions),
        pd.concat(contributions, ignore_index=True) if contributions else pd.DataFrame(),
        {
            "settings": asdict(settings),
            "model": asdict(model),
            "variants": variants,
            "source": dataset.source,
            "training_rows": len(dataset.table),
            "held_out_rows_used": 0,
            "held_out_labels_used": False,
            "decision": "diagnostic only; no automatic preset/charge selection or certification",
            "thresholds": {
                "prediction_change_p95": 0.5,
                "prediction_change_max": 1.0,
                "common_translation_max": 1e-8,
            },
            "pose_perturbation": {
                "translation_A": 0.25,
                "rotation_degrees": 5,
                "reference": "first training row held fixed",
            },
            "versions": provenance.package_versions(),
            "runner_sha256": provenance.file_sha256(Path(__file__)),
            "workflow_sha256": _workflow_hashes(),
            "elapsed_seconds": time.monotonic() - start,
        },
    )


def _workflow_hashes() -> dict[str, str]:
    # Local import avoids a second, divergent list of the qualified sources.
    from sarscope.field_qualification import workflow_hashes

    return workflow_hashes()


def load_training(payload: bytes, source: dict[str, Any]) -> FieldDataset:
    verify_source(payload, source["train"])
    raw, molecules = upload_table(payload, "sdf")
    if len(raw) != source["train"]["valid"] or raw.smiles.duplicated().any():
        raise ValueError("Training record accounting/structure uniqueness differs from protocol.")
    table = raw[["molecule_id", "smiles"]].copy()
    table["pactivity"] = pd.to_numeric(raw[source["activity"]], errors="raise")
    if not np.isfinite(table.pactivity).all():
        raise ValueError("Training labels must be finite.")
    return FieldDataset(
        table, molecules, table.copy(), pd.DataFrame(), {**source, "scope": "training_only"}
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("Use a new empty directory; preserve earlier evidence.")
    args.out.mkdir(parents=True, exist_ok=True)
    _json(args.out / "protocol.json", protocol)
    summaries = []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for source in protocol["datasets"]:
            response = client.get(protocol["data_base_url"] + source["name"] + "_train.sdf")
            response.raise_for_status()
            # No test SDF is fetched or parsed by this command.
            dataset = load_training(response.content, source)
            for profile in protocol["profiles"]:
                folder = args.out / source["name"] / profile["name"]
                folder.mkdir(parents=True)
                params = FieldQsarParams(**{**protocol["fields"], **profile["settings"]})
                print(f"Training-only stability: {source['name']}/{profile['name']}…", flush=True)
                result = run_sensitivity(dataset, params, ModelParams(**protocol["model"]))
                result.summary.to_csv(folder / "summary.csv", index=False)
                result.predictions.to_csv(folder / "training_oof_predictions.csv", index=False)
                result.contributions.to_csv(folder / "contributions.csv", index=False)
                _json(folder / "provenance.json", result.provenance)
                summaries.append(
                    result.summary.assign(dataset=source["name"], profile=profile["name"])
                )
                pd.concat(summaries, ignore_index=True).to_csv(
                    args.out / "summary.csv", index=False
                )
    _manifest(args.out)


if __name__ == "__main__":
    main()

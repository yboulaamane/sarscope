"""Audit already-frozen evidence; never fit, tune or alter a source result.

An evidence summary is not a second model-selection layer or a vendor-parity
claim. Independent recalculation of recorded scores catches reporting errors.
Source-graph warnings are retrospective audit flags, not exclusion rules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope import provenance
from sarscope.field_campaign import _json, _manifest
from sarscope.field_qualification import verify_freeze


def verify_manifest(folder: Path) -> None:
    for name, digest in json.loads((folder / "manifest.sha256.json").read_text()).items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or provenance.file_sha256(path) != digest:
            raise ValueError(f"Evidence changed: {folder.name}/{name}")


def assess(root: Path, out: Path) -> dict[str, Any]:
    if out.exists():
        raise FileExistsError("Retain earlier assessment; use a new path.")
    folders = {
        "sensitivity": root / "public-se-sensitivity-20261016",
        "evaluation": root / "public-se-evaluation-20261016",
        "python313": root / "tripos-components-20261016-v3",
        "python312": root / "tripos-components-20261016-py312",
    }
    for folder in folders.values():
        verify_manifest(folder)
    freeze = verify_freeze(
        root / "public-se-qualification-20261016.freeze.json",
        root / "public-se-qualification-protocol.json",
    )
    original = pd.read_csv(folders["python313"] / "summary.csv")
    repeated = pd.read_csv(folders["python312"] / "summary.csv")
    comparisons = []
    for _, row in original.iterrows():
        other = repeated[repeated.dataset.eq(row.dataset) & repeated.method.eq(row.method)].iloc[0]
        case = Path(row.dataset) / row.method
        a = pd.read_csv(folders["python313"] / case / "predictions.csv")
        b = pd.read_csv(folders["python312"] / case / "predictions.csv")
        if not a.molecule_id.equals(b.molecule_id) or not a.split.equals(b.split):
            raise ValueError("Cross-stack compound/split order differs.")
        comparisons.append(
            {
                "dataset": row.dataset,
                "method": row.method,
                "max_prediction_difference": float(np.max(np.abs(a.prediction - b.prediction))),
                "q2_difference": abs(float(row.q2 - other.q2)),
                "test_r2_difference": abs(float(row.test_r2 - other.test_r2)),
                "rf_rmse_difference": abs(float(row.rf_test_rmse - other.rf_test_rmse)),
                "same_component_count": bool(row.components == other.components),
                "same_retained_column_count": bool(row.retained_columns == other.retained_columns),
            }
        )
    comparison = pd.DataFrame(comparisons)
    stability = pd.read_csv(folders["sensitivity"] / "summary.csv")
    phase = stability[stability.kind.eq("grid_phase")]
    poses = stability[stability.kind.eq("relative_pose")]
    control = stability[stability.kind.eq("numerical_control")]
    performance = pd.read_csv(folders["evaluation"] / "summary.csv")
    models = []
    for _, row in performance.iterrows():
        case = folders["evaluation"] / row.dataset / row.method
        predictions = pd.read_csv(case / "predictions.csv")
        heldout = predictions[predictions.split.eq("held_out")]
        error = heldout.pactivity.to_numpy() - heldout.prediction.to_numpy()
        r2 = 1 - float(error @ error) / float(
            np.sum((heldout.pactivity - heldout.pactivity.mean()) ** 2)
        )
        rmse = float(np.sqrt(np.mean(error**2)))
        if not np.isclose(r2, row.test_r2, atol=1e-12) or not np.isclose(
            rmse, row.test_rmse, atol=1e-12
        ):
            raise ValueError("Independent test-score recalculation differs.")
        null = pd.read_csv(case / "permutations.csv")
        pvalue = (1 + int((null.q2 >= row.q2).sum())) / (1 + len(null))
        if len(null) != 99 or not np.isclose(pvalue, row.permutation_pvalue):
            raise ValueError("Permutation accounting differs.")
        checks = pd.read_csv(case / "checks.csv")
        training = predictions[predictions.split.eq("train")]
        mean_rmse = float(np.sqrt(np.mean((heldout.pactivity - training.pactivity.mean()) ** 2)))
        if not np.isclose(mean_rmse, row.mean_test_rmse):
            raise ValueError("Recorded training-mean baseline differs.")
        contribution = pd.read_csv(case / "contributions.csv")
        bootstrap = pd.read_csv(case / "bootstrap.csv")
        interval = bootstrap[bootstrap.metric.eq("R²")].iloc[0]
        models.append(
            {
                "method": row.method,
                "nested_q2": float(row.q2),
                "test_r2": r2,
                "test_r2_conditional_95_interval": [
                    float(interval.lower_95),
                    float(interval.upper_95),
                ],
                "test_rmse": rmse,
                "mean_baseline_rmse": mean_rmse,
                "rf_baseline_rmse": float(row.rf_test_rmse),
                "permutation_pvalue": pvalue,
                "passes_predeclared_predictive_thresholds": bool(
                    row.q2 > 0.5 and r2 > 0.6 and pvalue <= 0.05 and rmse < mean_rmse
                ),
                "failed_pipeline_checks": checks.loc[~checks.passed, "check"].tolist(),
                "pca_box_coverage": float(heldout.pca_box_in_domain.mean()),
                "rows_over_missingness_limit": int((~heldout.mean_fill_within_limit).sum()),
                "retained_columns_by_field": dict(
                    zip(contribution.field, contribution.retained_columns.astype(int), strict=True)
                ),
            }
        )
    graph_rows = []
    for folder, dataset, method in [
        *((folders["python313"], name, "comsia_se") for name in ("AChE", "CCR5", "THR")),
        (folders["evaluation"], "ATA", "gaussian_se_primary"),
    ]:
        table = pd.read_csv(folder / dataset / method / "predictions.csv")
        for row in table.itertuples():
            mol = Chem.MolFromSmiles(row.smiles)
            if mol is None:
                raise ValueError("An exported graph cannot be parsed for the source audit.")
            graph_rows.append(
                {
                    "dataset": dataset,
                    "molecule_id": row.molecule_id,
                    "split": row.split,
                    "radical_electrons": sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms()),
                    "formal_charge": Chem.GetFormalCharge(mol),
                    "fragments": len(Chem.GetMolFrags(mol)),
                }
            )
    graph_audit = pd.DataFrame(graph_rows)
    record = {
        "freeze_verified_against_current_workflow": True,
        "protocol_sha256": freeze["protocol_sha256"],
        "all_input_manifests_verified": True,
        "sampling": {
            "phase_checks_passed": int(phase.engineering_stability_flag.sum()),
            "phase_checks_total": len(phase),
            "worst_phase_prediction_change": float(phase.oof_change_max.max()),
            "common_translation_max": float(control.oof_change_max.max()),
            "relative_pose_flags_failed": int((~poses.engineering_stability_flag).sum()),
            "relative_pose_flags_total": len(poses),
        },
        "cross_environment": {
            "same_machine_not_independent_lab": True,
            "max_prediction_difference": float(comparison.max_prediction_difference.max()),
            "all_component_counts_agree": bool(comparison.same_component_count.all()),
            "all_column_counts_agree": bool(comparison.same_retained_column_count.all()),
            "max_rf_rmse_difference": float(comparison.rf_rmse_difference.max()),
        },
        "held_out_models": models,
        "source_graph_audit": {
            name: {
                "rows": len(group),
                "graphs_with_radical_electrons": int((group.radical_electrons > 0).sum()),
                "disconnected_graphs": int((group.fragments > 1).sum()),
            }
            for name, group in graph_audit.groupby("dataset")
        },
        "predictive_release_decision": (
            "NO-GO: both reserved-cohort models fail predictive criteria"
        ),
        "scope": "Documented public SE components, reviewed pre-aligned SDF; not full SYBYL parity",
        "package_certified": False,
        "test_set_status": (
            "ATA is now consumed; do not describe it as reserved/untouched or retune on it"
        ),
        "source_audit_interpretation": (
            "Radicals/disconnected graphs are retrospective flags requiring source adjudication, "
            "not proof that a given structure is wrong or an excuse to discard poor predictions. "
            "Nothing was repaired, removed, or rescored."
        ),
        "reporter_sha256": provenance.file_sha256(Path(__file__)),
    }
    out.mkdir(parents=True)
    comparison.to_csv(out / "cross_environment.csv", index=False)
    graph_audit.to_csv(out / "source_graph_flags.csv", index=False)
    _json(out / "assessment.json", record)
    _manifest(out)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(assess(args.root, args.out), indent=2))

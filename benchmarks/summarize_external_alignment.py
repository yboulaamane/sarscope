"""Collect measured comparison results; retain failed import attempts.

Publish only compact aggregates/protocols, not noncommercial primary structures
or legacy GPL binaries. Full local logs and per-compound outputs remain in runs/.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import r2_score, root_mean_squared_error

from benchmarks.compare_o3a_alignment import sha, write_json


def run(out: Path, publish: Path) -> None:
    if publish.exists():
        raise FileExistsError("Preserve the previous published evidence")
    publish.mkdir(parents=True)
    model_rows = [json.loads(p.read_text()) for p in sorted(out.glob("*/summary.json"))]
    pd.DataFrame(model_rows).drop(
        columns=["grid", "model_status", "previous_absolute_differences"], errors="ignore"
    ).to_csv(publish / "model_metrics.csv", index=False)
    geometry = pd.read_csv(out / "egfr_geometry.csv")
    standalone = pd.read_csv(out / "standalone_geometry_audit.csv")
    rows = []
    for name, frame in geometry.groupby("route"):
        rows.append(
            {
                "cohort": "EGFR",
                "route": name,
                "n": len(frame),
                "identity_stereo_all_pass": bool(frame.identity_and_stereo_preserved.all()),
                "max_internal_distance_error_A": frame.distance_error_A.max(),
                "median_core_rmsd_A": frame.core_rmsd_A.median(),
                "max_core_rmsd_A": frame.core_rmsd_A.max(),
            }
        )
    for name, frame in standalone.groupby("cohort"):
        rows.append(
            {
                "cohort": name,
                "route": "open3dalign_atom",
                "n": len(frame),
                "identity_stereo_all_pass": bool(frame.identity_and_stereo_preserved.all()),
                "max_internal_distance_error_A": frame.distance_error_A.max(),
                "median_core_rmsd_A": frame.core_rmsd_A.median() if name == "EGFR" else None,
                "max_core_rmsd_A": frame.core_rmsd_A.max() if name == "EGFR" else None,
            }
        )
    pd.DataFrame(rows).to_csv(publish / "geometry_summary.csv", index=False)
    solver = []
    root = out / "standalone"
    for name, variant in (("THR", "babel3"), ("AChE", "dx_three_values")):
        expected = pd.read_csv(
            root / f"{name}_expected_predictions.csv", dtype={"molecule_id": str}
        )
        expected = expected.set_index("molecule_id")
        folder = root / f"{name}_open3dqsar_solver_{variant}"
        results = []
        for file, split in (
            ("train_predictions.sdf", "train"),
            ("test_predictions.sdf", "held_out"),
        ):
            for mol in Chem.SDMolSupplier(str(folder / file), removeHs=False):
                identifier = mol.GetProp("_Name")
                row = expected.loc[identifier]
                if row.split != split or abs(float(mol.GetProp("Actual")) - row.pactivity) > 1e-8:
                    raise ValueError("External partition/activity differs")
                results.append(
                    {
                        "molecule_id": identifier,
                        "split": split,
                        "activity": row.pactivity,
                        "ours": row.prediction,
                        "external": float(mol.GetProp("PC 3")),
                    }
                )
        frame = pd.DataFrame(results)
        if len(frame) != len(expected) or not frame.molecule_id.is_unique:
            raise ValueError("Missing/duplicate external predictions")
        frame["difference"] = abs(frame.ours - frame.external)
        frame.to_csv(folder / "final_prediction_comparison.csv", index=False)
        test = frame[frame.split == "held_out"]
        row = {
            "cohort": name,
            "n": len(frame),
            "components": 3,
            "format": variant,
            "max_absolute_prediction_difference": float(frame.difference.max()),
            "tolerance": 5e-4,
            "parity_pass": bool(frame.difference.max() <= 5e-4),
            "sarscope_test_r2": r2_score(test.activity, test.ours),
            "open3dqsar_test_r2": r2_score(test.activity, test.external),
            "sarscope_test_rmse": root_mean_squared_error(test.activity, test.ours),
            "open3dqsar_test_rmse": root_mean_squared_error(test.activity, test.external),
        }
        if name == "AChE":
            matrix = np.loadtxt(root / "AChE_scaled_matrix.txt")
            returned = np.vstack(
                [
                    np.loadtxt(folder / f"roundtrip_fld-01_obj-{i:02d}.agrd", skiprows=6)
                    .reshape(20, 10, 10)
                    .transpose(2, 1, 0)
                    .ravel()
                    for i in range(1, 112)
                ]
            )
            row["matrix_roundtrip_max_difference"] = float(abs(matrix - returned).max())
            row["matrix_roundtrip_terms_abs_diff_gt_1e_minus_4"] = int(
                (abs(matrix - returned) > 1e-4).sum()
            )
        solver.append(row)
    pd.DataFrame(solver).to_csv(publish / "solver_parity.csv", index=False)
    rf = {}
    for name in ("THR", "AChE"):
        original = pd.read_csv(out / f"{name}_original/predictions.csv")
        for route in ("o3a_mmff", "o3a_crippen", "open3dalign_atom"):
            changed = pd.read_csv(out / f"{name}_{route}/predictions.csv")
            rf[f"{name}_{route}"] = float(
                abs(original.baseline_prediction - changed.baseline_prediction).max()
            )
    summaries = {
        "scope": "Retrospective alignment sensitivity and PLS solver parity, not field equivalence",
        "model_fits": len(model_rows),
        "model_successes": sum(r["status"] == "ok" for r in model_rows),
        "original_baselines_reproduced": all(
            r.get("original_reproduces", True) for r in model_rows
        ),
        "solver_parity_both_pass": all(row["parity_pass"] for row in solver),
        "rf_prediction_max_changes": rf,
        "geometry": rows,
        "source_data_licences": {
            "EGFR": "ChEMBL CC-BY-SA-3.0",
            "THR_AChE": "Publisher primary data CC-BY-NC-4.0",
        },
        "failure_retention": "All failed interfaces retained under "
        "runs/o3a-comparison-20261019/standalone",
        "model_selection": "No route chosen by held-out scores; no production changes",
        "permutations": 0,
        "interpretation": "No new validation status granted",
    }
    write_json(publish / "summary.json", summaries)
    for src, target in (
        (out / "protocol.json", "protocol.json"),
        (root / "protocol.json", "standalone_protocol.json"),
        (out / "standalone-model-extension.json", "standalone_model_extension.json"),
        (root / "interchange_audit.csv", "failed_free_format_audit.csv"),
        (root / "opendx-three-value-protocol.json", "opendx_retry_protocol.json"),
        (out / "full-suite.xml", "full-suite.xml"),
    ):
        shutil.copy2(src, publish / target)
    raw_manifest = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
    write_json(publish / "raw_manifest.sha256.json", raw_manifest)
    source_paths = [
        p
        for p in Path("benchmarks").glob("*.py")
        if p.name
        in {
            "compare_o3a_alignment.py",
            "compare_open3dtools.py",
            "analyze_external_alignment.py",
            "audit_open3dqsar_interchange.py",
            "retry_open3dqsar_dx.py",
            "summarize_external_alignment.py",
        }
    ]
    write_json(publish / "driver_sha256.json", {str(p): sha(p) for p in source_paths})
    write_json(
        publish / "manifest.sha256.json", {p.name: sha(p) for p in publish.iterdir() if p.is_file()}
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--publish", type=Path, required=True)
    args = parser.parse_args()
    run(args.out, args.publish)

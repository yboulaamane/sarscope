"""Audit exported standalone geometry and fit unchanged SARscope field models.

This is a prespecified extension to the fixed-reference alignment experiment;
neither component candidates nor scientific success thresholds are changed.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import r2_score, root_mean_squared_error
from threadpoolctl import threadpool_limits

from benchmarks.compare_o3a_alignment import (
    COHORTS,
    graph,
    native_inputs,
    sha,
    write_json,
)
from sarscope.analysis.align3d import best_mapping
from sarscope.conformers import validate_conformer_stereo
from sarscope.field_qsar import FieldDataset, prepare_field_series, run_field_qsar
from sarscope.params import FieldQsarParams, ModelParams


def audit(out: Path) -> list[dict]:
    root = out / "standalone"
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    for name, info in protocol["sources"].items():
        originals = list(Chem.SDMolSupplier(str(root / f"{name}.sdf"), removeHs=False))
        files = list((root / f"{name}_open3dalign_babel3/aligned").glob("*.sdf"))
        if len(files) != 1:
            raise ValueError("Expected exactly one fixed-template output per cohort")
        aligned = list(Chem.SDMolSupplier(str(files[0]), removeHs=False))
        if len(aligned) != len(originals) or any(m is None for m in aligned):
            raise ValueError("Standalone output has missing/invalid structures")
        ref_index = info["reference_object"] - 1
        reference = Chem.RemoveHs(originals[ref_index])
        for before, after in zip(originals, aligned, strict=True):
            identifier = before.GetProp("_Name")
            if identifier != after.GetProp("_Name") or graph(before) != graph(after):
                raise ValueError("Standalone changed identity/order")
            validate_conformer_stereo(after, graph(before))
            if before.GetNumAtoms() != after.GetNumAtoms() or any(
                a.GetAtomicNum() != b.GetAtomicNum()
                for a, b in zip(before.GetAtoms(), after.GetAtoms(), strict=True)
            ):
                raise ValueError("Atom order/count changed: cannot compare indexed distances")
            xyz, new = before.GetConformer().GetPositions(), after.GetConformer().GetPositions()
            distance_error = float(
                np.max(
                    np.abs(
                        np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
                        - np.linalg.norm(new[:, None] - new[None, :], axis=2)
                    )
                )
            )
            row = {
                "cohort": name,
                "route": "open3dalign_atom",
                "molecule_id": identifier,
                "identity_and_stereo_preserved": True,
                "distance_error_A": distance_error,
                "distance_preserved_at_sdf_precision": distance_error < 4e-4,
            }
            if name == "EGFR":
                heavy = Chem.RemoveHs(after)
                mapping, rmsd, _ = best_mapping(
                    heavy, reference, FieldQsarParams(), fixed_frame=True
                )
                row.update(core_rmsd_A=rmsd, core_coverage=len(mapping) / heavy.GetNumHeavyAtoms())
            rows.append(row)
    pd.DataFrame(rows).to_csv(out / "standalone_geometry_audit.csv", index=False)
    return rows


def fit(out: Path, name: str) -> None:
    frozen, _, params, source = native_inputs(name)
    root = out / "standalone"
    aligned_path = list((root / f"{name}_open3dalign_babel3/aligned").glob("*.sdf"))[0]
    molecules = list(Chem.SDMolSupplier(str(aligned_path), removeHs=False))
    folder = out / f"{name}_open3dalign_atom"
    folder.mkdir(exist_ok=False)
    train = list(np.flatnonzero(frozen.split.to_numpy() == "train"))
    test = list(np.flatnonzero(frozen.split.to_numpy() == "held_out"))
    table = frozen[["molecule_id", "smiles", "pactivity"]].copy()
    dataset = FieldDataset(table, molecules, table.copy(), pd.DataFrame(), source)
    start = time.monotonic()
    try:
        with threadpool_limits(limits=1):
            result = run_field_qsar(
                prepare_field_series(dataset, params),
                ModelParams(
                    split="random",
                    cv_folds=3,
                    test_fraction=0.33,
                    oversample=False,
                    leakage_audit=False,
                    seed=42,
                ),
                split_indices=(train, test),
            )
        for filename, frame in (
            ("predictions", result.predictions),
            ("statistics", result.statistics),
            ("checks", result.checks),
        ):
            frame.to_csv(folder / f"{filename}.csv", index=False)
        write_json(folder / "provenance.json", result.provenance)
        row = result.statistics.loc[result.statistics.algorithm == "field_pls"].iloc[0]
        summary = {
            "cohort": name,
            "route": "open3dalign_atom",
            "status": "ok",
            "n": len(table),
            "train": len(train),
            "test": len(test),
            "q2": float(row.cv_pooled_q2),
            "test_r2": float(row.test_r2),
            "test_rmse": float(row.test_rmse),
            "selected_components": int(row.selected_components),
            "grid": asdict(result.fields.grid),
            "model_status": result.status,
        }
    except Exception as exc:
        summary = {
            "cohort": name,
            "route": "open3dalign_atom",
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary["elapsed_s"] = time.monotonic() - start
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary), flush=True)


def solver_audit(out: Path) -> None:
    rows = []
    details = []
    root = out / "standalone"
    for name in COHORTS:
        expected = pd.read_csv(
            root / f"{name}_expected_predictions.csv", dtype={"molecule_id": str}
        )
        expected = expected.set_index("molecule_id")
        folder = root / f"{name}_open3dqsar_solver_babel3"
        for file, split in (
            ("train_predictions.sdf", "train"),
            ("test_predictions.sdf", "held_out"),
        ):
            for mol in Chem.SDMolSupplier(str(folder / file), removeHs=False):
                identifier = mol.GetProp("_Name")
                row = expected.loc[identifier]
                if row.split != split or abs(float(mol.GetProp("Actual")) - row.pactivity) > 1e-8:
                    raise ValueError("External solver changed activities/membership")
                external = float(mol.GetProp("PC 3"))
                details.append(
                    {
                        "cohort": name,
                        "molecule_id": identifier,
                        "split": split,
                        "pactivity": row.pactivity,
                        "sarscope_prediction": row.prediction,
                        "open3dqsar_prediction": external,
                        "absolute_difference": abs(row.prediction - external),
                    }
                )
        frame = pd.DataFrame([r for r in details if r["cohort"] == name])
        test = frame[frame.split == "held_out"]
        rows.append(
            {
                "cohort": name,
                "n": len(frame),
                "max_absolute_prediction_difference": frame.absolute_difference.max(),
                "parity_tolerance": 5e-4,
                "parity_pass": bool(frame.absolute_difference.max() <= 5e-4),
                "test_r2_sarscope": r2_score(test.pactivity, test.sarscope_prediction),
                "test_r2_open3dqsar": r2_score(test.pactivity, test.open3dqsar_prediction),
                "test_rmse_sarscope": root_mean_squared_error(
                    test.pactivity, test.sarscope_prediction
                ),
                "test_rmse_open3dqsar": root_mean_squared_error(
                    test.pactivity, test.open3dqsar_prediction
                ),
            }
        )
    pd.DataFrame(rows).to_csv(out / "solver_parity.csv", index=False)
    pd.DataFrame(details).to_csv(out / "solver_prediction_comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "audit", "fit", "solver"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(COHORTS))
    args = parser.parse_args()
    if args.action == "freeze":
        path = args.out / "standalone-model-extension.json"
        if path.exists():
            raise FileExistsError("Keep original protocol")
        write_json(
            path,
            {
                "driver_sha256": sha(Path(__file__)),
                "routes": ["open3dalign_atom"],
                "cohorts": list(COHORTS),
                "params": "Same as frozen original/RDKit routes",
                "scope": "Retrospective sensitivity only; not independent validation",
                "no_parameter_tuning": True,
                "permutations": 0,
            },
        )
    elif args.action == "audit":
        audit(args.out)
    elif args.action == "solver":
        solver_audit(args.out)
    else:
        if args.cohort is None:
            parser.error("fit requires --cohort")
        protocol = json.loads((args.out / "standalone-model-extension.json").read_text())
        if protocol["driver_sha256"] != sha(Path(__file__)):
            raise ValueError("Driver changed after freeze")
        fit(args.out, args.cohort)

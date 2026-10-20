"""Bounded native-MMFF Open3DQSAR comparison, not an application backend.

Use official binaries as black boxes. No GPL algorithm source is consulted.
All historical compounds/partitions are retained. The external MMFF buffered
14-7 VDW/Coulomb fields are NOT SARscope Gaussian or Tripos LJ fields.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, root_mean_squared_error
from threadpoolctl import threadpool_limits

from benchmarks.compare_o3a_alignment import native_inputs, sha, write_json
from benchmarks.compare_open3dtools import execute
from sarscope.analysis.fields import GridSpec
from sarscope.analysis.regression import evaluate_regression
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.params import ModelParams

PREVIOUS = Path("runs/o3a-comparison-20261019")
CASES = [
    (cohort, route) for cohort in ("THR", "AChE") for route in ("original", "open3dalign_atom")
]


def case_inputs(cohort: str, route: str) -> tuple:
    frame, original, params, source = native_inputs(cohort)
    if route == "original":
        molecules = original
    else:
        paths = list(
            (PREVIOUS / "standalone" / f"{cohort}_open3dalign_babel3/aligned").glob("*.sdf")
        )
        if len(paths) != 1:
            raise ValueError("Expected one frozen standalone alignment")
        molecules = list(Chem.SDMolSupplier(str(paths[0]), removeHs=False))
        if len(molecules) != len(original) or any(m is None for m in molecules):
            raise ValueError("Incomplete external alignment; no exclusions")
        for a, b in zip(original, molecules, strict=True):
            if a.GetProp("_Name") != b.GetProp("_Name") or (
                Chem.MolToSmiles(Chem.RemoveHs(a)) != Chem.MolToSmiles(Chem.RemoveHs(b))
            ):
                raise ValueError("Alignment changed graph/order")
    train = np.flatnonzero(frame.split.to_numpy() == "train")
    test = np.flatnonzero(frame.split.to_numpy() == "held_out")
    grid = GridSpec.around([molecules[i] for i in train], params)
    return frame, molecules, params, source, train, test, grid


def commands(case: Path, grid: GridSpec, test: list[int]) -> str:
    """Explicit training-derived grid and test exclusion before pretreatment."""
    box = " ".join(
        f"{axis}_start={origin:.17g} {axis}_nodes={count}"
        for axis, origin, count in zip("xyz", grid.origin, grid.shape, strict=True)
    )
    indices = ",".join(str(i + 1) for i in test)
    return (
        f"import type=sdf file={case}/input.sdf\n"
        f"box step={grid.spacing:.17g} {box}\n"
        f"import type=dependent file={case}/activities.txt\n"
        f"set object_list={indices} attribute=testset\n"
        "calc_field type=vdw probe_type=CR smooth_probe=no\n"
        "calc_field type=mm_ele diel_dep=const diel_const=1 smooth_probe=no\n"
        "cutoff type=max level=30\ncutoff type=min level=-30\n"
        "export type=object_field field_list=1,2 object_list=all format=ascii "
        f"file={case}/external/raw\n"
        "sdcut level=2\nscale_x_vars type=buw\n"
        f"pls pc=3 file={case}/external/train_predictions.sdf\n"
        f"cv type=loo pc=3 file={case}/external/loo_predictions.sdf\n"
        f"predict pc=3 file={case}/external/test_predictions.sdf\nquit\n"
    )


def prepare(out: Path, binary_dir: Path, babel_dir: Path) -> None:
    out.mkdir(exist_ok=False, parents=True)
    cases = []
    for cohort, route in CASES:
        frame, molecules, params, source, train, test, grid = case_inputs(cohort, route)
        case = out / f"{cohort}_{route}"
        case.mkdir()
        with Chem.SDWriter(str(case / "input.sdf")) as writer:
            for mol in molecules:
                writer.write(mol)
        frame.to_csv(case / "membership.csv", index=False)
        np.savetxt(
            case / "activities.txt", frame.pactivity, fmt="%.17g", header="activity", comments=""
        )
        (case / "commands.inp").write_text(commands(case, grid, test.tolist()))
        common = replace(
            params,
            method="comfa",
            field_subset=("steric", "electrostatic"),
            charge_model="mmff94",
            permutations=0,
        )
        cases.append(
            {
                "cohort": cohort,
                "route": route,
                "n": len(frame),
                "source": source,
                "grid": asdict(grid),
                "train": train.tolist(),
                "test": test.tolist(),
                "input_sha256": {p.name: sha(p) for p in case.iterdir()},
                "common_engine_params": asdict(common),
                "common_engine_note": "method=comfa selects two-block filtering only; "
                "descriptors supplied "
                "by native Open3DQSAR, NOT computed with SARscope's LJ equations",
            }
        )
    write_json(
        out / "protocol.json",
        {
            "purpose": "Retrospective native Open3DQSAR predictive/runtime comparison; "
            "no app changes",
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "driver_sha256": sha(Path(__file__)),
            "binary_sha256": sha(binary_dir / "open3dqsar"),
            "babel_launchers_sha256": {p.name: sha(p) for p in babel_dir.iterdir() if p.is_file()},
            "n_cpus": 1,
            "timeout_per_case_s": 180,
            "seed": 42,
            "field_definition": "Native Open3DQSAR 2.3 MMFF94 buffered 14-7 VDW (CR probe) "
            "and constant-dielectric Coulomb; unsmoothed; +/-30 kcal/mol truncation",
            "native_model": "Training-only SD >=2 kcal/mol, BUW, three fixed PLS components, "
            "LOO CV. No test-based tuning. LOO is descriptive after fixed outer-training "
            "pretreatment, NOT nested CV.",
            "common_model": "Reuse SARscope nested three-fold PLS; "
            "fold-local two-block variance filtering, "
            "2000 retained features maximum, block_std, component candidates 1-4 with n/5 cap. "
            "No permutations; exploratory diagnostics, not new validation certification.",
            "runtime_note": "Native wall time includes import, field calculation, ASCII export, "
            "pretreatment, fit, LOO and prediction. Common-engine timing is modelling only; "
            "do not compare against full SARscope app time or infer speed ratios.",
            "cases": cases,
            "no_exclusions": True,
            "source_manual_sha256": sha(binary_dir.parent / "open3dqsar/doc/Open3DQSAR.html"),
            "source_policy": "Public bundled user documentation and command examples only; "
            "no GPL algorithm source",
        },
    )


def protocol(out: Path) -> dict:
    value = json.loads((out / "protocol.json").read_text())
    if value["driver_sha256"] != sha(Path(__file__)):
        raise ValueError("Driver changed after freeze; retain attempt and use new output folder")
    return value


def generate(out: Path, binary_dir: Path, babel_dir: Path) -> None:
    frozen = protocol(out)
    if sha(binary_dir / "open3dqsar") != frozen["binary_sha256"]:
        raise ValueError("External binary changed")
    for info in frozen["cases"]:
        case = out / f"{info['cohort']}_{info['route']}"
        for name, expected in info["input_sha256"].items():
            if sha(case / name) != expected:
                raise ValueError("Frozen input changed")
        result = execute(
            "open3dqsar",
            (case / "commands.inp").read_text(),
            case / "external",
            binary_dir,
            babel_dir,
            timeout=180,
        )
        print(info["cohort"], info["route"], result, flush=True)


def read_ascii(path: Path, grid: GridSpec) -> np.ndarray:
    """Legacy ASCII exports x-fast, whereas SARscope matrices are z-fast."""
    header = path.read_text().splitlines()[:6]
    counts = tuple(int(v) for v in re.findall(r"\d+", header[4].split(":", 1)[1]))
    if counts != tuple(grid.shape):
        raise ValueError("External grid dimensions differ")
    values = np.loadtxt(path, skiprows=6)
    if values.size != grid.n_points or not np.isfinite(values).all():
        raise ValueError("Incomplete or undefined native field; no row/column dropping")
    return values.reshape(grid.shape[::-1]).transpose(2, 1, 0).ravel()


def prediction_rows(case: Path, frame: pd.DataFrame) -> pd.DataFrame:
    expected = frame.set_index("molecule_id")
    rows = []
    for file, split in (
        ("train_predictions.sdf", "train"),
        ("test_predictions.sdf", "held_out"),
        ("loo_predictions.sdf", "loo"),
    ):
        mols = list(Chem.SDMolSupplier(str(case / "external" / file), removeHs=False))
        if any(m is None for m in mols):
            raise ValueError("Malformed native predictions")
        for mol in mols:
            identifier = mol.GetProp("_Name")
            row = expected.loc[identifier]
            if (split == "held_out") != (row.split == "held_out"):
                raise ValueError("External train/test membership changed")
            if abs(float(mol.GetProp("Actual")) - row.pactivity) > 1e-7:
                raise ValueError("External activities changed")
            rows.append(
                {
                    "molecule_id": identifier,
                    "split": split,
                    "pactivity": row.pactivity,
                    "prediction": float(mol.GetProp("PC 3")),
                }
            )
        desired = int(frame.split.eq("held_out" if split == "held_out" else "train").sum())
        if len(mols) != desired or len({m.GetProp("_Name") for m in mols}) != desired:
            raise ValueError("Missing or duplicate native predictions")
    return pd.DataFrame(rows)


def analyze(out: Path) -> None:
    frozen = protocol(out)
    summaries = []
    for info in frozen["cases"]:
        cohort, route = info["cohort"], info["route"]
        case = out / f"{cohort}_{route}"
        summary = {"cohort": cohort, "alignment": route, "n": info["n"]}
        execution = json.loads((case / "external/execution.json").read_text())
        summary["native_wall_s"] = execution["elapsed_s"]
        try:
            if not execution["successful_completion"] or execution.get("returncode") != 0:
                raise ValueError("Native external workflow failed; inspect retained log")
            frame = pd.read_csv(case / "membership.csv", dtype={"molecule_id": str})
            native = prediction_rows(case, frame)
            native.to_csv(case / "native_predictions.csv", index=False)
            test = native[native.split == "held_out"]
            loo = native[native.split == "loo"]
            summary.update(
                native_status="ok",
                native_test_r2=r2_score(test.pactivity, test.prediction),
                native_test_rmse=root_mean_squared_error(test.pactivity, test.prediction),
                native_test_spearman=float(spearmanr(test.pactivity, test.prediction).statistic),
                native_loo_q2=r2_score(loo.pactivity, loo.prediction),
            )
            g = info["grid"]
            grid = GridSpec(tuple(g["origin"]), tuple(g["shape"]), g["spacing"])
            matrix = np.hstack(
                [
                    np.vstack(
                        [
                            read_ascii(
                                case / f"external/raw_fld-{field:02d}_obj-{i:02d}.agrd", grid
                            )
                            for i in range(1, len(frame) + 1)
                        ]
                    )
                    for field in (1, 2)
                ]
            )
            if np.max(abs(matrix)) > 30.0001:
                raise ValueError("Native fields exceed frozen cutoff")
            for block in np.split(matrix, 2, axis=1):
                if not np.any(block.std(axis=0) > 1e-8):
                    raise ValueError("An entire field is constant; refuse numerical artefact")
            np.save(case / "native_fields.npy", matrix)
            summary["matrix_shape"] = list(matrix.shape)
            from sarscope.params import FieldQsarParams

            settings = info["common_engine_params"].copy()
            for key in ("field_subset", "grid_phase"):
                settings[key] = tuple(settings[key])
            params = ModelParams(
                split="random",
                cv_folds=3,
                test_fraction=0.33,
                regression_algorithms=("field_pls",),
                oversample=False,
                leakage_audit=False,
                field_qsar=FieldQsarParams(**settings),
                seed=42,
            )
            table = add_scaffolds(frame)
            groups = [v if v is not None else f"acyclic_{i}" for i, v in enumerate(table.murcko)]
            start = time.monotonic()
            with threadpool_limits(limits=1):
                result = evaluate_regression(
                    matrix,
                    frame.pactivity.to_numpy(),
                    groups,
                    ["series"] * len(frame),
                    params,
                    split_indices=(info["train"], info["test"]),
                )
            summary["common_modelling_wall_s"] = time.monotonic() - start
            summary.update(
                common_status="ok",
                common_test_r2=r2_score(result.test_truth, result.test_predictions),
                common_test_rmse=root_mean_squared_error(
                    result.test_truth, result.test_predictions
                ),
                common_nested_q2=r2_score(
                    frame.pactivity.to_numpy()[info["train"]], result.cv_predictions
                ),
                common_components=int(result.fitted_model.selected_components_),
            )
            result.scores.to_csv(case / "common_statistics.csv", index=False)
            predictions = frame[["molecule_id", "split", "pactivity"]].copy()
            predictions["prediction"] = result.fitted_model.predict(
                result.fitted_filter.transform(matrix)
            ).reshape(-1)
            predictions["training_oof_prediction"] = np.nan
            predictions.loc[info["train"], "training_oof_prediction"] = result.cv_predictions
            predictions.to_csv(case / "common_predictions.csv", index=False)
            result.fitted_model.selection_table_.to_csv(case / "common_components.csv", index=False)
        except Exception as exc:
            summary.setdefault("native_status", "failed")
            summary.setdefault("common_status", "failed")
            summary["error"] = f"{type(exc).__name__}: {exc}"
        write_json(case / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(out / "results.csv", index=False)
    write_json(
        out / "manifest.sha256.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "manifest.sha256.json"
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "generate", "analyze"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--babel-dir", type=Path)
    args = parser.parse_args()
    if args.action == "analyze":
        analyze(args.out.resolve())
    else:
        if args.binary_dir is None or args.babel_dir is None:
            parser.error("prepare/generate require --binary-dir and --babel-dir")
        if args.action == "prepare":
            prepare(args.out.resolve(), args.binary_dir.resolve(), args.babel_dir.resolve())
        else:
            generate(args.out.resolve(), args.binary_dir.resolve(), args.babel_dir.resolve())

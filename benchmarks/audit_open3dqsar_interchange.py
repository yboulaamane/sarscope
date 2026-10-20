"""Lossless OpenDX alternative to audit the legacy FREE_FORMAT importer.

Do not repair predictions or silently replace failed comparisons. All import
attempts are retained, and all model settings/activities/membership stay fixed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import r2_score, root_mean_squared_error

from benchmarks.compare_o3a_alignment import sha, write_json
from benchmarks.compare_open3dtools import execute


def dx_text(values: np.ndarray) -> str:
    if values.shape != (2000,) or not np.isfinite(values).all():
        raise ValueError("The frozen solver fixture needs exactly 2000 finite descriptors")
    header = (
        "object 1 class gridpositions counts 10 10 20\n"
        "origin 0 0 0\ndelta 1 0 0\ndelta 0 1 0\ndelta 0 0 1\n"
        "object 2 class gridconnections counts 10 10 20\n"
        "object 3 class array type double rank 0 items 2000 data follows\n"
    )
    return (
        header
        + "\n".join(format(v, ".17g") for v in values)
        + (
            '\nattribute "dep" string "positions"\n'
            'object "regular field" class field\n'
            'component "positions" value 1\ncomponent "connections" value 2\n'
            'component "data" value 3\n'
        )
    )


def run(root: Path, binary_dir: Path, babel_dir: Path) -> None:
    folder = root / "AChE_open3dqsar_solver_dx"
    inputs = root / "AChE_dx_inputs"
    if folder.exists() or inputs.exists():
        raise FileExistsError("Keep previous attempts")
    inputs.mkdir()
    matrix = np.loadtxt(root / "AChE_scaled_matrix.txt")
    for i, row in enumerate(matrix, 1):
        text = dx_text(row)
        if not np.array_equal(np.fromstring("\n".join(text.splitlines()[7:2007]), sep="\n"), row):
            raise ValueError("DX export changed descriptor values")
        (inputs / f"row_{i:04d}.dx").write_text(text)
    previous = root / "AChE_open3dqsar_solver_babel3"
    commands = (previous / "input.inp").read_text().replace(str(previous), str(folder))
    commands = commands.replace(
        f"import type=free_format file={root}/AChE_scaled_matrix.txt",
        f"import type=opendx file={inputs}/row_????.dx",
    )
    commands = commands.replace(
        "import type=dependent",
        f"export type=object_field object_list=all format=ascii file={folder}/roundtrip\n"
        "import type=dependent",
    )
    write_json(
        root / "opendx-import-protocol.json",
        {
            "reason": "FREE_FORMAT round trip altered first descriptor; "
            "independent documented import format",
            "driver_sha256": sha(Path(__file__)),
            "input_float64_identical": True,
            "parameters_changed": "File format only; same matrix, rows, activities, "
            "components, train/test",
            "scope": "PLS solver parity only, not physical field equivalence "
            "or new model validation",
            "input_sha256": {p.name: sha(p) for p in inputs.iterdir()},
        },
    )
    print(execute("open3dqsar", commands, folder, binary_dir, babel_dir), flush=True)


def audit(root: Path) -> None:
    matrix = np.loadtxt(root / "AChE_scaled_matrix.txt")
    rows = []
    for variant in ("babel3", "first_decimal", "short_lines", "dx"):
        folder = root / f"AChE_open3dqsar_solver_{variant}"
        export = root / "AChE_roundtrip" if variant == "babel3" else folder
        if not (folder / "test_predictions.sdf").exists():
            continue
        returned = np.vstack(
            [
                np.loadtxt(export / f"roundtrip_fld-01_obj-{i:02d}.agrd", skiprows=6)
                .reshape(20, 10, 10)
                .transpose(2, 1, 0)
                .ravel()
                for i in range(1, 112)
            ]
        )
        differences = np.abs(matrix - returned)
        expected = pd.read_csv(root / "AChE_expected_predictions.csv", dtype={"molecule_id": str})
        expected = expected.set_index("molecule_id")
        predictions = []
        for file in ("train_predictions.sdf", "test_predictions.sdf"):
            for mol in Chem.SDMolSupplier(str(folder / file), removeHs=False):
                identifier = mol.GetProp("_Name")
                row = expected.loc[identifier]
                if abs(float(mol.GetProp("Actual")) - row.pactivity) > 1e-8:
                    raise ValueError("Activity changed")
                predictions.append(
                    {
                        "molecule_id": identifier,
                        "split": row.split,
                        "pactivity": row.pactivity,
                        "sarscope_prediction": row.prediction,
                        "open3dqsar_prediction": float(mol.GetProp("PC 3")),
                    }
                )
        frame = pd.DataFrame(predictions)
        frame["absolute_difference"] = abs(frame.sarscope_prediction - frame.open3dqsar_prediction)
        frame.to_csv(folder / "prediction_comparison.csv", index=False)
        test = frame[frame.split == "held_out"]
        rows.append(
            {
                "variant": variant,
                "n": len(frame),
                "matrix_max_difference": float(differences.max()),
                "matrix_terms_differing_gt_1e4": int((differences > 1e-4).sum()),
                "first_input_value": float(matrix[0, 0]),
                "first_exported_value": float(returned[0, 0]),
                "max_absolute_prediction_difference": float(frame.absolute_difference.max()),
                "solver_parity_pass": bool(frame.absolute_difference.max() <= 5e-4),
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
    pd.DataFrame(rows).to_csv(root / "interchange_audit.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path)
    parser.add_argument("--babel-dir", type=Path)
    args = parser.parse_args()
    if args.action == "audit":
        audit(args.root.resolve())
    elif args.binary_dir is None or args.babel_dir is None:
        parser.error("run requires --binary-dir and --babel-dir")
    else:
        run(args.root.resolve(), args.binary_dir.resolve(), args.babel_dir.resolve())

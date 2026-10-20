"""Documented-import compatibility retry: conventional three-value DX lines.

The 2.3 reader did not preserve one-value-per-line DX arrays. This retry keeps
every float64 value identical and retains the failed input/output separately.
"""

import argparse
from pathlib import Path

import numpy as np

from benchmarks.audit_open3dqsar_interchange import dx_text
from benchmarks.compare_o3a_alignment import sha, write_json
from benchmarks.compare_open3dtools import execute


def run(root: Path, binary_dir: Path, babel_dir: Path) -> None:
    inputs = root / "AChE_dx_three_value_inputs"
    inputs.mkdir(exist_ok=False)
    matrix = np.loadtxt(root / "AChE_scaled_matrix.txt")
    for i, row in enumerate(matrix, 1):
        lines = dx_text(row).splitlines()
        values = [
            " ".join(format(v, ".17g") for v in row[j : j + 3]) for j in range(0, len(row), 3)
        ]
        text = "\n".join(lines[:7] + values + lines[2007:]) + "\n"
        imported = np.fromstring("\n".join(values), sep=" ")
        if not np.array_equal(imported, row):
            raise ValueError("Line rewrapping changed descriptor values")
        (inputs / f"row_{i:04d}.dx").write_text(text)
    previous = root / "AChE_open3dqsar_solver_dx_printf"
    folder = root / "AChE_open3dqsar_solver_dx_three_values"
    commands = (previous / "input.inp").read_text().replace(str(previous), str(folder))
    commands = commands.replace(str(root / "AChE_dx_inputs"), str(inputs))
    write_json(
        root / "opendx-three-value-protocol.json",
        {
            "reason": "Legacy DX reader did not preserve one-value lines. "
            "Test conventional three-value lines.",
            "driver_sha256": sha(Path(__file__)),
            "float64_values_identical": True,
            "model_changes": "None",
            "input_sha256": {p.name: sha(p) for p in inputs.iterdir()},
        },
    )
    print(execute("open3dqsar", commands, folder, binary_dir, babel_dir), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--babel-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.root.resolve(), args.binary_dir.resolve(), args.babel_dir.resolve())

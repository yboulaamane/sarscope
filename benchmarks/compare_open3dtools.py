"""Black-box standalone tool experiment; uses documentation, not GPL source.

Alignment compares single conformers on one predetermined template. Solver
comparison imports the identical already-filtered/scaled SARscope matrix into
Open3DQSAR's documented FREE_FORMAT interface. The artificial one-field grid
only indexes columns; this is PLS solver parity, NOT field-method equivalence.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.cross_decomposition import PLSRegression
from threadpoolctl import threadpool_limits

from benchmarks.compare_o3a_alignment import (
    COHORTS,
    graph,
    native_inputs,
    sha,
    write_json,
)
from sarscope.analysis.field_filter import FieldColumnFilter
from sarscope.analysis.fields import GridSpec, compute_fields
from sarscope.conformers import embed_ligand_conformer
from sarscope.field_example import field_example_files


def execute(
    tool: str, commands: str, folder: Path, binary_dir: Path, babel_dir: Path, timeout: int = 180
) -> dict:
    """Keep commands/logs and both return code and textual failure indicators."""
    folder.mkdir(parents=True, exist_ok=False)
    (folder / "input.inp").write_text(commands)
    env = os.environ | {
        "LD_LIBRARY_PATH": str(binary_dir),
        "O3_N_CPUS": "1",
        "O3_RANDOM_SEED": "42",
        "O3_BABEL_PATH": str(babel_dir),
        "BABEL_LIBDIR": str(babel_dir / "plugins"),
        "BABEL_DATADIR": str(binary_dir.parent / "share/openbabel/2.3.2"),
        "O3_PYMOL": "",
        "O3_GNUPLOT": "",
    }
    start = time.monotonic()
    try:
        result = subprocess.run(
            [str(binary_dir / tool), "-i", str(folder / "input.inp")],
            cwd=folder,
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        (folder / "stdout.log").write_text(result.stdout)
        (folder / "stderr.log").write_text(result.stderr)
        summary = {
            "returncode": result.returncode,
            "timed_out": False,
            "successful_completion": "Successful completion" in result.stdout,
            "babel_unavailable": "Working OpenBabel binaries could not be found" in result.stdout,
            "command_failed": "failed." in result.stdout.lower(),
        }
    except subprocess.TimeoutExpired as exc:
        (folder / "stdout.log").write_bytes(exc.stdout or b"")
        (folder / "stderr.log").write_bytes(exc.stderr or b"")
        summary = {"timed_out": True, "successful_completion": False}
    summary.update(
        tool=tool, elapsed_s=time.monotonic() - start, binary_sha256=sha(binary_dir / tool)
    )
    write_json(folder / "execution.json", summary)
    return summary


def prepare(out: Path, binary_dir: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    source = {}
    contents = field_example_files("egfr")
    table = pd.read_csv(io.BytesIO(contents["egfr_3d.csv"]))
    refid = json.loads(contents["source.json"])["reference_id"]
    writer = Chem.SDWriter(str(out / "EGFR.sdf"))
    for row in table.itertuples():
        mol = embed_ligand_conformer(graph(Chem.MolFromSmiles(row.smiles)), 42)[0]
        mol.SetProp("_Name", str(row.molecule_id))
        writer.write(mol)
    writer.close()
    source["EGFR"] = {
        "reference_object": list(table.molecule_id).index(refid) + 1,
        "reference_id": refid,
        "n": len(table),
        "licence": "ChEMBL CC-BY-SA-3.0",
    }
    for name in COHORTS:
        frozen, molecules, params, metadata = native_inputs(name)
        writer = Chem.SDWriter(str(out / f"{name}.sdf"))
        for mol in molecules:
            writer.write(mol)
        writer.close()
        source[name] = metadata | {"reference_object": 1, "n": len(molecules)}
        train = np.flatnonzero(frozen.split.to_numpy() == "train")
        test = np.flatnonzero(frozen.split.to_numpy() == "held_out")
        grid = GridSpec.around([molecules[i] for i in train], params)
        fields = compute_fields(molecules, grid, params)
        filt = FieldColumnFilter(params).fit(fields.matrix[train])
        matrix = filt.transform(fields.matrix)
        np.savetxt(out / f"{name}_scaled_matrix.txt", matrix, fmt="%.17g")
        np.savetxt(out / f"{name}_activities.txt", frozen.pactivity, header="activity", comments="")
        previous = pd.read_csv(COHORTS[name][1] / "statistics.csv").set_index("algorithm")
        components = int(previous.loc["field_pls", "selected_components"])
        with threadpool_limits(limits=1):
            model = PLSRegression(n_components=components, scale=False, max_iter=500, tol=1e-6)
            model.fit(matrix[train], frozen.pactivity.to_numpy()[train])
            predictions = model.predict(matrix).reshape(-1)
        expected = pd.read_csv(COHORTS[name][1] / "predictions.csv").prediction
        if not np.allclose(predictions, expected, atol=1e-8, rtol=0):
            raise ValueError("Solver fixture does not reproduce SARscope predictions")
        frozen[["molecule_id", "split", "pactivity"]].assign(prediction=predictions).to_csv(
            out / f"{name}_expected_predictions.csv", index=False
        )
        source[name].update(
            solver_matrix_shape=list(matrix.shape),
            components=components,
            train_indices=train.tolist(),
            test_indices=test.tolist(),
            original_grid=asdict(grid),
        )
    protocol = {
        "scope": "Black-box alignment and identical-input PLS parity, "
        "NOT field parity or fresh validation",
        "sources": source,
        "seed": 42,
        "n_cpus": 1,
        "timeout_per_job_s": 180,
        "driver_sha256": sha(Path(__file__)),
        "tool_binaries": {name: sha(binary_dir / name) for name in ("open3dalign", "open3dqsar")},
        "sources_consulted": "Bundled user HTML manuals and sample command files only; "
        "no algorithm source",
        "solver": "2000 already train-filtered/scaled columns indexed by artificial 10x10x20 grid; "
        "no Open3DQSAR field scaling/selection; "
        "3 components fixed from original training selection; "
        "training centering inside each solver; test rows excluded with set attribute=testset",
        "solver_tolerance": "Prediction absolute difference <= 5e-4 (SDF output precision). "
        "No CV parity claim.",
        "file_sha256": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
    }
    write_json(out / "protocol.json", protocol)


def run(out: Path, binary_dir: Path, babel_dir: Path, action: str) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    if protocol["driver_sha256"] != sha(Path(__file__)):
        raise ValueError("Driver changed after freeze")
    for name, settings in protocol["sources"].items():
        if action == "align":
            folder = out / f"{name}_open3dalign"
            commands = (
                f"import type=sdf file={out / name}.sdf\n"
                f"align type=atom object_list={settings['reference_object']} "
                f"template=single candidate=single align_dir={folder / 'aligned'}\nquit\n"
            )
        elif name == "EGFR":
            continue
        else:
            folder = out / f"{name}_open3dqsar_solver"
            test_list = ",".join(str(i + 1) for i in settings["test_indices"])
            components = settings["components"]
            commands = (
                f"import type=sdf file={out / name}.sdf\n"
                "box step=1.0 x_start=0 x_nodes=10 y_start=0 y_nodes=10 z_start=0 z_nodes=20\n"
                f"import type=free_format file={out / name}_scaled_matrix.txt\n"
                f"import type=dependent file={out / name}_activities.txt\n"
                f"set object_list={test_list} attribute=testset\n"
                f"pls pc={components} file={folder / 'train_predictions.sdf'}\n"
                f"predict pc={components} file={folder / 'test_predictions.sdf'}\nquit\n"
            )
        summary = execute(
            "open3dalign" if action == "align" else "open3dqsar",
            commands,
            folder,
            binary_dir,
            babel_dir,
        )
        print(name, summary, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "align", "solver"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)
    parser.add_argument("--babel-dir", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve(), args.binary_dir.resolve())
    else:
        if args.babel_dir is None:
            parser.error("--babel-dir is required for execution")
        run(args.out.resolve(), args.binary_dir.resolve(), args.babel_dir.resolve(), args.action)

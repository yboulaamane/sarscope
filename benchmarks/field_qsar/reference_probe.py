"""Run an unmodified external comparator, never import it into SARscope.

Only the publication, README, licence, repository metadata and SDF data were
consulted. External source is fetched for execution/hashing, not inspected or
translated. This probe does not establish numerical equivalence by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parent


def graph_hash(mol: Any) -> str:
    smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    return hashlib.sha256(smiles.encode()).hexdigest()


def verify(payload: bytes, expected: str) -> None:
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Reference hash changed; stop and re-audit, never silently replace.")


def audit_sdf(payload: bytes, activity: str) -> dict[str, Any]:
    records, invalid = [], []
    for index, mol in enumerate(
        Chem.ForwardSDMolSupplier(io.BytesIO(payload), removeHs=False), start=1
    ):
        if mol is None:
            invalid.append(index)
            continue
        records.append(
            {
                "record_1based": index,
                "name": mol.GetProp("name") if mol.HasProp("name") else mol.GetProp("_Name"),
                "activity": float(mol.GetProp(activity)),
                "atoms": mol.GetNumAtoms(),
                "explicit_hydrogens": sum(a.GetAtomicNum() == 1 for a in mol.GetAtoms()),
                "graph_sha256": graph_hash(mol),
            }
        )
    return {"valid_records": len(records), "invalid_records": invalid, "records": records}


def command(python: Path, checkout: Path, settings: dict[str, Any]) -> list[str]:
    args = [str(python), str(checkout / "comsia.py")]
    for flag, value in (
        ("train_file", checkout / "sdfs/STEROIDS_train.sdf"),
        ("predict_file", checkout / "sdfs/STEROIDS_test.sdf"),
        ("sdf_activity", "Activity"),
        ("grid_resolution", settings["grid_resolution"]),
        ("grid_padding", settings["grid_padding"]),
        ("num_components", settings["num_components"]),
        ("column_filter", settings["column_filter"]),
    ):
        args.extend([f"--{flag}", str(value)])
    # The executable's --help uses SEH, unlike the README's individual names.
    args.extend(["--fields", settings["cli_fields"]])
    args.append("--disable_visualization")
    return args


def assess_outputs(out: Path, audit: dict[str, Any], protocol: dict[str, Any]) -> dict[str, Any]:
    """Recalculate the actual supplied external set, not a CLI-labelled internal split."""
    metric_files = list(out.rglob("PLS_Metrics.csv"))
    if len(metric_files) != 1:
        return {"status": "No unique CLI metrics file; inspect retained logs."}
    metrics = pd.read_csv(metric_files[0]).iloc[0]
    exported = pd.read_csv(out / "predictions.csv")
    records = audit["test"]["records"]
    if len(exported) != len(records) or not np.isfinite(exported["Predicted Activity"]).all():
        raise ValueError("Reference did not export one finite prediction per external molecule.")
    # These hashes were generated before SMILES removal from the external CSV.
    if exported.graph_sha256.tolist() != [row["graph_sha256"] for row in records]:
        raise ValueError("External prediction identities/order differ from the supplied test set.")
    actual = np.array([row["activity"] for row in records])
    prediction = exported["Predicted Activity"].to_numpy()
    errors = prediction - actual
    pd.DataFrame(
        {
            "name": [row["name"] for row in records],
            "observed": actual,
            "prediction": prediction,
            "prediction_minus_observed": errors,
            "squared_error": errors**2,
        }
    ).to_csv(out / "external_predictions_checked.csv", index=False)
    relative = metric_files[0].parent
    n_train = len(pd.read_csv(relative / "Training_Predictions_and_Residuals.csv"))
    n_internal_test = len(pd.read_csv(relative / "Test_Predictions_and_Residuals.csv"))
    return {
        "status": "Reference executed; full publication reproduction NOT established",
        "reference_q2": float(metrics.q2),
        "paper_q2": protocol["paper_targets"]["q2"],
        "q2_within_predeclared_tolerance": bool(
            abs(metrics.q2 - protocol["paper_targets"]["q2"])
            <= protocol["predeclared_tolerances"]["q2_absolute"]
        ),
        "reported_training_r2": float(metrics.r2_train),
        "reported_internal_test_r2": float(metrics.r2_test),
        "reported_training_count": n_train,
        "reported_internal_test_count": n_internal_test,
        "supplied_external_count": len(actual),
        "supplied_external_r2_recomputed": float(r2_score(actual, prediction)),
        "supplied_external_rmse_recomputed": float(np.sqrt(np.mean(errors**2))),
        "paper_training_r2": protocol["paper_targets"]["train_r2"],
        "training_r2_within_predeclared_tolerance": bool(
            abs(metrics.r2_train - protocol["paper_targets"]["train_r2"])
            <= protocol["predeclared_tolerances"]["train_r2_absolute"]
        ),
        "warning": "The CLI internal split is NOT the supplied external set. "
        "Its labelled test R2 must not be reported as external R2.",
        "sarscope_field_equivalence_established": False,
        "comfa_reproduction_established": False,
        "hydrogen_bond_field_reproduction_established": False,
    }


def run_probe(python: Path, out: Path, timeout: float = 180) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; preserve earlier evidence.")
    out.mkdir(parents=True, exist_ok=True)
    protocol_bytes = (ROOT / "reference-selection.json").read_bytes()
    protocol = json.loads(protocol_bytes)
    response = httpx.get(protocol["archive_url"], timeout=40, follow_redirects=True)
    response.raise_for_status()
    verify(response.content, protocol["archive_sha256"])
    (out / "reference-selection.json").write_bytes(protocol_bytes)
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "MPLBACKEND": "Agg",
            "PYVISTA_OFF_SCREEN": "true",
            "QT_QPA_PLATFORM": "offscreen",
        }
    )
    # No SARscope import path or user Python startup script enters the comparator.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONSTARTUP", None)
    with tempfile.TemporaryDirectory(prefix="sarscope-external-reference-") as scratch:
        with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as archive:
            archive.extractall(scratch, filter="data")
        checkout = Path(scratch) / f"pycomsia-{protocol['commit']}"
        audit = {}
        for split in ("train", "test"):
            metadata = protocol["dataset"][split]
            payload = (checkout / metadata["path"]).read_bytes()
            verify(payload, metadata["sha256"])
            audit[split] = audit_sdf(payload, protocol["dataset"]["activity_property"])
            if (
                audit[split]["valid_records"] != metadata["valid_records"]
                or len(audit[split]["invalid_records"]) != metadata["invalid_records"]
            ):
                raise ValueError(f"Reference {split} record accounting changed.")
        (out / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
        hashes = {
            str(path.relative_to(checkout)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(checkout.rglob("*"))
            if path.is_file()
        }
        (out / "external_file_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
        # Resolving a venv Python symlink would bypass that environment.
        args = command(python.absolute(), checkout, protocol["cli_settings"])
        with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
            try:
                result = subprocess.run(
                    args,
                    cwd=checkout,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                    check=False,
                )
                returncode, timed_out = result.returncode, False
            except subprocess.TimeoutExpired:
                returncode, timed_out = None, True
        # Retain numerical outputs, not upstream structures or program code.
        output_inventory = {}
        for path in sorted(checkout.rglob("*.csv")):
            relative = path.relative_to(checkout)
            if str(relative) in hashes:
                continue
            raw = path.read_bytes()
            frame = pd.read_csv(io.BytesIO(raw))
            if "SMILES" in frame:
                frame["graph_sha256"] = [
                    graph_hash(Chem.MolFromSmiles(value)) for value in frame.SMILES
                ]
                frame = frame.drop(columns="SMILES")
            destination = out / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(destination, index=False)
            output_inventory[str(relative)] = {
                "original_sha256": hashlib.sha256(raw).hexdigest(),
                "retained_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
                "note": "Structure strings removed; numeric CSV data retained.",
            }
        (out / "external_output_inventory.json").write_text(
            json.dumps(output_inventory, indent=2) + "\n"
        )
        if returncode == 0:
            assessment = assess_outputs(out, audit, protocol)
            (out / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
        # Record installed distribution versions without inspecting external program code.
        version_script = (
            "import importlib.metadata as m,json,sys; "
            "print(json.dumps({'python':sys.version,'packages':"
            "{d.metadata['Name']:d.version for d in m.distributions()}},sort_keys=True))"
        )
        versions = subprocess.run(
            [str(python.absolute()), "-c", version_script],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
            env=env,
        )
        (out / "environment.json").write_text(versions.stdout)
        run = {
            "created_utc": datetime.now(UTC).isoformat(),
            "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "selection_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
            "command": [part.replace(str(checkout), "<external-checkout>") for part in args],
            "reference_commit": protocol["commit"],
            "returncode": returncode,
            "timed_out": timed_out,
            "timeout_seconds": timeout,
            "reference_program_modified": False,
            "reference_source_consulted_for_implementation": False,
            "numerical_reproduction_established": False,
            "note": "Process success is not numerical equivalence; compare retained outputs.",
        }
        (out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    manifest = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    (out / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True, help="Separate comparator Python")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()
    result = run_probe(args.python, args.out, args.timeout)
    print(json.dumps(result, indent=2))
    if result["returncode"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

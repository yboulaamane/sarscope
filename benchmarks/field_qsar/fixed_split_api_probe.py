"""Execute the reference's explicit-split public API on independently generated SEH fields.

API signatures/docstrings and numerical outputs only; no implementation source
is read or translated. Historical preprocessing stays outside the application.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def worker(checkout: Path, out: Path, source: str) -> None:
    sys.path.insert(0, str(checkout))
    from src.PLSAnalysisTestSets import PLSAnalysis

    protocol = json.loads((ROOT / "fixed-split-api-protocol.json").read_text())
    with np.load(ROOT / protocol["field_arrays"] / f"{source}.npz") as arrays:
        train = {f"{name}_field": arrays[name][:21] for name in protocol["fields"]}
        test = {f"{name}_field": arrays[name][21:] for name in protocol["fields"]}
    audit = json.loads((ROOT / protocol["labels"]).read_text())
    y = np.array([row["activity"] for row in audit["train"]["records"]])
    yt = np.array([row["activity"] for row in audit["test"]["records"]])
    model = PLSAnalysis()
    model.convert_fields_to_X(train, test, filter=protocol["column_filter"])
    model.perform_loo_analysis(y, max_components=protocol["max_components"])
    model.fit_final_model(y, yt)
    (out / "PLS_Analysis").mkdir(parents=True)
    model.export_metrics_to_csv(str(out))
    model.export_predictions_and_residuals(str(out))
    # Numerical state exported through the public API, not source inspection.
    np.savez_compressed(out / "processed_fields.npz", train=model.X_train, test=model.X_test)
    (out / "api_state.json").write_text(
        json.dumps(
            {
                "optimal_components": int(model.optimal_n_components),
                "q2_scores": [float(x) for x in model.q2_scores],
                "estimator_parameters": {
                    key: value.item() if isinstance(value, np.generic) else value
                    for key, value in model.pls_model.get_params().items()
                },
                "python": sys.version,
                "packages": {
                    name: importlib.metadata.version(name)
                    for name in ("numpy", "rdkit", "scikit-learn")
                },
            },
            indent=2,
        )
        + "\n"
    )


def run(python: Path, out: Path) -> dict[str, Any]:
    import httpx

    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory.")
    out.mkdir(parents=True, exist_ok=True)
    out = out.absolute()
    protocol_path = ROOT / "fixed-split-api-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    selection = json.loads((ROOT / protocol["reference_selection"]).read_text())
    for folder in (ROOT / protocol["field_arrays"], (ROOT / protocol["labels"]).parent):
        manifest = json.loads((folder / "manifest.sha256.json").read_text())
        for name, expected in manifest.items():
            if hashlib.sha256((folder / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Frozen input changed; stop rather than silently rerun.")
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    payload = httpx.get(selection["archive_url"], follow_redirects=True, timeout=40)
    payload.raise_for_status()
    if hashlib.sha256(payload.content).hexdigest() != selection["archive_sha256"]:
        raise ValueError("External archive hash mismatch.")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="sarscope-fixed-split-") as scratch:
        with tarfile.open(fileobj=io.BytesIO(payload.content), mode="r:gz") as archive:
            archive.extractall(scratch, filter="data")
        checkout = Path(scratch) / f"pycomsia-{selection['commit']}"
        for source in protocol["sources"]:
            folder = out / source
            folder.mkdir()
            with (
                (folder / "stdout.txt").open("w") as stdout,
                (folder / "stderr.txt").open("w") as stderr,
            ):
                subprocess.run(
                    [
                        str(python.absolute()),
                        str(Path(__file__).resolve()),
                        "--checkout",
                        str(checkout),
                        "--source",
                        source,
                        "--out",
                        str(folder),
                    ],
                    cwd=scratch,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=120,
                    check=True,
                )
    rows = {}
    for source in protocol["sources"]:
        folder = out / source
        metrics = pd.read_csv(folder / "PLS_Analysis/PLS_Metrics.csv").iloc[0]
        rows[source] = metrics.to_dict()
        rows[source]["paper_q2_within_tolerance"] = bool(
            abs(metrics.q2 - protocol["paper_targets"]["q2"])
            <= protocol["tolerances"]["paper_q2_absolute"]
        )
        rows[source]["paper_train_r2_within_tolerance"] = bool(
            abs(metrics.r2_train - protocol["paper_targets"]["train_r2"])
            <= protocol["tolerances"]["paper_train_r2_absolute"]
        )
    differences = {}
    for split in ("Training", "Test"):
        path = f"PLS_Analysis/{split}_Predictions_and_Residuals.csv"
        a = pd.read_csv(out / "reference" / path)
        b = pd.read_csv(out / "sarscope" / path)
        if len(a) != (21 if split == "Training" else 10) or not np.array_equal(a.Index, b.Index):
            raise ValueError("Explicit split row identities changed.")
        differences[split] = float(np.max(np.abs(a["Predicted Values"] - b["Predicted Values"])))
    result = {
        "reference_commit": selection["commit"],
        "archive_sha256": selection["archive_sha256"],
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                ROOT / protocol["field_arrays"] / "manifest.sha256.json",
                ROOT / protocol["labels"],
            )
        },
        "results": rows,
        "cross_source_prediction_max_absolute_differences": differences,
        "same_historical_api_not_independent_pls_implementation": True,
        "production_preprocessing_changed": False,
        "coMFA_or_hbond_field_reproduction": False,
        "independent_predictive_validation": False,
    }
    (out / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    (out / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--source", choices=["reference", "sarscope"])
    args = parser.parse_args()
    if args.source:
        worker(args.checkout, args.out, args.source)
    else:
        if args.python is None:
            parser.error("--python is required")
        print(json.dumps(run(args.python, args.out), indent=2))


if __name__ == "__main__":
    main()

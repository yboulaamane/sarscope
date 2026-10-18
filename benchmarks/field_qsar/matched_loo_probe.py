"""Compare published-style LOO through isolated unmodified reference APIs.

Historical combined-set scaling and test-label-dependent exports are diagnostics,
never an app workflow or evidence of prospective prediction. No source copying.
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

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
FIELDS = ("steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def worker(checkout: Path, data: Path, own: Path, out: Path) -> None:
    sys.path.insert(0, str(checkout))
    from src.MolecularFieldCalculator import MolecularFieldCalculator
    from src.MolecularGridCalculator import MolecularGridCalculator
    from src.PLSAnalysisTestSets import PLSAnalysis

    molecules = []
    for split in ("train", "test"):
        items = list(Chem.SDMolSupplier(str(data / f"THR_{split}.sdf"), removeHs=False))
        assert len(items) == (59 if split == "train" else 29) and all(m is not None for m in items)
        molecules.extend((m, split == "train") for m in items)
    target = np.array([float(m.GetProp("Activity")) for m, _ in molecules])
    identifiers = [m.GetProp("_Name") for m, _ in molecules]
    own_state = json.loads((own / "assessment.json").read_text())
    spacing, shape, origin = MolecularGridCalculator().generate_grid(
        molecules, resolution=1.0, padding=4.0
    )
    assert list(shape) == own_state["grid"]["shape"]
    assert np.array_equal(origin, own_state["grid"]["origin"])
    raw = MolecularFieldCalculator().calc_field(molecules, spacing, shape, origin)
    arrays = {
        n: np.asarray(raw["train_fields"][f"{n}_field"] + raw["pred_fields"][f"{n}_field"]).reshape(
            88, -1
        )
        for n in FIELDS
    }
    with np.load(own / "own_arrays.npz") as own_arrays:
        actual = own_arrays["raw_fields"].reshape(88, 5, -1)
    np.savez_compressed(out / "reference_fields.npz", **arrays)
    summaries = []
    for subset, names in (("SEH", FIELDS[:3]), ("SEHDA", FIELDS)):
        for source in ("reference", "sarscope"):
            blocks = (
                arrays if source == "reference" else {n: actual[:, i] for i, n in enumerate(FIELDS)}
            )
            training = {f"{n}_field": blocks[n][:59] for n in names}
            testing = {f"{n}_field": blocks[n][59:] for n in names}
            case = out / f"{subset}__{source}"
            (case / "PLS_Analysis").mkdir(parents=True)
            model = PLSAnalysis()
            model.convert_fields_to_X(training, testing, filter=0.0)
            model.perform_loo_analysis(target[:59], max_components=12)
            model.fit_final_model(target[:59], target[59:])
            model.export_metrics_to_csv(str(case))
            model.export_predictions_and_residuals(str(case))
            exported = pd.read_csv(case / "PLS_Analysis/Test_Predictions_and_Residuals.csv")
            assert np.allclose(exported["Actual Values"], target[59:]) and len(exported) == 29
            values = exported["Predicted Values"].to_numpy()
            centered = np.asarray(model.pls_model.predict(model.X_test)).reshape(-1)
            centered_train = np.asarray(model.pls_model.predict(model.X_train)).reshape(-1)
            # The public API exposes an estimator trained on centred targets.
            # Reconstruct using the TRAINING target mean only, and verify
            # against its training exports before interpreting test values.
            training_export = pd.read_csv(
                case / "PLS_Analysis/Training_Predictions_and_Residuals.csv"
            )["Predicted Values"].to_numpy()
            training_mean = float(target[:59].mean())
            train_direct = centered_train + training_mean
            direct = centered + training_mean
            if not np.allclose(training_export, train_direct, atol=1e-10, rtol=0):
                raise ValueError("Unknown reference target transformation; no test-based repair.")
            pd.DataFrame(
                {
                    "molecule_id": identifiers[59:],
                    "observed": target[59:],
                    "exported_prediction": values,
                    "estimator_prediction": direct,
                    "centered_estimator_output": centered,
                }
            ).to_csv(case / "checked_external_predictions.csv", index=False)
            pd.DataFrame(
                {
                    "molecule_id": identifiers[:59],
                    "observed": target[:59],
                    "estimator_prediction": train_direct,
                    "centered_estimator_output": centered_train,
                }
            ).to_csv(case / "checked_train_predictions.csv", index=False)
            row = {
                "subset": subset,
                "field_generator": source,
                "components": int(model.optimal_n_components),
                "loo_q2": float(max(model.q2_scores)),
                "train_r2_recalculated": float(r2_score(target[:59], train_direct)),
                "export_test_r2": float(r2_score(target[59:], values)),
                "export_test_rmse": float(np.sqrt(mean_squared_error(target[59:], values))),
                "estimator_test_r2": float(r2_score(target[59:], direct)),
                "estimator_test_rmse": float(np.sqrt(mean_squared_error(target[59:], direct))),
            }
            pd.DataFrame(
                {"components": np.arange(1, len(model.q2_scores) + 1), "loo_q2": model.q2_scores}
            ).to_csv(case / "component_curve.csv", index=False)
            # Keep component selection unchanged; refit only with a test-label
            # offset to audit the exported outputs' dependence on held-out truth.
            shifted = case / "test_labels_plus10"
            (shifted / "PLS_Analysis").mkdir(parents=True)
            model.fit_final_model(target[:59], target[59:] + 10)
            model.export_predictions_and_residuals(str(shifted))
            after = pd.read_csv(shifted / "PLS_Analysis/Test_Predictions_and_Residuals.csv")[
                "Predicted Values"
            ].to_numpy()
            change = after - values
            row["export_test_label_invariance_passes"] = bool(
                np.allclose(change, 0, atol=1e-10, rtol=0)
            )
            row["export_mean_change_when_test_labels_plus10"] = float(change.mean())
            save(case / "assessment.json", row)
            summaries.append(row)
            print(subset, source, row, flush=True)
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    a = pd.read_csv(
        out / "SEHDA__sarscope/checked_external_predictions.csv", dtype={"molecule_id": str}
    )
    b = pd.read_csv(own / "predictions.csv", dtype={"molecule_id": str}).query("split == 'test'")
    assert a.molecule_id.tolist() == b.molecule_id.tolist()
    # Only compare the exposed estimator outputs, never the label-shifted exports.
    save(
        out / "independent_estimator_comparison.json",
        {
            "max_abs_difference": float(
                np.max(np.abs(a.estimator_prediction.to_numpy() - b.prediction.to_numpy()))
            ),
            "scope": (
                "Own five-field historical implementation versus reference estimator "
                "on own fields, restoring only the training target mean"
            ),
            "not_native_predictive_validation": True,
        },
    )
    save(
        out / "environment.json",
        {
            "python": sys.version,
            "packages": {
                p: importlib.metadata.version(p)
                for p in ("numpy", "rdkit", "scikit-learn", "pandas", "scipy")
            },
            "grid": {
                "shape": list(shape),
                "origin": [float(v) for v in origin],
                "spacing": list(spacing),
            },
        },
    )


def run(python: Path, data: Path, own: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a fresh output directory.")
    out.mkdir(parents=True, exist_ok=True)
    out, data, own = out.resolve(), data.resolve(), own.resolve()
    selection = json.loads((ROOT / "reference-selection.json").read_text())
    source_protocol = json.loads((ROOT / "loo-thrombin-20261018-protocol.json").read_text())
    for split in ("train", "test"):
        if sha(data / f"THR_{split}.sdf") != source_protocol["data"][split]["sha256"]:
            raise ValueError("Frozen source inputs differ.")
    save(
        out / "protocol.json",
        {
            "reference": selection,
            "own_loo_protocol": source_protocol,
            "max_components": 12,
            "column_filter": 0.0,
            "field_subsets": ["SEH", "SEHDA"],
            "runtime": str(python.resolve()),
            "runner_sha256": sha(Path(__file__)),
            "own_raw_fields_sha256": sha(own / "own_arrays.npz"),
            "scope": (
                "Historical matched-LOO diagnostic; radical-encoded inputs not "
                "qualified for native models"
            ),
            "prediction_audit": (
                "Compare training-mean-restored estimator outputs and exports separately; "
                "shift only test labels by+10"
            ),
        },
    )
    payload = httpx.get(selection["archive_url"], timeout=45, follow_redirects=True)
    payload.raise_for_status()
    assert hashlib.sha256(payload.content).hexdigest() == selection["archive_sha256"]
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", MPLBACKEND="Agg")
    with tempfile.TemporaryDirectory(prefix="sarscope-matched-loo-") as scratch:
        with tarfile.open(fileobj=io.BytesIO(payload.content), mode="r:gz") as archive:
            archive.extractall(scratch, filter="data")
        checkout = Path(scratch) / ("pycomsia-" + selection["commit"])
        with (out / "stdout.txt").open("w") as stdout, (out / "stderr.txt").open("w") as stderr:
            completed = subprocess.run(
                [
                    str(python.absolute()),
                    str(Path(__file__).resolve()),
                    "--checkout",
                    str(checkout),
                    "--data",
                    str(data),
                    "--own",
                    str(own),
                    "--out",
                    str(out),
                ],
                cwd=scratch,
                env=env,
                stdout=stdout,
                stderr=stderr,
                timeout=900,
                check=False,
            )
        if completed.returncode:
            save(out / "failure.json", {"returncode": completed.returncode, "logs_retained": True})
            raise RuntimeError("Reference execution failed; inspect retained stderr.txt.")
    save(
        out / "manifest.sha256.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.sha256.json"
        },
    )
    print((out / "summary.csv").read_text())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--checkout", type=Path)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--own", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.checkout:
        worker(args.checkout, args.data, args.own, args.out)
    else:
        run(args.python, args.data, args.own, args.out)

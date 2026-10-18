"""Verify completed validation exports and build a small paper evidence snapshot.

No models are fitted and no configurations are selected. All headline native
metrics are recomputed from row predictions, independently of the model code.
Chemical structures, source labels and external implementation code are not
redistributed by this aggregate-only snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def verify_manifest(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        if digest((root / name).read_bytes()) != expected:
            raise ValueError(f"Manifest mismatch: {root / name}")
    return manifest


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = truth - prediction
    return {
        "r2": float(1 - (error**2).sum() / ((truth - truth.mean()) ** 2).sum()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.abs(error).mean()),
        "spearman": float(spearmanr(truth, prediction).statistic),
    }


def native_case(name: str, folder: Path, out: Path) -> tuple[dict, pd.DataFrame]:
    predictions = pd.read_csv(folder / "predictions.csv", dtype={"molecule_id": str})
    stats = pd.read_csv(folder / "statistics.csv").set_index("algorithm")
    actual = stats.loc["field_pls"]
    train = predictions.query("split == 'train'")
    test = predictions.query("split == 'held_out'")
    if len(train) + len(test) != len(predictions) or predictions.molecule_id.duplicated().any():
        raise ValueError("Unaccounted prediction rows or duplicate identifiers.")
    q2 = metrics(train.pactivity.to_numpy(), train.training_oof_prediction.to_numpy())["r2"]
    scores = metrics(test.pactivity.to_numpy(), test.prediction.to_numpy())
    if not np.isclose(q2, actual.cv_pooled_q2, atol=1e-10, rtol=0):
        raise ValueError("Pooled nested q2 differs from row-level recomputation.")
    for key, value in scores.items():
        if not np.isclose(value, actual[f"test_{key}"], atol=1e-10, rtol=0):
            raise ValueError(f"Held-out {key} differs from row-level recomputation.")
    # The legacy CSV column is the RF comparator, not the constant baseline.
    rf = metrics(test.pactivity.to_numpy(), test.baseline_prediction.to_numpy())
    for key in ("r2", "rmse"):
        if not np.isclose(rf[key], stats.loc["random_forest", f"test_{key}"], atol=1e-10):
            raise ValueError(f"RF {key} differs from row-level recomputation.")
    baseline = np.full(len(test), train.pactivity.mean())
    mean_rmse = float(np.sqrt(np.mean((test.pactivity.to_numpy() - baseline) ** 2)))
    if not np.isclose(mean_rmse, stats.loc["mean_baseline", "test_rmse"], atol=1e-10):
        raise ValueError("Constant baseline differs from training-mean recomputation.")
    test = test.assign(training_mean_prediction=baseline)
    null = pd.read_csv(folder / "permutations.csv")
    pvalue = float((1 + (null.q2 >= q2).sum()) / (1 + len(null)))
    if len(null) != 99 or not np.isclose(pvalue, actual.permutation_pvalue, atol=1e-10, rtol=0):
        raise ValueError("Permutation accounting differs.")
    row = {
        "dataset": name,
        "train": len(train),
        "test": len(test),
        "nested_pooled_q2": q2,
        **{f"test_{key}": value for key, value in scores.items()},
        "components": int(actual.selected_components),
        "permutations": len(null),
        "permutation_p": pvalue,
        "rf_test_r2": rf["r2"],
        "rf_test_rmse": rf["rmse"],
        "mean_test_rmse": mean_rmse,
        "native_strict_checks_pass": bool(pd.read_csv(folder / "checks.csv").passed.all()),
    }
    target = out / name
    target.mkdir()
    for filename in ("statistics.csv", "checks.csv", "bootstrap.csv", "permutations.csv"):
        shutil.copyfile(folder / filename, target / filename)
    return row, test


def summarize(runs: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use an empty destination; do not overwrite earlier evidence.")
    out.mkdir(parents=True, exist_ok=True)
    thr = runs / "primary-thrombin-native-v2"
    ache = runs / "primary-ache-reproducible-driver"
    reference = runs / "thrombin-reference-matched-loo-v2/verified-exports"
    reference_repeat = runs / "thrombin-reference-reproducible-driver"
    verified = {}
    for label, root in (("THR", thr), ("AChE", ache), ("reference", reference)):
        verified[label] = {"path": str(root), "files_verified": len(verify_manifest(root))}
    verified["reference_repeat"] = {
        "path": str(reference_repeat),
        "files_verified": len(verify_manifest(reference_repeat)),
    }
    earlier = pd.read_csv(reference / "summary.csv")
    repeat = pd.read_csv(reference_repeat / "summary.csv")
    if not np.allclose(
        earlier[["loo_q2", "test_r2_training_mean_only", "test_r2_historical_export"]],
        repeat[["loo_q2", "estimator_test_r2", "export_test_r2"]],
        atol=1e-10,
        rtol=0,
    ):
        raise ValueError("Reusable reference driver did not reproduce verified numerical results.")
    main = thr / "fivefield_1A_block_primary"
    with ZipFile(main / "sarscope_evidence.zip") as archive:
        manifest = json.loads(archive.read("manifest.sha256.json"))
        for name, expected in manifest.items():
            if digest(archive.read(name)) != expected:
                raise ValueError(f"Evidence ZIP checksum mismatch: {name}")
        source_check = pd.read_csv(io.BytesIO(archive.read("source_graph_audit.csv")))
        if len(source_check) != 88 or source_check.radical_electrons.ne(0).any():
            raise ValueError("Native evidence did not retain all closed-shell source audits.")
        verified["THR_ZIP"] = {"files_verified": len(manifest), "closed_shell_rows": 88}
    audit = pd.read_csv(thr / "source_adjudication.csv", dtype={"molecule_id": str})
    if (
        len(audit) != 88
        or audit.primary_radical_electrons.ne(0).any()
        or int(audit.later_radical_electrons.gt(0).sum()) != 59
        or audit.heavy_atom_coordinate_max_change_A.ne(0).any()
        or not audit.heavy_atom_connectivity_identical.all()
    ):
        raise ValueError("Primary-source adjudication no longer agrees with all 88 rows.")
    save(
        out / "source_adjudication.json",
        {
            "records": len(audit),
            "activities_unchanged": True,
            "heavy_atom_coordinates_and_connectivity_identical": True,
            "later_radical_graphs": int(audit.later_radical_electrons.gt(0).sum()),
            "primary_radical_graphs": int(audit.primary_radical_electrons.gt(0).sum()),
            "rows_with_changed_bond_orders": int(audit.bond_orders_different.gt(0).sum()),
            "rows_with_changed_formal_charge": int(
                (audit.later_formal_charge != audit.primary_formal_charge).sum()
            ),
            "policy": "Use original source graphs, not inferred repairs or score-based exclusions",
        },
    )
    rows, tables = [], []
    for name, root in (("THR", main), ("AChE", ache / "AChE/gaussian_fivefield_primary")):
        row, table = native_case(name, root, out)
        rows.append(row)
        tables.append(table)
    pd.DataFrame(rows).to_csv(out / "native_summary.csv", index=False)
    for filename in ("stability.csv", "test_label_invariance.json"):
        shutil.copyfile(main / filename, out / "THR" / filename)
    shutil.copyfile(reference / "summary.csv", out / "matched_loo.csv")
    shutil.copyfile(reference / "assessment.json", out / "matched_loo_assessment.json")
    for filename in ("summary.csv", "protocol.json", "independent_estimator_comparison.json"):
        shutil.copyfile(reference_repeat / filename, out / f"reference_repeat_{filename}")
    for name, root in (("THR", thr), ("AChE", ache)):
        shutil.copyfile(root / "protocol.json", out / name / "protocol.json")
    junit = ET.parse(runs / "targeted-tests.xml").getroot()
    suites = [junit] if junit.tag == "testsuite" else list(junit.iter("testsuite"))
    tests = {
        key: sum(int(s.get(key, 0)) for s in suites)
        for key in ("tests", "errors", "failures", "skipped")
    }
    if tests["errors"] or tests["failures"]:
        raise ValueError("Targeted tests include failures.")
    shutil.copyfile(runs / "targeted-tests.xml", out / "targeted-tests.xml")
    offline_tests = None
    if (runs / "offline-suite.xml").exists():
        offline = ET.parse(runs / "offline-suite.xml").getroot()
        suites = [offline] if offline.tag == "testsuite" else list(offline.iter("testsuite"))
        offline_tests = {
            key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "errors", "failures", "skipped")
        }
        if offline_tests["errors"] or offline_tests["failures"]:
            raise ValueError("Offline suite contains failures; retain evidence, do not hide them.")
        shutil.copyfile(runs / "offline-suite.xml", out / "offline-suite.xml")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.1), constrained_layout=True)
    for ax, row, table in zip(axes, rows, tables, strict=True):
        observed, predicted = table.pactivity, table.prediction
        limits = [
            min(observed.min(), predicted.min()) - 0.15,
            max(observed.max(), predicted.max()) + 0.15,
        ]
        ax.plot(
            limits, limits, color="#475569", linestyle="--", linewidth=1, label="Ideal prediction"
        )
        ax.axhline(
            table.training_mean_prediction.iloc[0],
            color="#d97706",
            linestyle=":",
            label="Training-mean baseline",
        )
        ax.scatter(
            observed, predicted, c="#0f766e", s=30, alpha=0.8, edgecolors="white", linewidths=0.4
        )
        ax.set(
            xlim=limits,
            ylim=limits,
            xlabel="Observed source log activity",
            ylabel="Predicted log activity",
            title=(
                f"{row['dataset']} · historical held-out n={row['test']}\n"
                f"R²={row['test_r2']:.3f}; RMSE={row['test_rmse']:.3f}"
            ),
        )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(fontsize=7, loc="upper left")
    fig.savefig(out / "held_out_predictions.svg")
    fig.savefig(out / "held_out_predictions.png", dpi=180)
    plt.close(fig)
    save(
        out / "verification.json",
        {
            "manifests": verified,
            "tests": tests,
            "offline_tests": offline_tests,
            "reference_reusable_driver_metrics_reproduced": True,
            "native_row_metrics_recomputed": True,
            "no_model_refitting": True,
            "package_fully_validated": False,
            "prospective_or_blinded": False,
            "generator_sha256": digest(Path(__file__).read_bytes()),
        },
    )
    save(
        out / "manifest.sha256.json",
        {
            str(p.relative_to(out)): digest(p.read_bytes())
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.sha256.json"
        },
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.runs, args.out)

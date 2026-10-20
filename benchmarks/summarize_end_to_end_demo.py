"""Audit and present the frozen worked example, retaining failed alternatives."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from rdkit import Chem

from benchmarks.validate_end_to_end_demo import params
from sarscope import provenance
from sarscope.analysis.fields import GridSpec, compute_fields
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_campaign import _json, _manifest


def independent_prediction_audit(folder: Path) -> dict:
    """Generate held-out descriptors from exported molecules, without labels or refitting."""
    with zipfile.ZipFile(folder / "complete_evidence.zip") as zipped:
        record = json.loads(zipped.read("provenance.json"))
        expected = pd.read_csv(io.BytesIO(zipped.read("predictions.csv")))
        molecules = list(
            Chem.ForwardSDMolSupplier(io.BytesIO(zipped.read("aligned.sdf")), removeHs=False)
        )
        test = np.flatnonzero(expected.split.eq("held_out"))
        if len(molecules) != len(expected) or any(m is None for m in molecules):
            raise ValueError("Molecule accounting differs")
        held_out = [molecules[int(i)] for i in test]
        for mol in held_out:
            for key in mol.GetPropNames():
                mol.ClearProp(key)
        raw_grid = record["grid"]
        grid = GridSpec(tuple(raw_grid["origin"]), tuple(raw_grid["shape"]), raw_grid["spacing"])
        fresh = compute_fields(held_out, grid, params(record["field_qsar"])).matrix
        with np.load(io.BytesIO(zipped.read("fields.npz")), allow_pickle=False) as saved:
            mask = saved["retained_mask"]
            transformed = fresh[:, mask] / saved["column_scale"][mask]
            prediction = (
                transformed @ saved["coefficients"].reshape(-1)
                + saved["prediction_at_scaled_zero"].item()
            )
            field_change = float(np.max(np.abs(fresh - saved["fields"][test])))
    difference = float(np.max(np.abs(prediction - expected.iloc[test].prediction)))
    return {
        "held_out": len(test),
        "test_properties_removed_before_descriptor_calculation": True,
        "model_refit": False,
        "grid_rebuilt": False,
        "prediction_max_error": difference,
        "raw_field_max_error": field_change,
        "passed": difference < 1e-8,
    }


def paired_baseline(predictions: pd.DataFrame, repeats: int = 2000) -> dict:
    """Conditional scaffold-block interval for RF RMSE minus field RMSE."""
    test = add_scaffolds(predictions[predictions.split.eq("held_out")].reset_index(drop=True))
    groups = test.groupby("murcko", dropna=False).indices
    blocks = [np.asarray(v) for v in groups.values()]
    if len(blocks) < 2:
        return {"available": False, "reason": "Fewer than two scaffold blocks"}
    truth, field, rf = (
        test[column].to_numpy() for column in ("pactivity", "prediction", "baseline_prediction")
    )

    def delta(rows: np.ndarray) -> float:
        return float(
            np.sqrt(np.mean((truth[rows] - rf[rows]) ** 2))
            - np.sqrt(np.mean((truth[rows] - field[rows]) ** 2))
        )

    rng = np.random.default_rng(42)
    samples = [
        delta(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))
        for _ in range(repeats)
    ]
    return {
        "available": True,
        "rf_rmse_minus_field_rmse": delta(np.arange(len(test))),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "repeats": repeats,
        "scaffold_blocks": len(blocks),
        "seed": 42,
        "scope": "Conditional paired scaffold-block interval on one historical test set; "
        "not general method superiority",
    }


def summarize(source: Path, out: Path) -> None:
    manifest = json.loads((source / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        if provenance.file_sha256(source / name) != expected:
            raise ValueError("Frozen artifact changed: " + name)
    out.mkdir(parents=True, exist_ok=False)
    for name in ("protocol.json", "summary.csv"):
        shutil.copy2(source / name, out / name)
    shutil.copy2(source / "manifest.sha256.json", out / "raw_manifest.sha256.json")
    summary = pd.read_csv(source / "summary.csv")
    for row in summary.itertuples():
        target = out / row.case
        target.mkdir()
        folder = source / row.case
        for path in folder.glob("*.csv"):
            frame = pd.read_csv(path, dtype={"molecule_id": str})
            frame.drop(columns="smiles", errors="ignore").to_csv(target / path.name, index=False)
        for path in folder.glob("*.json"):
            shutil.copy2(path, target / path.name)
        if row.status == "completed":
            audit = independent_prediction_audit(folder)
            _json(target / "fresh_held_out_prediction_audit.json", audit)
            if not audit["passed"]:
                raise ValueError("Separately generated held-out predictions did not replay")
            predictions = pd.read_csv(folder / "predictions.csv")
            _json(target / "paired_rf_comparison.json", paired_baseline(predictions))
            with zipfile.ZipFile(folder / "complete_evidence.zip") as zipped:
                for name in zipped.namelist():
                    if name.startswith("figures/"):
                        (target / "figures").mkdir(exist_ok=True)
                        (target / name).write_bytes(zipped.read(name))
    main = summary[summary.case.eq("thrombin_source_alignment")].iloc[0]
    folder = out / main.case
    bootstrap = pd.read_csv(folder / "bootstrap.csv").set_index("metric")
    interval = bootstrap.loc["R²"]
    sensitivity = pd.read_csv(folder / "training_sensitivity.csv")
    audit = json.loads((folder / "fresh_held_out_prediction_audit.json").read_text())
    baseline = json.loads((folder / "paired_rf_comparison.json").read_text())
    figure = Figure(figsize=(9, 3.8), layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, kind, title in zip(
        axes,
        ("relative_pose", "grid_phase"),
        ("Small pose changes", "Half-grid sampling shifts"),
        strict=True,
    ):
        rows = sensitivity[sensitivity.kind.eq(kind)]
        axis.barh(
            rows.variant,
            rows.oof_change_p95,
            color=["#b64b4b" if value > 0.5 else "#24867b" for value in rows.oof_change_p95],
        )
        axis.axvline(0.5, color="#9b622c", linestyle="--", label="0.5-log engineering flag")
        axis.set(xlabel="95th-percentile change in training OOF prediction", title=title)
        axis.legend(fontsize=7)
    figure.savefig(out / "sensitivity.png", dpi=160)
    sections = [
        "<!doctype html><html lang='en'><meta charset='utf-8'>"
        "<title>SARscope · measured thrombin worked example</title>",
        "<style>body{font:16px/1.65 system-ui;color:#1f2933;max-width:1080px;margin:36px auto;"
        "padding:24px}h1,h2{color:#155d59}table{border-collapse:collapse;max-width:100%;"
        "display:block;overflow:auto;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd}"
        "img{max-width:100%}.note{background:#fff2dd;border-left:5px solid #b77728;padding:16px}"
        ".pass{background:#e8f4ef;border-left:5px solid #24867b;padding:16px}"
        "code{background:#edf2f2;padding:3px}</style>",
        "<h1>Measured thrombin · end-to-end 3D-QSAR worked example</h1>",
        "<p>88 measured compounds · original 59/29 partition · supplied source alignment · "
        "SARscope-native MMFF94 fields and nested PLS · 99 label permutations</p>",
        "<p>Source and state checks → review supplied alignment → calculate training-grid fields "
        "→ nested selection → held-out prediction → uncertainty, domain and export audits.</p>",
        f"<p class='pass'>Positive fixed-alignment result: nested q² <b>{main.nested_q2:.3f}</b>; "
        f"held-out predictive R² <b>{main.test_r2:.3f}</b> "
        f"(conditional 95% interval {interval.lower_95:.3f}–{interval.upper_95:.3f}); "
        f"RMSE <b>{main.test_rmse:.3f}</b>; Spearman <b>{main.test_spearman:.3f}</b>; "
        f"permutation p = <b>{main.permutation_p:.2f}</b>.</p>",
        "<p class='note'>This is a retrospective demonstration on an already inspected public "
        "benchmark, not a new blinded study or package-wide certification. Predictions are "
        "conditional on the supplied alignment. Small-pose stress checks failed: use contours "
        "and potency gains as exploratory associations, not robust optimisation advice.</p>",
        "<h2>Held-out predictions and same-split baselines</h2>",
        "<p>Predictive R² is 1 − SSE/SST, not Pearson correlation squared. No test compound was "
        "removed and no test-label mean correction was applied.</p>",
        f"<p>3D RMSE {main.test_rmse:.3f}; 2D ECFP4 RF {main.rf_test_rmse:.3f}; "
        f"training-mean baseline {main.mean_rmse:.3f}. "
        f"RF minus 3D RMSE = {baseline['rf_rmse_minus_field_rmse']:.3f}, "
        f"paired scaffold-block 95% interval [{baseline['lower_95']:.3f}, "
        f"{baseline['upper_95']:.3f}]. One historical split is not general superiority.</p>",
        "<img src='thrombin_source_alignment/figures/predicted_vs_observed.png' "
        "alt='Observed and predicted activities for all held-out compounds'>",
        "<h2>Uncertainty and applicability</h2>",
        "<p>Intervals use 2,000 paired scaffold-block resamples across 24 held-out scaffold "
        "blocks, conditional on the fitted model and test population; they do not include "
        "alignment or experimental uncertainty.</p>",
        bootstrap.reset_index().to_html(index=False, float_format=lambda x: f"{x:.3f}"),
        f"<p>All {int(main.in_domain)}/{int(main.test)} test molecules fall inside the "
        "training-PCA box. This coarse descriptor-range diagnostic is not a calibrated "
        "confidence probability or evidence that every compound is reliably predictable.</p>",
        "<h2>What is stable—and what is not</h2>",
        "<p>Half-grid shifts and the common-translation control pass the declared numerical "
        "checks. Three 0.25 Å/5° relative-pose perturbations fail the original engineering "
        "limits. These use training compounds only and do not select a replacement alignment.</p>",
        "<img src='sensitivity.png' alt='Training-only pose and grid sensitivity'>",
        "<h2>Actual prediction reuse</h2>",
        "<p>Exported numerical coefficients, preprocessing and intercept reproduce every "
        "prediction without an executable pickle. A separate audit removed all compound "
        "properties and recalculated only the 29 held-out molecules' fields on the frozen "
        f"grid; maximum prediction discrepancy was {audit['prediction_max_error']:.2g}. "
        "Changing test activities by +100 left fitted predictions and training CV unchanged.</p>",
        "<h2>All declared routes—including the unsuccessful one</h2>",
        summary[[c for c in ("case", "status", "nested_q2", "test_r2", "error") if c in summary]]
        .fillna("")
        .to_html(index=False, escape=True),
        "<p>The CCR5 CSV route did not complete constrained embedding; no partial series "
        "or substitute alignment was scored. Thus this demo supports reviewed pre-aligned "
        "input, not a guaranteed SMILES-to-bioactive-alignment workflow.</p>",
        "<h2>Try it in Streamlit</h2><p>Open <b>3D QSAR · own compounds</b> → "
        "<b>Try a measured 3D-QSAR example</b> → <b>Thrombin</b>. Select the <b>MMFF94</b> "
        "worked-example profile, review the source terms, fetch the data, apply its settings, "
        "then prepare and review the overlay. Modelling requires a separate Run click. "
        "For a CSV-only upload, you must generate/review a new alignment; do not expect "
        "the supplied-frame scores to carry over.</p>",
        "<p>Source: <a href='https://doi.org/10.1021/jm0497141'>Sutherland et al. (2004)</a>; "
        "original series: <a href='https://doi.org/10.1021/jm981062r'>Böhm et al. (1999)</a>. "
        "Original ACTIVITY values/states and complete source partition retained. Supporting "
        "structures have separate CC BY-NC 4.0 terms and are not bundled here. "
        "No docking, ligand-state guessing or external QSAR solver was used.</p></html>",
    ]
    (out / "report.html").write_text("\n".join(sections))
    _json(
        out / "scope.json",
        {
            "positive_fixed_alignment_worked_example": bool(
                main.passes_worked_example_predictive_criteria
            ),
            "new_blinded_validation": False,
            "csv_route_success": bool(
                summary.loc[summary.case.eq("ccr5_generated_from_csv"), "status"]
                .eq("completed")
                .all()
            ),
            "pose_stress_qualified": bool(
                sensitivity.loc[
                    sensitivity.kind.eq("relative_pose"), "engineering_stability_flag"
                ].all()
            ),
            "package_certified": False,
            "source_structures_redistributed": False,
        },
    )
    _manifest(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summarize(args.source, args.out)

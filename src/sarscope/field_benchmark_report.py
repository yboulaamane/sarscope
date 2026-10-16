"""Readable benchmark evidence; show negative results without dropping compounds."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd
from matplotlib.figure import Figure


def write_benchmark_report(out: Path) -> Path:
    summary = pd.read_csv(out / "summary.csv")
    audit = pd.read_csv(out / "input_audit.csv")
    run = json.loads((out / "run.json").read_text())
    images = {}
    figure = Figure(figsize=(11, 4.5), layout="constrained")
    axes = figure.subplots(1, 2)
    for axis, preset in zip(axes, ("native_default", "seh_block_std_fixed3"), strict=True):
        for source, marker, color in (
            ("sdf_sign_corrected", "o", "#238c72"),
            ("article_table2", "x", "#9061b5"),
        ):
            predictions = pd.read_csv(out / f"{preset}__{source}" / "predictions.csv")
            test = predictions[predictions["split"] == "held_out"]
            axis.scatter(
                test.pactivity, test.prediction, marker=marker, color=color, label=source, alpha=0.8
            )
        axis.plot([4.5, 9], [4.5, 9], "--", color="grey", linewidth=1)
        axis.set(
            xlabel="Observed historical pActivity",
            ylabel="Held-out prediction",
            title=preset.replace("_", " "),
        )
        axis.legend(fontsize=8)
    path = out / "held_out_predictions.png"
    figure.savefig(path, dpi=150)
    images["Held-out predictions: no test compounds removed"] = path

    figure = Figure(figsize=(7, 4.5), layout="constrained")
    axis = figure.subplots()
    for source, color in (("sdf_sign_corrected", "#238c72"), ("article_table2", "#9061b5")):
        rows = pd.read_csv(out / f"native_default__{source}" / "permutations.csv")
        axis.hist(rows.q2, bins=15, histtype="step", color=color, label=source)
        observed = float(
            summary.query("preset == 'native_default' and activity_source == @source").iloc[0].q2
        )
        axis.axvline(observed, color=color, linestyle="--")
    axis.set(xlabel="Native nested-CV q² under label permutation", ylabel="Count")
    axis.legend(fontsize=8)
    path = out / "permutation_controls.png"
    figure.savefig(path, dpi=150)
    images["Native label-permutation controls (99 per label source)"] = path

    figure = Figure(figsize=(8, 4.5), layout="constrained")
    axis = figure.subplots()
    for preset, marker in (("seh_column_std_fixed3", "o"), ("sehad_column_std_fixed3", "x")):
        rows = pd.read_csv(out / f"{preset}__sdf_sign_corrected" / "predictions.csv")
        test = rows[rows["split"] == "held_out"]
        axis.scatter(test.pactivity, test.prediction, marker=marker, label=preset)
    axis.plot([4.5, 9], [4.5, 9], "--", color="grey", linewidth=1)
    axis.set_yscale("symlog", linthresh=10)
    axis.set(
        xlabel="Observed pActivity",
        ylabel="Predicted pActivity (signed-log axis)",
        title="Column scaling: pathological extrapolation — reject",
    )
    axis.legend(fontsize=8)
    path = out / "scaling_failure.png"
    figure.savefig(path, dpi=150)
    images["Scaling failure retained in the evidence"] = path

    numeric = [
        "preset",
        "activity_source",
        "q2",
        "train_r2",
        "test_r2",
        "test_rmse",
        "test_spearman",
        "components",
        "n_features",
        "permutation_pvalue",
        "published_q2",
    ]
    sections = [
        "<!doctype html><html><meta charset='utf-8'><title>SARscope steroid benchmark</title>",
        "<style>body{font:16px/1.55 system-ui;max-width:1200px;margin:30px auto;padding:20px}"
        "table{border-collapse:collapse;display:block;overflow:auto;font-size:12px}"
        "td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}"
        ".warn{border-left:5px solid #c96c26;padding:12px;background:#fff3e3}</style>",
        "<h1>SARscope · frozen steroid/CBG comparison</h1>",
        "<p class='warn'><b>No-Go for a general validation claim.</b> "
        "The literature comparison has been executed, but exact historical method "
        "reproduction and reliable held-out prediction have not been established.</p>",
        "<p>Eight prespecified comparisons: four method/scaling presets × two "
        "activity sources. Each uses 21 training and 10 fixed test compounds. "
        "The extra 32nd deposited compound is excluded by the published cohort, "
        "not because of a prediction error. No test outlier was removed.</p>",
        "<h2>Actual results</h2>",
        summary[numeric].to_html(
            index=False, escape=True, float_format=lambda value: f"{value:.6g}"
        ),
        "<p>Native q² is nested three-fold CV; the fixed-three-component reference-like "
        "q² values use LOO. These are different evaluation protocols, not a fair "
        "method ranking. Published reference values are 0.609 (Python SEH), 0.630 "
        "(Python SEHAD), and 0.665 (classic SEH as reported in the same paper).</p>",
        "<p>High internal q² and a label-permutation p-value of 0.01 do not establish "
        "external predictive performance. Column-wise scaling amplifies nearly "
        "constant training fields and can generate physically meaningless predictions. "
        "These failed conditions are reported, not treated as valid models.</p>",
        "<h2>Input adjudication</h2>",
        "<p>The SDF stores negative CBG activities; all 31 were checked against "
        "the published positive Table 2 after explicit sign inversion. Two differences "
        "exceed two-decimal rounding. Both label conventions are retained.</p>",
        audit.loc[
            audit.beyond_two_decimal_rounding,
            ["molecule_id", "sdf_pactivity", "article_pactivity", "sdf_minus_article"],
        ].to_html(index=False),
        "<p>Structure graph/stereochemistry metadata checks are internal consistency "
        "checks, not independent experimental structure verification. Supplied "
        "heavy-atom coordinates were retained. An exact common-core RMSD was not "
        "assessed; the declared external alignment is an input hypothesis.</p>",
    ]
    for title, path in images.items():
        encoded = base64.b64encode(path.read_bytes()).decode()
        sections.append(
            f"<h2>{html.escape(title)}</h2><img alt='{html.escape(title)}' "
            f"src='data:image/png;base64,{encoded}'>"
        )
    sections.extend(
        [
            "<h2>Reproducibility and limits</h2>",
            "<p>Sources, transformations, presets and hashes are in protocol.json, "
            "input_audit.csv, primary_reference.json and each case's provenance.json. "
            "Native runs contain all 99 permutation scores. Method/scaling settings "
            "were not chosen using test performance. Field parameterisation, "
            "grid/hydrogen handling, training-only scaling and atom-centred H-bond "
            "features differ from historical implementations. No GPL source was consulted.</p>",
            "<p>The pinned upstream SDF is not redistributed: its redistribution "
            "permission was not established. Published reference tables are attributed "
            "under the article's CC BY 4.0 licence. One historical target dataset is "
            "not an independent multi-target-family or prospective validation.</p>",
            f"<p>{html.escape(run['conclusion'])}</p>",
            "<p>Primary comparison source: Haga et al. (2025), "
            "<a href='https://doi.org/10.3390/ph18030440'>Py-CoMSIA, Pharmaceuticals 18:440</a>, "
            "Tables 1–2 and Methods 4.4–4.7. Table 1 is used because some narrative "
            "metric assignments conflict with the table.</p></html>",
        ]
    )
    destination = out / "report.html"
    destination.write_text("\n".join(sections))
    return destination

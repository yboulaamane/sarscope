"""Portable field-QSAR evidence bundles, with failures visible in the report."""

from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import zipfile
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope.analysis.contours import open_dx, pymol_script
from sarscope.field_interpretation import interpretation_review
from sarscope.field_qsar import FieldQsarResult, PreparedFieldSeries

METRIC_DEFINITIONS = {
    "cv_pooled_q2": (
        "Pooled nested outer-CV q²: 1 − sum of out-of-fold squared errors / "
        "sum of squared deviations from the training-set mean. Each training compound "
        "is predicted without fitting to its activity. This is the primary CV score."
    ),
    "cv_fold_mean_r2": (
        "Unweighted mean of predictive R² across outer CV folds. Each fold uses its own "
        "observed mean and SST. Small or narrow-potency folds can have tiny SST and "
        "very negative R²; this need not match pooled q². Not Pearson correlation squared."
    ),
    "cv_fold_sd_r2": "Sample standard deviation of the finite per-fold predictive R² values.",
    "cv_r2": "Legacy alias for cv_fold_mean_r2; not pooled q².",
    "cv_r2_sd": "Legacy alias for cv_fold_sd_r2.",
    "test_r2": "Held-out predictive R² = 1 − held-out SSE / held-out SST; can be negative.",
    "pearson_r2": (
        "Squared held-out Pearson correlation. Measures association, not predictive R² "
        "or absolute calibration. Origin-fit diagnostics also inspect k, k_prime and R0 values."
    ),
}


def charge_audit(result: FieldQsarResult) -> pd.DataFrame:
    """Link zero-based computational row indices to the visible molecule identifiers."""
    audit = pd.DataFrame(result.fields.charges).copy()
    if "molecule_id" not in audit:
        audit.insert(0, "molecule_id", result.prepared.dataset.table.molecule_id.to_numpy())
    if "reason" in audit:
        audit["reason"] = audit["reason"].replace("", "Not needed (no fallback)")
    return audit


def unmodelled_field_warning(summary: pd.DataFrame) -> str:
    """A zero-field contribution can reflect feature removal, not chemistry."""
    absent = summary.loc[summary.retained_columns.eq(0), "field"].astype(str).tolist()
    if not absent:
        return ""
    return (
        "Filtering retained no columns for: " + ", ".join(absent) + ". "
        "These fields do not contribute to this fitted model. Zero contribution is not "
        "evidence of no biological effect; unchanged predictions after changing charges "
        "cannot demonstrate charge robustness if electrostatic columns were removed."
    )


def aligned_sdf(prepared: PreparedFieldSeries, reference_only: bool = False) -> str:
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    indices = (
        [prepared.params.reference_index] if reference_only else range(len(prepared.dataset.table))
    )
    for index in indices:
        mol = Chem.Mol(prepared.alignment.molecules[index])
        row = prepared.dataset.table.iloc[index]
        mol.SetProp("_Name", str(row.molecule_id))
        mol.SetDoubleProp("pactivity", float(row.pactivity))
        writer.write(mol)
    writer.close()
    return buffer.getvalue()


def _plots(result: FieldQsarResult) -> dict[str, bytes]:
    from matplotlib.figure import Figure

    test = result.predictions[result.predictions["split"] == "held_out"]
    images = {}
    for kind in ("predicted_vs_observed", "residuals", "permutations"):
        figure = Figure(figsize=(6, 4), layout="constrained")
        axis = figure.subplots()
        if kind == "predicted_vs_observed":
            axis.scatter(test.pactivity, test.prediction, c="#1b9e77", label="3D field PLS")
            axis.scatter(
                test.pactivity,
                test.baseline_prediction,
                c="#b47ed8",
                marker="x",
                label="2D ECFP4 RF",
            )
            lo = min(test.pactivity.min(), test.prediction.min(), test.baseline_prediction.min())
            hi = max(test.pactivity.max(), test.prediction.max(), test.baseline_prediction.max())
            axis.plot([lo, hi], [lo, hi], "--", color="grey")
            axis.set(xlabel="Observed pActivity", ylabel="Held-out prediction")
            axis.legend()
        elif kind == "residuals":
            axis.scatter(test.prediction, test.residual, c="#d28738")
            axis.axhline(0, color="grey", linestyle="--")
            axis.set(xlabel="Held-out predicted pActivity", ylabel="Observed − predicted")
        else:
            if not result.permutations.empty:
                axis.hist(
                    result.permutations.q2, bins=min(15, len(result.permutations)), color="#a997cb"
                )
            q2 = float(result.statistics.iloc[0].cv_pooled_q2)
            axis.axvline(q2, color="#1b9e77", label="Real nested-CV q²")
            axis.set(xlabel="Training-only permutation q²", ylabel="Count")
            axis.legend()
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", dpi=130)
        images[kind] = buffer.getvalue()
    return images


def field_qsar_files(result: FieldQsarResult) -> dict[str, bytes]:
    """No executable pickle: raw inputs, geometry, fields and provenance suffice for audit."""
    files: dict[str, bytes] = {}
    tables = {
        "predictions": result.predictions,
        "statistics": result.statistics,
        "validation_checks": result.checks,
        "permutations": result.permutations,
        "field_contributions": result.contours.summary,
        "alignment": result.prepared.alignment.report,
        "measurements": result.prepared.dataset.measurements,
        "input_accounting": result.prepared.dataset.accounting,
        "charges": charge_audit(result),
        "component_selection": result.regression.fitted_model.selection_table_,
    }
    if "source_graph_audit" in result.provenance:
        tables["source_graph_audit"] = pd.DataFrame(result.provenance["source_graph_audit"])
    for name, table in tables.items():
        files[f"{name}.csv"] = table.to_csv(index=False).encode()
    files["metric_definitions.csv"] = (
        pd.DataFrame(METRIC_DEFINITIONS.items(), columns=["metric", "definition"])
        .to_csv(index=False)
        .encode()
    )
    files["aligned.sdf"] = aligned_sdf(result.prepared).encode()
    files["reference.sdf"] = aligned_sdf(result.prepared, reference_only=True).encode()
    files["contours.pml"] = pymol_script(result.contours).encode()
    for name, values in result.contours.maps.items():
        files[f"{name}.dx"] = open_dx(result.fields.grid, values).encode()
    matrix = io.BytesIO()
    np.savez_compressed(
        matrix,
        fields=result.fields.matrix,
        labels=np.array(result.fields.labels),
        retained_mask=result.regression.fitted_filter.support_,
        coefficients=np.asarray(result.regression.fitted_model.coef_),
        training_sd=result.regression.fitted_filter.training_sd_,
        column_scale=result.regression.fitted_filter.column_scale_,
        training_column_means=result.regression.fitted_filter.column_mean_,
    )
    files["fields.npz"] = matrix.getvalue()
    review = interpretation_review(result)
    files["interpretation.json"] = json.dumps(review, indent=2, allow_nan=False).encode()
    record: dict[str, Any] = {
        **result.provenance,
        "status": result.status,
        "interpretation_review": review,
    }
    files["provenance.json"] = json.dumps(record, indent=2, allow_nan=False).encode()
    if result.training_stability is not None:
        for name in ("summary", "predictions", "contributions"):
            files[f"training_stability/{name}.csv"] = (
                result.training_stability[name].to_csv(index=False).encode()
            )
        files["training_stability/provenance.json"] = json.dumps(
            {
                **result.training_stability["provenance"],
                "fitted_model_signature": result.training_stability.get("fitted_model_signature"),
            },
            indent=2,
            allow_nan=False,
        ).encode()
    images = _plots(result)
    sections = [
        "<!doctype html><html><meta charset='utf-8'><title>SARscope field QSAR</title>",
        "<style>body{max-width:1100px;margin:30px auto;font:16px/1.5 system-ui;padding:20px}"
        "table{border-collapse:collapse;display:block;overflow:auto;font-size:12px}"
        "td,th{padding:6px;border:1px solid #ddd}img{max-width:100%}"
        ".warning{border-left:4px solid #c76b22;padding:12px;background:#fff2df}</style>",
        "<h1>SARscope · uncertainty-aware, experimental 3D SAR</h1>",
        f"<p class='warning'>{html.escape(result.status)}</p>",
        f"<p class='warning'>{html.escape(review['interpretation_message'])}</p>",
        "<p>Use a closely related measured series with reviewed alignment. These are "
        "interactive hypotheses, not automatic lead optimisation. Small-pose stress tests "
        "measure training OOF prediction sensitivity, not stable contour locations "
        "or new-compound confidence intervals.</p>",
        "<p>Model associations, not binding energies, observed contacts or proof of mechanism. "
        "No published q² reproduction or superiority over existing software is claimed.</p>",
        f"<p>{html.escape(result.prepared.alignment.assumption)}</p>",
        f"<p>{html.escape(result.fields.convention)}</p>",
        "<p><b>CV metrics:</b> pooled nested-CV q² uses one training-wide SST; "
        "cv_fold_mean_r2 (legacy cv_r2) averages fold-specific predictive R², "
        "each with its own SST. Narrow/small folds can produce very negative fold R². "
        "These are different statistics, not conflicting estimates of the same quantity. "
        "See metric_definitions.csv. Pearson r² is a separate association diagnostic.</p>",
    ]
    field_warning = unmodelled_field_warning(result.contours.summary)
    if field_warning:
        sections.append(f"<p class='warning'>{html.escape(field_warning)}</p>")
    if review["dominant_field"]:
        dominant = review["dominant_field"]
        sections.append(
            f"<p class='warning'>Field imbalance: {html.escape(dominant['field'])} accounts "
            f"for {dominant['percent']:.1f}% of absolute contour contributions. "
            "Inspect field scaling; other properties are not thereby biologically irrelevant. "
            "The 95% warning is an interpretation flag, not a validity criterion.</p>"
        )
    sections.append(
        "<h2>Same-split 2D baseline comparison</h2>"
        "<p>Compare 3D field PLS, 2D ECFP4 RF and training-mean errors in Statistics below. "
        "Point estimates on one split are not a test of superiority. "
        "Do not select a replacement preset from held-out results.</p>"
    )
    for name in (
        "validation_checks",
        "statistics",
        "field_contributions",
        "alignment",
        "component_selection",
    ):
        sections.extend(
            [
                f"<h2>{html.escape(name.replace('_', ' ').title())}</h2>",
                tables[name].to_html(index=False, escape=True),
            ]
        )
    sections.append(
        "<p>Field percentages are shares of absolute coefficient × training-SD "
        "contributions, not independent explained variance or binding energy.</p>"
    )
    if result.training_stability is not None:
        sections.extend(
            [
                "<h2>Training-only stability diagnostics</h2>",
                "<p>Held-out labels/rows were excluded. Flags concern sensitivity to numerical "
                "sampling, charges or small relative pose changes; they do not certify "
                "a binding mode, validate generated conformers "
                "or establish model predictivity.</p>",
                result.training_stability["summary"].to_html(index=False, escape=True),
            ]
        )
    for name, content in images.items():
        files[f"figures/{name}.png"] = content
        sections.append(
            f"<h2>{name.replace('_', ' ').title()}</h2><img alt='{name}' "
            f"src='data:image/png;base64,{base64.b64encode(content).decode()}'>"
        )
    sections.append(
        "<p>Random series splits measure interpolation. The PCA box is only a "
        "coarse applicability diagnostic. Reusing test scores for tuning invalidates "
        "the hold-out. Selection-CV q² is not the nested evaluation q².</p></html>"
    )
    files["report.html"] = "\n".join(sections).encode()
    manifest = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
    files["manifest.sha256.json"] = json.dumps(manifest, indent=2).encode()
    return files


def field_qsar_archive(result: FieldQsarResult) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in field_qsar_files(result).items():
            archive.writestr(name, content)
    return buffer.getvalue()

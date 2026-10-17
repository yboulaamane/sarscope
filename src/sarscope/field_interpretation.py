"""Shared, fail-closed interpretation of field-QSAR evidence, never certification.

Use the existing fitted model and training-only sensitivity results. No model
is fitted here, and no held-out result chooses a replacement alignment/preset.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sarscope.field_qsar import FieldQsarResult

POSE_VARIANTS = ("pose_1101", "pose_1102", "pose_1103")
POSE_STATUS_LABELS = {
    "not_assessed": "Not checked",
    "incomplete": "Incomplete · rerun needed",
    "sensitive": "Alignment-sensitive",
    "within_tested_limits": "Within tested limits · not validated",
}


def _finite(value: Any) -> float | None:
    try:
        return float(value) if np.isfinite(float(value)) else None
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    # NaN and the string "False" must never become a passing Boolean.
    return isinstance(value, (bool, np.bool_)) and bool(value)


def _flags(values: pd.Series) -> pd.Series:
    # Empty Arrow-string Series preserve their dtype through map; force Boolean
    # output so malformed/empty evidence cannot fail during logical operations.
    return pd.Series([_flag(value) for value in values], index=values.index, dtype=bool)


def interpretation_review(result: FieldQsarResult) -> dict[str, Any]:
    """Separate absent/incomplete/failed evidence from passing bounded diagnostics."""
    status, complete, maximum, failed = "not_assessed", False, None, 0
    other_failures = 0
    message = (
        "Alignment sensitivity has not been assessed. Contours and candidate predictions "
        "are exploratory hypotheses; a good test score does not establish pose robustness."
    )
    evidence = result.training_stability
    if evidence is not None:
        summary = evidence.get("summary", pd.DataFrame())
        required = {
            "variant",
            "kind",
            "completed",
            "engineering_stability_flag",
            "oof_change_p95",
            "oof_change_max",
        }
        status = "incomplete"
        message = (
            "Alignment sensitivity is incomplete or not bound to this fitted model. "
            "Rerun the training-only check; missing evidence is not a stability pass."
        )
        if required <= set(summary):
            others = summary[summary.kind.isin(["charge", "grid_phase"])]
            other_failures = int(
                (~_flags(others.completed) | ~_flags(others.engineering_stability_flag)).sum()
            )
            poses = summary[summary.kind.eq("relative_pose")].copy()
            finished = _flags(poses.completed)
            finite = poses[["oof_change_p95", "oof_change_max"]].apply(
                pd.to_numeric, errors="coerce"
            )
            valid = finished & np.isfinite(finite).all(axis=1) & finite.ge(0).all(axis=1)
            # Recheck the actual fixed engineering limits, not only a stored checkmark.
            passes = (
                _flags(poses.engineering_stability_flag)
                & valid
                & finite.oof_change_p95.between(0, 0.5)
                & finite.oof_change_max.between(0, 1.0)
            )
            failed = int((valid & ~passes).sum())
            maximum = _finite(finite.loc[valid, "oof_change_max"].max())
            provenance = evidence.get("provenance", {})
            bound = False
            if evidence.get("fitted_model_signature"):
                # Reuse one canonical model identity; do not duplicate its hashing rules.
                from sarscope.field_design import field_model_signature

                try:
                    bound = evidence["fitted_model_signature"] == field_model_signature(result)
                except (ValueError, AttributeError):
                    bound = False
            controls = summary[summary.variant.isin(["baseline", "common_translation"])]
            control_values = controls[["oof_change_p95", "oof_change_max"]].apply(
                pd.to_numeric, errors="coerce"
            )
            control_ok = (
                set(controls.variant) == {"baseline", "common_translation"}
                and len(controls) == 2
                and _flags(controls.completed).all()
                and _flags(controls.engineering_stability_flag).all()
                and np.isfinite(control_values).all(axis=None)
                and control_values.ge(0).all(axis=None)
                and control_values.le(1e-8).all(axis=None)
            )
            complete = bool(
                bound
                and len(poses) == len(POSE_VARIANTS)
                and set(poses.variant) == set(POSE_VARIANTS)
                and valid.all()
                and control_ok
                and provenance.get("held_out_rows_used") == 0
                and provenance.get("held_out_labels_used") is False
            )
            if (
                failed
                and bound
                and control_ok
                and (
                    provenance.get("held_out_rows_used") == 0
                    and provenance.get("held_out_labels_used") is False
                )
            ):
                status = "sensitive"
                message = (
                    f"Alignment-sensitive predictions: {failed} completed pose stress check(s) "
                    "exceed the declared limits. Treat contour interpretations and proposed "
                    "potency gains as unstable hypotheses, not optimisation recommendations."
                )
            elif complete and passes.all():
                status = "within_tested_limits"
                message = (
                    "The three small-pose stress checks are within the tested limits. "
                    "This does not validate the binding alignment, contour locations, "
                    "new conformers or prospective predictions."
                )
    if result.provenance.get("training_stability_attempt_error"):
        status, complete = "incomplete", False
        message = (
            "The latest stability attempt did not complete. Earlier evidence is retained, "
            "but must not be presented as a fresh successful check. " + message
        )
    interpretation_message = message + (
        f" {other_failures} charge/grid check(s) also failed or did not complete; "
        "pose checks alone do not remove that uncertainty."
        if other_failures
        else ""
    )
    stats = result.statistics.set_index("algorithm")
    baseline: dict[str, Any] = {}
    if {"field_pls", "random_forest"} <= set(stats.index):
        field = _finite(stats.loc["field_pls", "test_rmse"])
        rf = _finite(stats.loc["random_forest", "test_rmse"])
        baseline = {
            "field_rmse": field,
            "rf_rmse": rf,
            "mean_rmse": _finite(stats.loc["mean_baseline", "test_rmse"])
            if "mean_baseline" in stats.index
            else None,
            "field_minus_rf_rmse": field - rf if field is not None and rf is not None else None,
            "interpretation": "Same-split point estimates only; not a test of superiority.",
        }
    dominant = None
    contributions = result.contours.summary
    if not contributions.empty:
        percentages = pd.to_numeric(contributions.absolute_contribution_percent, errors="coerce")
        if percentages.notna().any() and len(contributions) > 1:
            index = percentages.idxmax()
            if percentages.loc[index] >= 95:
                dominant = {
                    "field": str(contributions.loc[index, "field"]),
                    "percent": float(percentages.loc[index]),
                    "warning_threshold_percent": 95,
                    "interpretation": (
                        "Coefficient × training-SD accounting, not causal binding contributions. "
                        "Inspect field scaling; do not infer the other properties are irrelevant."
                    ),
                }
    return {
        "policy_version": 1,
        "purpose": "Interactive, uncertainty-aware 3D SAR; not automatic lead optimisation",
        "pose_status": status,
        "pose_message": message,
        "interpretation_message": interpretation_message,
        "other_sensitivity_failures": other_failures,
        "complete_pose_protocol": complete,
        "failed_pose_checks": failed,
        "maximum_training_oof_pose_change": maximum,
        "pose_thresholds": {"p95_log_units": 0.5, "maximum_log_units": 1.0},
        "limits": (
            "Training-only refitted OOF prediction sensitivity; not a new-compound uncertainty "
            "interval, contour-location stability test or universal validity criterion."
        ),
        "requires_exploratory_consent": bool(
            not result.checks.passed.all() or status != "within_tested_limits" or other_failures > 0
        ),
        "baseline": baseline,
        "dominant_field": dominant,
    }


def interpretation_signature(result: FieldQsarResult) -> str:
    """Invalidate cached candidate labels/downloads when diagnostic evidence changes."""
    digest = hashlib.sha256(json.dumps(interpretation_review(result), sort_keys=True).encode())
    if result.training_stability is not None:
        for name in ("summary", "predictions", "contributions"):
            table = result.training_stability.get(name)
            if isinstance(table, pd.DataFrame):
                digest.update(table.to_csv(index=False).encode())
        digest.update(
            json.dumps(
                result.training_stability.get("provenance", {}), sort_keys=True, default=str
            ).encode()
        )
    return digest.hexdigest()


def training_pose_ranges(result: FieldQsarResult) -> pd.DataFrame:
    """Describe training OOF variation only; never invent candidate prediction intervals."""
    columns = ["molecule_id", "minimum_pose_oof", "maximum_pose_oof", "pose_oof_range"]
    if result.training_stability is None:
        return pd.DataFrame(columns=columns)
    table = result.training_stability.get("predictions", pd.DataFrame())
    if not {"variant", "molecule_id", "oof_prediction"} <= set(table):
        return pd.DataFrame(columns=columns)
    ids = set(result.prepared.dataset.table.iloc[result.regression.train_index].molecule_id)
    selected = table[table.variant.isin(POSE_VARIANTS) & table.molecule_id.isin(ids)].copy()
    selected["oof_prediction"] = pd.to_numeric(selected.oof_prediction, errors="coerce")
    selected = selected[np.isfinite(selected.oof_prediction)]
    selected = selected.drop_duplicates(["variant", "molecule_id"], keep=False)
    ranges = (
        selected.groupby("molecule_id")
        .oof_prediction.agg(
            minimum_pose_oof="min", maximum_pose_oof="max", completed_pose_checks="count"
        )
        .reset_index()
    )
    ranges["pose_oof_range"] = ranges.maximum_pose_oof - ranges.minimum_pose_oof
    return ranges.sort_values("pose_oof_range", ascending=False)

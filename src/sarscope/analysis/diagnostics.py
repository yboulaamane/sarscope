"""Held-out regression diagnostics; none of these choose or tune a model."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from sarscope.analysis.regression import RegressionResult, regression_metrics


def regression_summary(result: RegressionResult) -> pd.DataFrame:
    truth, prediction = result.test_truth, result.test_predictions
    errors = np.abs(truth - prediction)
    r2, rmse, rho = regression_metrics(truth, prediction)
    baseline = getattr(result, "baseline_test_predictions", None)
    baseline_rmse = (
        float(np.sqrt(np.mean((truth - baseline) ** 2))) if baseline is not None else np.nan
    )
    rows = [
        ("Test R²", r2, "1 is perfect; negative means worse than the test-mean reference."),
        ("Test RMSE", rmse, "Log units; emphasizes large prediction errors."),
        ("Test MAE", float(errors.mean()), "Mean absolute potency error in log units."),
        ("Test Spearman", rho, "Rank correlation; does not measure calibration."),
        ("Median absolute error", float(np.median(errors)), "Typical error in log units."),
        (
            "Typical fold error",
            float(10 ** np.median(errors)),
            "10 raised to median absolute error.",
        ),
        (
            "Within 3-fold",
            float(np.mean(errors <= np.log10(3))),
            "Fraction within ±0.477 log units.",
        ),
        ("Within 10-fold", float(np.mean(errors <= 1)), "Fraction within ±1 log unit."),
        (
            "Mean residual",
            float(np.mean(truth - prediction)),
            "Measured minus predicted; positive = underprediction.",
        ),
        (
            "Training-mean baseline RMSE",
            baseline_rmse,
            "Constant prediction fitted on training labels only.",
        ),
        (
            "RMSE gain versus baseline",
            baseline_rmse - rmse,
            "Positive = lower test error than the baseline.",
        ),
    ]
    return pd.DataFrame(rows, columns=["metric", "value", "interpretation"])


def bootstrap_regression(
    result: RegressionResult,
    groups: Sequence[Any],
    *,
    repeats: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Approximate 95% intervals using paired held-out scaffold-block resampling.

    Blank/acyclic scaffolds are individual blocks. No fitting, test-driven
    selection, or hyperparameter changes occur. These intervals are conditional
    on the fitted model and test population, not a guarantee under temporal or
    target shift. Very few independent blocks make the estimate unstable.
    """
    truth = np.asarray(result.test_truth, dtype=float)
    prediction = np.asarray(result.test_predictions, dtype=float)
    if len(groups) != len(truth) or len(truth) < 3 or repeats < 20:
        raise ValueError(
            "Bootstrap requires aligned test scaffolds, >=3 test rows, and >=20 repeats."
        )
    blocks: dict[str, list[int]] = {}
    for i, group in enumerate(groups):
        key = str(group) if pd.notna(group) and str(group).strip() else f"__row_{i}"
        blocks.setdefault(key, []).append(i)
    if len(blocks) < 2:
        raise ValueError("Bootstrap needs at least two independent held-out scaffold blocks.")
    members = [np.asarray(rows, dtype=int) for rows in blocks.values()]
    rng = np.random.default_rng(seed)
    baseline = getattr(result, "baseline_test_predictions", None)

    def values(rows: np.ndarray) -> list[float]:
        measured, estimated = truth[rows], prediction[rows]
        error = measured - estimated
        rmse = float(np.sqrt(np.mean(error**2)))
        variance = float(np.sum((measured - measured.mean()) ** 2))
        r2 = 1 - float(np.sum(error**2)) / variance if variance > 0 else np.nan
        gain = (
            float(np.sqrt(np.mean((measured - baseline[rows]) ** 2))) - rmse
            if baseline is not None
            else np.nan
        )
        return [r2, rmse, float(np.abs(error).mean()), gain]

    samples = np.asarray(
        [
            values(
                np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
            )
            for _ in range(repeats)
        ]
    )
    estimates = values(np.arange(len(truth)))
    rows = []
    for column, metric in enumerate(["R²", "RMSE", "MAE", "RMSE gain versus baseline"]):
        finite = samples[np.isfinite(samples[:, column]), column]
        low, high = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
        rows.append(
            {
                "metric": metric,
                "estimate": estimates[column],
                "lower_95": low,
                "upper_95": high,
                "method": "paired scaffold-block percentile bootstrap",
                "repeats": repeats,
                "valid_repeats": len(finite),
                "scaffold_blocks": len(blocks),
                "seed": seed,
            }
        )
    return pd.DataFrame(rows)


def novelty_summary(profile: pd.DataFrame) -> pd.DataFrame:
    """Display error by predeclared structural-similarity bins, never optimize the bins."""
    table = profile.copy()
    table["similarity_bin"] = pd.cut(
        table["max_training_similarity"],
        [-0.001, 0.3, 0.5, 0.7, 1.0],
        labels=["≤0.3", "0.3–0.5", "0.5–0.7", ">0.7"],
    )
    rows = []
    for name, part in table.groupby("similarity_bin", observed=True):
        errors = part["absolute_error"].to_numpy(dtype=float)
        rows.append(
            {
                "nearest_training_similarity": str(name),
                "n": len(part),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(errors.mean()),
                "within_3_fold": float(np.mean(errors <= np.log10(3))),
                "within_10_fold": float(np.mean(errors <= 1)),
            }
        )
    return pd.DataFrame(rows)

"""Leak-free continuous QSAR models for pActivity.

Classification is useful for triage, but potency is continuous.  This module
keeps the same outer split and fold-local feature filtering used by the
classifier while fitting regressors directly to pActivity.  Model selection is
based on cross-validated RMSE; the held-out test set is touched only once.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR

from sarscope.analysis.features import VarianceCorrelationFilter, feature_filter
from sarscope.analysis.model import _cv_folds, _outer_split
from sarscope.params import ModelParams

REGRESSION_ALGORITHMS: dict[str, Callable[[int], Any]] = {
    "mean_baseline": lambda seed: DummyRegressor(strategy="mean"),
    "ridge": lambda seed: Ridge(alpha=10.0),
    "nearest_neighbors": lambda seed: KNeighborsRegressor(n_neighbors=4, weights="distance"),
    "svr": lambda seed: SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1),
    "gradient_boosting": lambda seed: GradientBoostingRegressor(
        n_estimators=200, random_state=seed
    ),
    "extra_trees": lambda seed: ExtraTreesRegressor(
        n_estimators=200, min_samples_split=7, random_state=seed, n_jobs=-1
    ),
    "random_forest": lambda seed: RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1
    ),
    "neural_net": lambda seed: MLPRegressor(
        hidden_layer_sizes=(64, 32),
        alpha=0.01,
        early_stopping=False,
        max_iter=500,
        random_state=seed,
    ),
}

REGRESSION_SCORE_COLUMNS: tuple[str, ...] = (
    "algorithm",
    "train_r2",
    "cv_r2",
    "cv_r2_sd",
    "test_r2",
    "train_rmse",
    "cv_rmse",
    "cv_rmse_sd",
    "test_rmse",
    "train_spearman",
    "cv_spearman",
    "cv_spearman_sd",
    "test_spearman",
    "train_mae",
    "cv_mae",
    "cv_mae_sd",
    "test_mae",
    "n_features",
)


@dataclass
class RegressionResult:
    scores: pd.DataFrame
    best_algorithm: str
    test_index: NDArray[np.intp]
    train_index: NDArray[np.intp]
    test_predictions: NDArray[np.float64]
    test_truth: NDArray[np.float64]
    fitted_filter: VarianceCorrelationFilter
    fitted_model: Any
    #: Training-CV residual band; it has no formal coverage guarantee.
    empirical_half_width: float = math.nan
    empirical_test_coverage: float = math.nan
    interval_level: float = 0.9
    baseline_test_predictions: NDArray[np.float64] | None = None


def _spearman(y_true: NDArray[Any], y_pred: NDArray[Any]) -> float:
    if len(y_true) < 2 or np.unique(y_true).size < 2 or np.unique(y_pred).size < 2:
        return float("nan")
    return float(spearmanr(y_true, y_pred).statistic)


def regression_metrics(y_true: NDArray[Any], y_pred: NDArray[Any]) -> tuple[float, float, float]:
    """Return R2, RMSE and Spearman rho, in that order."""
    return (
        float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
        float(mean_squared_error(y_true, y_pred) ** 0.5),
        _spearman(np.asarray(y_true), np.asarray(y_pred)),
    )


def _fit(
    name: str, X: NDArray[Any], y: NDArray[np.float64], params: ModelParams
) -> tuple[VarianceCorrelationFilter, Any]:
    filt = feature_filter(params.features).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = REGRESSION_ALGORITHMS[name](params.seed).fit(filt.transform(X), y)
    return filt, fitted


def _predict(model: Any, X: NDArray[Any]) -> NDArray[np.float64]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(X), dtype=float)


def _row(
    name: str,
    train: tuple[float, float, float],
    cv: list[tuple[float, float, float]],
    test: tuple[float, float, float],
    n_features: int,
    mae: tuple[float, list[float], float],
) -> dict[str, Any]:
    values = np.asarray(cv, dtype=float)

    def mean(column: int) -> float:
        finite = values[np.isfinite(values[:, column]), column]
        return float(finite.mean()) if len(finite) else math.nan

    def sd(column: int) -> float:
        finite = values[np.isfinite(values[:, column]), column]
        return float(finite.std(ddof=1)) if len(finite) > 1 else math.nan

    return {
        "algorithm": name,
        "train_r2": train[0],
        "cv_r2": mean(0),
        "cv_r2_sd": sd(0),
        "test_r2": test[0],
        "train_rmse": train[1],
        "cv_rmse": mean(1),
        "cv_rmse_sd": sd(1),
        "test_rmse": test[1],
        "train_spearman": train[2],
        "cv_spearman": mean(2),
        "cv_spearman_sd": sd(2),
        "test_spearman": test[2],
        "train_mae": mae[0],
        "cv_mae": float(np.mean(mae[1])) if mae[1] else math.nan,
        "cv_mae_sd": float(np.std(mae[1], ddof=1)) if len(mae[1]) > 1 else math.nan,
        "test_mae": mae[2],
        "n_features": n_features,
    }


def evaluate_regression(
    X: NDArray[Any],
    y: Sequence[float],
    groups: Sequence[str | None],
    stratify_labels: Sequence[str],
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
    source_test: Sequence[bool] | None = None,
) -> RegressionResult:
    """Evaluate continuous models with fold-local feature selection.

    The activity classes stratify random/scaffold validation only; time
    validation keeps whole document years in chronological order. Estimators
    always receive the original continuous pActivity values.
    """
    names = (
        list(REGRESSION_ALGORITHMS)
        if tuple(params.regression_algorithms) == ("all",)
        else list(params.regression_algorithms)
    )
    unknown = [name for name in names if name not in REGRESSION_ALGORITHMS]
    if unknown:
        raise ValueError(
            f"unknown regression algorithm(s) {unknown}; known: {sorted(REGRESSION_ALGORITHMS)}"
        )
    if not names:
        raise ValueError("Choose at least one regression algorithm.")
    requested = names.copy()
    if "mean_baseline" not in names:
        names = [*names, "mean_baseline"]

    X = np.asarray(X)
    target = np.asarray(y, dtype=float)
    labels = np.asarray(stratify_labels, dtype=object).astype(str)
    group_ids = np.asarray(
        [g if g is not None else f"__acyclic_{i}" for i, g in enumerate(groups)], dtype=object
    )
    train_idx, test_idx = _outer_split(X, labels, group_ids, params, years, source_test)
    X_train, y_train, groups_train = X[train_idx], target[train_idx], group_ids[train_idx]
    train_years = [years[i] for i in train_idx] if years is not None else None

    rows: list[dict[str, Any]] = []
    cv_errors: dict[str, list[float]] = {}
    for name in names:
        cv: list[tuple[float, float, float]] = []
        cv_mae: list[float] = []
        residuals: list[float] = []
        for fit_rows, eval_rows in _cv_folds(labels[train_idx], groups_train, params, train_years):
            filt, fitted = _fit(name, X_train[fit_rows], y_train[fit_rows], params)
            prediction = _predict(fitted, filt.transform(X_train[eval_rows]))
            cv.append(regression_metrics(y_train[eval_rows], prediction))
            cv_mae.append(float(mean_absolute_error(y_train[eval_rows], prediction)))
            residuals.extend(np.abs(y_train[eval_rows] - prediction).tolist())
        cv_errors[name] = residuals
        filt, fitted = _fit(name, X_train, y_train, params)
        train_prediction = _predict(fitted, filt.transform(X_train))
        test_prediction = _predict(fitted, filt.transform(X[test_idx]))
        rows.append(
            _row(
                name,
                regression_metrics(y_train, train_prediction),
                cv,
                regression_metrics(target[test_idx], test_prediction),
                filt.n_after_correlation_,
                (
                    float(mean_absolute_error(y_train, train_prediction)),
                    cv_mae,
                    float(mean_absolute_error(target[test_idx], test_prediction)),
                ),
            )
        )

    scores = pd.DataFrame(rows, columns=list(REGRESSION_SCORE_COLUMNS))
    ranked = scores[scores["algorithm"].isin(requested)].assign(
        _key=scores.loc[scores["algorithm"].isin(requested), "cv_rmse"].fillna(math.inf)
    )
    best = str(ranked.sort_values(["_key", "algorithm"]).iloc[0]["algorithm"])
    fitted_filter, fitted_model = _fit(best, X_train, y_train, params)
    test_predictions = _predict(fitted_model, fitted_filter.transform(X[test_idx]))
    held_out_errors = np.asarray(cv_errors[best], dtype=float)
    half_width = (
        float(np.quantile(held_out_errors, 0.9, method="higher"))
        if len(held_out_errors)
        else math.nan
    )
    coverage = (
        float(np.mean(np.abs(target[test_idx] - test_predictions) <= half_width))
        if math.isfinite(half_width)
        else math.nan
    )
    return RegressionResult(
        scores=scores,
        best_algorithm=best,
        test_index=test_idx,
        train_index=train_idx,
        test_predictions=test_predictions,
        test_truth=target[test_idx],
        fitted_filter=fitted_filter,
        fitted_model=fitted_model,
        empirical_half_width=half_width,
        empirical_test_coverage=coverage,
        baseline_test_predictions=np.full(len(test_idx), float(y_train.mean())),
    )


def fit_deployment_model(
    X: NDArray[Any], y: Sequence[float], algorithm: str, params: ModelParams
) -> tuple[VarianceCorrelationFilter, Any]:
    """Refit the selected algorithm on all curated rows for later prediction."""
    if algorithm not in REGRESSION_ALGORITHMS:
        raise ValueError(f"unknown regression algorithm {algorithm!r}")
    return _fit(algorithm, np.asarray(X), np.asarray(y, dtype=float), params)

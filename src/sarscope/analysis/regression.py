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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sarscope.analysis.features import VarianceCorrelationFilter
from sarscope.analysis.model import _cv_folds, _outer_split
from sarscope.params import ModelParams

REGRESSION_ALGORITHMS: dict[str, Callable[[int], Any]] = {
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
    feats = params.features
    filt = VarianceCorrelationFilter(feats.variance_threshold, feats.correlation_threshold).fit(X)
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
        "n_features": n_features,
    }


def evaluate_regression(
    X: NDArray[Any],
    y: Sequence[float],
    groups: Sequence[str | None],
    stratify_labels: Sequence[str],
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> RegressionResult:
    """Evaluate continuous models with fold-local feature selection.

    The activity classes are used only to stratify the outer and CV splits;
    the estimators receive the original continuous pActivity values.
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

    X = np.asarray(X)
    target = np.asarray(y, dtype=float)
    labels = np.asarray(stratify_labels, dtype=object).astype(str)
    group_ids = np.asarray(
        [g if g is not None else f"__acyclic_{i}" for i, g in enumerate(groups)], dtype=object
    )
    train_idx, test_idx = _outer_split(X, labels, group_ids, params, years)
    X_train, y_train, groups_train = X[train_idx], target[train_idx], group_ids[train_idx]

    rows: list[dict[str, Any]] = []
    for name in names:
        cv: list[tuple[float, float, float]] = []
        for fit_rows, eval_rows in _cv_folds(labels[train_idx], groups_train, params):
            filt, fitted = _fit(name, X_train[fit_rows], y_train[fit_rows], params)
            prediction = _predict(fitted, filt.transform(X_train[eval_rows]))
            cv.append(regression_metrics(y_train[eval_rows], prediction))
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
            )
        )

    scores = pd.DataFrame(rows, columns=list(REGRESSION_SCORE_COLUMNS))
    ranked = scores.assign(_key=scores["cv_rmse"].fillna(math.inf))
    best = str(ranked.sort_values(["_key", "algorithm"]).iloc[0]["algorithm"])
    fitted_filter, fitted_model = _fit(best, X_train, y_train, params)
    test_predictions = _predict(fitted_model, fitted_filter.transform(X[test_idx]))
    return RegressionResult(
        scores=scores,
        best_algorithm=best,
        test_index=test_idx,
        train_index=train_idx,
        test_predictions=test_predictions,
        test_truth=target[test_idx],
        fitted_filter=fitted_filter,
        fitted_model=fitted_model,
    )


def fit_deployment_model(
    X: NDArray[Any], y: Sequence[float], algorithm: str, params: ModelParams
) -> tuple[VarianceCorrelationFilter, Any]:
    """Refit the selected algorithm on all curated rows for later prediction."""
    if algorithm not in REGRESSION_ALGORITHMS:
        raise ValueError(f"unknown regression algorithm {algorithm!r}")
    return _fit(algorithm, np.asarray(X), np.asarray(y, dtype=float), params)

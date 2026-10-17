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
from sklearn.cross_decomposition import PLSRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
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
    #: Honest outer-training CV predictions, including component selection inside each fit.
    cv_predictions: NDArray[np.float64] | None = None


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
    name: str,
    X: NDArray[Any],
    y: NDArray[np.float64],
    params: ModelParams,
    groups: NDArray[Any] | None = None,
    labels: NDArray[Any] | None = None,
    years: Sequence[int | float | None] | None = None,
) -> tuple[Any, Any]:
    if name == "field_pls":
        return _fit_field_pls(X, y, params, groups, labels, years)
    if params.field_qsar is not None:
        from sarscope.analysis.field_filter import FieldColumnFilter

        filt = FieldColumnFilter(params.field_qsar).fit(X)
    else:
        filt = feature_filter(params.features).fit(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = REGRESSION_ALGORITHMS[name](params.seed).fit(filt.transform(X), y)
    return filt, fitted


def _fit_field_pls(
    X: NDArray[Any],
    y: NDArray[np.float64],
    params: ModelParams,
    groups: NDArray[Any] | None,
    labels: NDArray[Any] | None,
    years: Sequence[int | float | None] | None,
) -> tuple[Any, Any]:
    """Select capped PLS components inside fitting rows, not on the test set.

    Every inner split refits the raw-unit variance filter. Mean centering is
    performed by PLS; autoscaling is deliberately off to preserve field
    magnitudes. q2 is pooled PRESS/SST, not the mean of fold R2 values.
    The n/5 cap is a conservative configurable heuristic, not a validity proof.
    """
    from sarscope.analysis.field_filter import FieldColumnFilter

    settings = params.field_qsar
    if settings is None:
        raise ValueError(
            "field_pls requires explicit FieldQsarParams and aligned molecular fields."
        )
    ratio = settings.component_sample_ratio

    def row_cap(n_rows: int) -> int:
        return n_rows - 1 if ratio is None else n_rows // ratio

    def estimator(components: int) -> Any:
        return PLSRegression(
            n_components=components,
            scale=False,
            max_iter=settings.pls_max_iter,
            tol=settings.pls_tolerance,
        )

    if settings.fixed_components is not None:
        filt = FieldColumnFilter(settings).fit(X)
        transformed = filt.transform(X)
        cap = min(
            settings.component_cap,
            row_cap(len(y)),
            transformed.shape[1],
            np.linalg.matrix_rank(transformed - transformed.mean(0)),
        )
        if settings.fixed_components > cap:
            raise ValueError("Prespecified PLS count exceeds fitting-fold n/5 or feature-rank cap.")
        model = estimator(settings.fixed_components).fit(transformed, y)
        model.selection_table_ = pd.DataFrame(
            [
                {
                    "components": settings.fixed_components,
                    "selection_q2": np.nan,
                    "selection": "prespecified before evaluation; not tuned on this split",
                }
            ]
        )
        model.selected_components_ = settings.fixed_components
        model.effective_component_cap_ = int(cap)
        model.selection_stop_reason_ = "Component count was fixed before evaluation; no search."
        return filt, model
    labels = np.full(len(y), "series", dtype=object) if labels is None else labels
    folds = list(_regression_folds(labels, groups, params, years))
    filtered = []
    cap = min(settings.component_cap, row_cap(len(y)))
    for train, validation in folds:
        filt = FieldColumnFilter(settings).fit(X[train])
        a, b = filt.transform(X[train]), filt.transform(X[validation])
        filtered.append((train, validation, a, b))
        cap = min(
            cap, row_cap(len(train)), a.shape[1], max(0, np.linalg.matrix_rank(a - a.mean(0)))
        )
    if cap < 1:
        raise ValueError("Too few independent training rows/features for capped, nested PLS.")
    rows, previous, chosen = [], -math.inf, 1
    row_limit = f"n/{ratio}" if ratio is not None else "n−1"
    stop_reason = (
        f"Effective search cap: {cap} (requested {settings.component_cap}); "
        f"limited by smallest fitting-fold {row_limit}, feature count/rank."
    )
    for components in range(1, cap + 1):
        prediction = np.full(len(y), np.nan)
        for train, validation, a, b in filtered:
            model = estimator(components).fit(a, y[train])
            prediction[validation] = np.asarray(model.predict(b)).reshape(-1)
        valid = np.isfinite(prediction)
        denominator = float(np.sum((y[valid] - y.mean()) ** 2))
        q2 = (
            1 - float(np.sum((y[valid] - prediction[valid]) ** 2)) / denominator
            if denominator
            else -math.inf
        )
        rows.append(
            {
                "components": components,
                "selection_q2": q2,
                "n_features_min_inner_fold": min(a.shape[1] for _, _, a, _ in filtered),
            }
        )
        if settings.component_selection == "max_q2":
            if q2 > previous:
                chosen, previous = components, q2
            continue
        if components > 1 and q2 - previous <= settings.component_improvement * max(
            abs(previous), 1e-12
        ):
            stop_reason += (
                f" Early stopping at {components} tested components: CV q² improvement "
                f"did not exceed {settings.component_improvement:.0%} of |previous q²|."
            )
            break
        chosen, previous = components, q2
    filt = FieldColumnFilter(settings).fit(X)
    model = estimator(chosen).fit(filt.transform(X), y)
    model.selection_table_ = pd.DataFrame(rows)
    model.selection_table_["selected"] = model.selection_table_.components.eq(chosen)
    model.selected_components_ = chosen
    model.effective_component_cap_ = int(cap)
    model.selection_stop_reason_ = stop_reason
    return filt, model


def regression_cv_predictions(
    name: str,
    X: NDArray[Any],
    y: NDArray[np.float64],
    labels: NDArray[Any],
    groups: NDArray[Any] | None,
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> tuple[NDArray[np.float64], list[tuple[float, float, float]], list[float]]:
    """Shared outer CV engine, also used for full-pipeline field permutations."""
    prediction = np.full(len(y), np.nan)
    metrics, errors = [], []
    for train, validation in _regression_folds(labels, groups, params, years):
        filt, fitted = _fit(
            name,
            X[train],
            y[train],
            params,
            groups[train] if groups is not None else None,
            labels[train],
            [years[i] for i in train] if years is not None else None,
        )
        values = _predict(fitted, filt.transform(X[validation])).reshape(-1)
        prediction[validation] = values
        metrics.append(regression_metrics(y[validation], values))
        errors.append(float(mean_absolute_error(y[validation], values)))
    return prediction, metrics, errors


def _regression_folds(
    labels: NDArray[Any],
    groups: NDArray[Any] | None,
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> Any:
    if params.regression_cv == "loo":
        if params.split != "random":
            raise ValueError(
                "LOO must not bypass scaffold, source or chronological group boundaries."
            )
        return LeaveOneOut().split(labels)
    return _cv_folds(labels, groups, params, years)


def regression_split(
    X: NDArray[Any],
    labels: NDArray[Any],
    groups: NDArray[Any],
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
    source_test: Sequence[bool] | None = None,
    split_indices: tuple[Sequence[int], Sequence[int]] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """Validate a frozen external partition, or use the existing split engine.

    Explicit partitions are for independently specified benchmarks and are
    never inferred from activities. Reject duplicate, overlapping or omitted
    indices instead of silently changing the evaluation cohort.
    """
    if split_indices is None:
        return _outer_split(X, labels, groups, params, years, source_test)
    parts = []
    for values in split_indices:
        raw = np.asarray(values)
        if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer) or not len(raw):
            raise ValueError("Frozen split indices must be nonempty one-dimensional integers.")
        parts.append(np.sort(raw.astype(np.intp)))
    combined = np.concatenate(parts)
    if not np.array_equal(np.sort(combined), np.arange(len(X))):
        raise ValueError("Frozen split must partition every row exactly once without overlap.")
    return parts[0], parts[1]


def _predict(model: Any, X: NDArray[Any]) -> NDArray[np.float64]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Older sklearn PLS versions return (n, 1) for a single target. Keep
        # residual/coverage subtraction row-wise, never an (n, n) broadcast.
        return np.asarray(model.predict(X), dtype=float).reshape(-1)


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
    *,
    split_indices: tuple[Sequence[int], Sequence[int]] | None = None,
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
    if tuple(params.regression_algorithms) == ("all",):
        names = [name for name in names if name != "field_pls"]
    allowed = set(REGRESSION_ALGORITHMS) | ({"field_pls"} if params.field_qsar else set())
    unknown = [name for name in names if name not in allowed]
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
    train_idx, test_idx = regression_split(
        X, labels, group_ids, params, years, source_test, split_indices
    )
    X_train, y_train, groups_train = X[train_idx], target[train_idx], group_ids[train_idx]
    train_years = [years[i] for i in train_idx] if years is not None else None

    rows: list[dict[str, Any]] = []
    cv_errors: dict[str, list[float]] = {}
    cv_predictions: dict[str, NDArray[np.float64]] = {}
    for name in names:
        prediction, cv, cv_mae = regression_cv_predictions(
            name, X_train, y_train, labels[train_idx], groups_train, params, train_years
        )
        valid = np.isfinite(prediction)
        cv_errors[name] = np.abs(y_train[valid] - prediction[valid]).tolist()
        cv_predictions[name] = prediction
        filt, fitted = _fit(
            name, X_train, y_train, params, groups_train, labels[train_idx], train_years
        )
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
    fitted_filter, fitted_model = _fit(
        best, X_train, y_train, params, groups_train, labels[train_idx], train_years
    )
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
        cv_predictions=cv_predictions[best],
    )


def fit_deployment_model(
    X: NDArray[Any], y: Sequence[float], algorithm: str, params: ModelParams
) -> tuple[VarianceCorrelationFilter, Any]:
    """Refit the selected algorithm on all curated rows for later prediction."""
    if algorithm not in REGRESSION_ALGORITHMS:
        raise ValueError(f"unknown regression algorithm {algorithm!r}")
    return _fit(algorithm, np.asarray(X), np.asarray(y, dtype=float), params)

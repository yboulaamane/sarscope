"""Multiclass QSAR bake-off, with the validation done in an order that cannot leak.

The fourteen algorithms and their hyperparameters follow scikit-learn's
"classifier comparison" example, which is the usual starting grid in published
QSAR bake-offs (hence GaussianNB and QDA at their defaults).
``random_state=seed`` wherever the estimator takes one.

These are starting points, not tuned models. A bake-off answers "which family
is worth pursuing on this data", not "how good can this get" - the winner
deserves a proper hyperparameter search afterwards.

**The leak-free protocol** (the default):

    1. Hold out ``test_fraction`` of molecules. "scaffold": StratifiedGroupKFold
       with n_splits = round(1 / test_fraction), groups = Murcko scaffold, first
       fold is the test set. Acyclic molecules are each their own group.
       "random": StratifiedShuffleSplit. "time": first document year at or
       before the cutoff for training, later first-seen compounds for test.
    2. Cross-validate on the training set: StratifiedGroupKFold / StratifiedKFold
       for scaffold / random, or expanding chronological folds for time.
    3. Inside every fit - each CV fold, and the final fit on the whole training
       set - fit VarianceCorrelationFilter on the fitting rows only, then
       randomly oversample those rows only (duplicate minority-class rows up to
       the majority count, seeded). Evaluation rows are never filtered-on,
       never duplicated.

**The naive protocol** (``leakage_audit``) is: select features on all data,
oversample all data, then split and cross-validate. This ordering is extremely
common in published QSAR, and it leaks twice over. Oversampling duplicates
minority-class rows, so after the split copies of one molecule sit on both
sides - the model is graded partly on molecules it trained on. Selecting
features on all data leaks more quietly, because which features survive was
decided using the held-out rows.

Running both protocols on the same data and reporting the gap turns "this
leaks" from an assertion into a number for your dataset. On labels that carry
no signal at all, the naive order scores Extra Trees at MCC 0.99 where the
leak-free order correctly scores 0.00.

**Metrics:** accuracy, balanced accuracy, and multiclass MCC
(``sklearn.metrics.matthews_corrcoef``). MCC is the one to read on imbalanced
classes: accuracy rewards a model that simply predicts the majority class.
Micro-averaged recall is not reported because in single-label multiclass it is
identical to accuracy by definition. "train" is resubstitution on the training
set, reported for comparison only and never used for ranking.

**Choosing the best model uses CV MCC, not test.** Picking the winner by test
score is a second leak; the test set is scored once, for the chosen model and
for comparison.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sarscope.analysis.features import VarianceCorrelationFilter
from sarscope.params import ModelParams

#: name -> factory(seed) -> unfitted estimator. Tree ensembles use every core;
#: results do not depend on it.
ALGORITHMS: dict[str, Callable[[int], Any]] = {
    "nearest_neighbors": lambda seed: KNeighborsClassifier(n_neighbors=4, weights="uniform"),
    "linear_svm": lambda seed: SVC(kernel="linear", C=0.25, random_state=seed),
    "polynomial_svm": lambda seed: SVC(kernel="poly", C=0.025, random_state=seed),
    "rbf_svm": lambda seed: SVC(kernel="rbf", C=2, gamma=2, random_state=seed),
    "gaussian_process": lambda seed: GaussianProcessClassifier(1.0 * RBF(1.0), random_state=seed),
    "gradient_boosting": lambda seed: GradientBoostingClassifier(
        n_estimators=200, random_state=seed
    ),
    "decision_tree": lambda seed: DecisionTreeClassifier(
        max_depth=5, min_samples_split=8, random_state=seed
    ),
    "extra_trees": lambda seed: ExtraTreesClassifier(
        n_estimators=200, min_samples_split=7, random_state=seed, n_jobs=-1
    ),
    "random_forest": lambda seed: RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=seed, n_jobs=-1
    ),
    "neural_net": lambda seed: MLPClassifier(alpha=0.5, max_iter=1500, random_state=seed),
    "adaboost": lambda seed: AdaBoostClassifier(n_estimators=200, random_state=seed),
    "naive_bayes": lambda seed: GaussianNB(),
    "qda": lambda seed: QuadraticDiscriminantAnalysis(),
    "sgd": lambda seed: SGDClassifier(loss="squared_error", penalty="l2", random_state=seed),
}

#: Columns of ModelResult.scores.
SCORE_COLUMNS: tuple[str, ...] = (
    "algorithm",
    "protocol",  # "leak_free" or "naive"
    "train_accuracy",
    "cv_accuracy",
    "cv_accuracy_sd",
    "test_accuracy",
    "train_balanced_accuracy",
    "cv_balanced_accuracy",
    "test_balanced_accuracy",
    "train_mcc",
    "cv_mcc",
    "cv_mcc_sd",
    "test_mcc",
    "cv_roc_auc_ovr",
    "test_roc_auc_ovr",
    "cv_pr_auc_ovr",
    "test_pr_auc_ovr",
    "test_auc_classes",
    "n_features",  # after the filter, on the final training fit
)


@dataclass
class ModelResult:
    scores: pd.DataFrame  # SCORE_COLUMNS, one row per (algorithm, protocol)
    best_algorithm: str  # highest leak-free cv_mcc; ties broken by name
    #: Row positions (into the input X) of the leak-free test set, for
    #: domain.py and for the report.
    test_index: NDArray[np.intp]
    train_index: NDArray[np.intp]
    #: Held-out predictions from the CV-selected model, aligned to test_index.
    test_predictions: NDArray[Any] | None = None
    test_truth: NDArray[Any] | None = None
    test_score_classes: tuple[str, ...] = ()
    test_class_scores: NDArray[np.float64] | None = None
    test_class_metrics: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_confusion: pd.DataFrame = field(default_factory=pd.DataFrame)


def evaluate(
    X: NDArray[Any],
    y: Sequence[str],
    groups: Sequence[str | None],
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> ModelResult:
    """Run every algorithm in ``params.algorithms`` under the leak-free protocol,
    and also under the naive protocol if ``params.leakage_audit``.

    ``groups`` is the Murcko scaffold per row (None for acyclic), used only when
    ``params.split == "scaffold"``. The single name "all" expands to every key
    of ALGORITHMS. Unknown names are a ValueError raised before any fitting.
    """
    names = list(ALGORITHMS) if tuple(params.algorithms) == ("all",) else list(params.algorithms)
    unknown = [n for n in names if n not in ALGORITHMS]
    if unknown:
        raise ValueError(f"unknown algorithm(s) {unknown}; known: {sorted(ALGORITHMS)}")

    X = np.asarray(X)
    labels = np.asarray(y, dtype=object).astype(str)
    group_ids = np.array(
        [g if g is not None else f"__acyclic_{i}" for i, g in enumerate(groups)], dtype=object
    )
    train_idx, test_idx = _outer_split(X, labels, group_ids, params, years)

    rows = []
    for name in names:
        rows.append(_leak_free(name, X, labels, group_ids, train_idx, test_idx, params, years))
        if params.leakage_audit and params.split != "time":
            rows.append(_naive_protocol(name, X, labels, params))
    scores = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))

    ranked = scores[scores["protocol"] == "leak_free"].assign(
        _key=lambda f: f["cv_mcc"].fillna(-math.inf)
    )
    best = ranked.sort_values(["_key", "algorithm"], ascending=[False, True]).iloc[0]["algorithm"]
    best_filter, best_model = _fit_leak_free(str(best), X[train_idx], labels[train_idx], params)
    best_X_test = best_filter.transform(X[test_idx])
    best_predictions = _predict(best_model, best_X_test)
    score_classes, class_scores = _score_matrix(best_model, best_X_test)
    class_metrics = _class_diagnostics(
        labels[test_idx], best_predictions, score_classes, class_scores
    )
    reported_classes = class_metrics["activity_class"].tolist()
    matrix = confusion_matrix(labels[test_idx], best_predictions, labels=reported_classes)
    confusion = pd.DataFrame(matrix, index=reported_classes, columns=reported_classes)
    confusion.index.name = "actual"
    confusion.columns.name = "predicted"
    return ModelResult(
        scores,
        str(best),
        test_idx,
        train_idx,
        best_predictions,
        labels[test_idx],
        score_classes,
        class_scores,
        class_metrics,
        confusion,
    )


# -- protocol pieces ---------------------------------------------------------


def _outer_split(
    X: NDArray[Any],
    y: NDArray[Any],
    groups: NDArray[Any],
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    if params.split == "time":
        if years is None:
            raise ValueError("time split needs document_year for every molecule")
        year_values = pd.to_numeric(pd.Series(years), errors="coerce").to_numpy(dtype=float)
        if len(year_values) != len(y) or np.isnan(year_values).any():
            raise ValueError("time split needs a known document_year for every molecule")
        train = np.flatnonzero(year_values <= params.time_cutoff)
        test = np.flatnonzero(year_values > params.time_cutoff)
        if not len(train) or not len(test):
            raise ValueError(
                f"time split at {params.time_cutoff} produced {len(train)} training and "
                f"{len(test)} test molecules"
            )
    elif params.split == "scaffold":
        n_splits = max(2, round(1 / params.test_fraction))
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=params.seed)
        train, test = next(splitter.split(X, y, groups))
    else:
        shuffle = StratifiedShuffleSplit(
            n_splits=1, test_size=params.test_fraction, random_state=params.seed
        )
        train, test = next(shuffle.split(X, y))
    return np.sort(train).astype(np.intp), np.sort(test).astype(np.intp)


def _cv_folds(
    y: NDArray[Any],
    groups: NDArray[Any] | None,
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
    if params.split == "time":
        if years is None:
            raise ValueError("time cross-validation needs document years")
        year_values = pd.to_numeric(pd.Series(years), errors="coerce").to_numpy(dtype=float)
        if len(year_values) != len(y) or np.isnan(year_values).any():
            raise ValueError("time cross-validation needs a known year for every molecule")
        unique_years = np.unique(year_values)
        if len(unique_years) <= params.cv_folds:
            raise ValueError(
                f"time cross-validation needs at least {params.cv_folds + 1} distinct "
                f"training years for {params.cv_folds} expanding folds; found {len(unique_years)}"
            )
        folds = []
        for validation_year in unique_years[-params.cv_folds :]:
            fit = np.flatnonzero(year_values < validation_year).astype(np.intp)
            validation = np.flatnonzero(year_values == validation_year).astype(np.intp)
            if len(np.unique(y[fit])) < 2:
                raise ValueError(
                    f"time cross-validation has fewer than two activity classes "
                    f"before {int(validation_year)}; choose a later cutoff or fewer CV folds"
                )
            folds.append((fit, validation))
        return iter(folds)
    placeholder = np.zeros(len(y))
    if groups is not None and params.split == "scaffold":
        sgkf = StratifiedGroupKFold(params.cv_folds, shuffle=True, random_state=params.seed)
        return iter(sgkf.split(placeholder, y, groups))
    skf = StratifiedKFold(params.cv_folds, shuffle=True, random_state=params.seed)
    return iter(skf.split(placeholder, y))


def oversample(X: NDArray[Any], y: NDArray[Any], seed: int) -> tuple[NDArray[Any], NDArray[Any]]:
    """Duplicate random minority-class rows until every class matches the largest."""
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    index = [np.arange(len(y))]
    for label, count in zip(classes, counts, strict=True):
        members = np.flatnonzero(y == label)
        index.append(rng.choice(members, counts.max() - count, replace=True))
    order = np.concatenate(index)
    return X[order], y[order]


def _metrics(y_true: NDArray[Any], y_pred: NDArray[Any]) -> tuple[float, float, float]:
    return (
        float(accuracy_score(y_true, y_pred)),
        float(balanced_accuracy_score(y_true, y_pred)),
        float(matthews_corrcoef(y_true, y_pred)),
    )


def _score_matrix(
    model: Any, X: NDArray[Any]
) -> tuple[tuple[str, ...], NDArray[np.float64] | None]:
    """Class-aligned probability or decision scores for ranking metrics."""
    classes = tuple(str(value) for value in model.classes_)
    raw = None
    if hasattr(model, "predict_proba"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = np.asarray(model.predict_proba(X), dtype=float)
        except (ValueError, FloatingPointError):
            pass
    if raw is None and hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(X), dtype=float)
        if raw.ndim == 1 and len(classes) == 2:
            raw = np.column_stack((-raw, raw))
    if raw is None:
        return classes, None
    if raw.shape != (len(X), len(classes)) or not np.isfinite(raw).all():
        return classes, None
    return classes, raw


def _ranking_metrics(
    truth: NDArray[Any], classes: tuple[str, ...], scores: NDArray[np.float64] | None
) -> tuple[float, float, int]:
    """Macro one-vs-rest ROC AUC and average precision over evaluable classes."""
    if scores is None:
        return math.nan, math.nan, 0
    roc: list[float] = []
    precision: list[float] = []
    for column, label in enumerate(classes):
        positive = np.asarray(truth == label, dtype=bool)
        if not positive.any() or positive.all():
            continue
        roc.append(float(roc_auc_score(positive, scores[:, column])))
        precision.append(float(average_precision_score(positive, scores[:, column])))
    if not roc:
        return math.nan, math.nan, 0
    return float(np.mean(roc)), float(np.mean(precision)), len(roc)


def _class_diagnostics(
    truth: NDArray[Any],
    predictions: NDArray[Any],
    classes: tuple[str, ...],
    scores: NDArray[np.float64] | None,
) -> pd.DataFrame:
    labels = sorted(set(classes) | set(map(str, truth)) | set(map(str, predictions)))
    prec, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    columns = {label: index for index, label in enumerate(classes)}
    rows = []
    for index, label in enumerate(labels):
        positive = np.asarray(truth == label, dtype=bool)
        roc = ap = math.nan
        if scores is not None and label in columns and positive.any() and not positive.all():
            class_score = scores[:, columns[label]]
            roc = float(roc_auc_score(positive, class_score))
            ap = float(average_precision_score(positive, class_score))
        rows.append(
            {
                "activity_class": label,
                "support": int(support[index]),
                "prevalence": float(positive.mean()),
                "precision": float(prec[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "roc_auc_ovr": roc,
                "pr_auc_ovr": ap,
            }
        )
    return pd.DataFrame(rows)


def _fit(name: str, X: NDArray[Any], y: NDArray[Any], seed: int) -> Any:
    with warnings.catch_warnings():
        # QDA on binary bits reports collinearity and MLP/SGD may not fully
        # converge; the scores say what matters and the log would drown in it.
        warnings.simplefilter("ignore")
        return ALGORITHMS[name](seed).fit(X, y)


def _predict(model: Any, X: NDArray[Any]) -> NDArray[Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(X))


def _fit_leak_free(
    name: str, X: NDArray[Any], y: NDArray[Any], params: ModelParams
) -> tuple[VarianceCorrelationFilter, Any]:
    """Filter fitted on these rows only, then oversample these rows only."""
    feats = params.features
    filt = VarianceCorrelationFilter(feats.variance_threshold, feats.correlation_threshold).fit(X)
    Xf = filt.transform(X)
    if params.oversample:
        Xf, y = oversample(Xf, y, params.seed)
    return filt, _fit(name, Xf, y, params.seed)


def _row(
    name: str,
    protocol: str,
    train: tuple[float, float, float],
    cv: list[tuple[float, float, float]],
    test: tuple[float, float, float],
    cv_ranking: list[tuple[float, float, int]],
    test_ranking: tuple[float, float, int],
    n_features: int,
) -> dict[str, Any]:
    cv_arr = np.array(cv, dtype=float)
    cv_rank_arr = np.array(cv_ranking, dtype=float)

    def rank_mean(column: int) -> float:
        finite = cv_rank_arr[np.isfinite(cv_rank_arr[:, column]), column]
        return float(finite.mean()) if len(finite) else math.nan

    return {
        "algorithm": name,
        "protocol": protocol,
        "train_accuracy": train[0],
        "cv_accuracy": float(cv_arr[:, 0].mean()),
        "cv_accuracy_sd": float(cv_arr[:, 0].std(ddof=1)) if len(cv) > 1 else math.nan,
        "test_accuracy": test[0],
        "train_balanced_accuracy": train[1],
        "cv_balanced_accuracy": float(cv_arr[:, 1].mean()),
        "test_balanced_accuracy": test[1],
        "train_mcc": train[2],
        "cv_mcc": float(cv_arr[:, 2].mean()),
        "cv_mcc_sd": float(cv_arr[:, 2].std(ddof=1)) if len(cv) > 1 else math.nan,
        "test_mcc": test[2],
        "cv_roc_auc_ovr": rank_mean(0),
        "test_roc_auc_ovr": test_ranking[0],
        "cv_pr_auc_ovr": rank_mean(1),
        "test_pr_auc_ovr": test_ranking[1],
        "test_auc_classes": test_ranking[2],
        "n_features": n_features,
    }


def _leak_free(
    name: str,
    X: NDArray[Any],
    y: NDArray[Any],
    groups: NDArray[Any],
    train_idx: NDArray[np.intp],
    test_idx: NDArray[np.intp],
    params: ModelParams,
    years: Sequence[int | float | None] | None = None,
) -> dict[str, Any]:
    X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
    cv = []
    cv_ranking = []
    train_years = [years[i] for i in train_idx] if years is not None else None
    for fit_rows, eval_rows in _cv_folds(y_tr, g_tr, params, train_years):
        filt, model = _fit_leak_free(name, X_tr[fit_rows], y_tr[fit_rows], params)
        X_eval = filt.transform(X_tr[eval_rows])
        cv.append(_metrics(y_tr[eval_rows], _predict(model, X_eval)))
        cv_ranking.append(_ranking_metrics(y_tr[eval_rows], *_score_matrix(model, X_eval)))
    filt, model = _fit_leak_free(name, X_tr, y_tr, params)
    train = _metrics(y_tr, _predict(model, filt.transform(X_tr)))
    X_test = filt.transform(X[test_idx])
    test = _metrics(y[test_idx], _predict(model, X_test))
    test_ranking = _ranking_metrics(y[test_idx], *_score_matrix(model, X_test))
    return _row(
        name, "leak_free", train, cv, test, cv_ranking, test_ranking, filt.n_after_correlation_
    )


def _naive_protocol(
    name: str, X: NDArray[Any], y: NDArray[Any], params: ModelParams
) -> dict[str, Any]:
    """Select features on everything, oversample everything, then split.

    The leaky ordering, run deliberately so the gap can be measured.
    """
    feats = params.features
    filt = VarianceCorrelationFilter(feats.variance_threshold, feats.correlation_threshold).fit(X)
    Xo, yo = oversample(filt.transform(X), y, params.seed)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xo, yo, test_size=params.test_fraction, random_state=params.seed, stratify=yo
    )
    cv = []
    cv_ranking = []
    for fit_rows, eval_rows in _cv_folds(y_tr, None, params):
        model = _fit(name, X_tr[fit_rows], y_tr[fit_rows], params.seed)
        cv.append(_metrics(y_tr[eval_rows], _predict(model, X_tr[eval_rows])))
        cv_ranking.append(_ranking_metrics(y_tr[eval_rows], *_score_matrix(model, X_tr[eval_rows])))
    model = _fit(name, X_tr, y_tr, params.seed)
    train = _metrics(y_tr, _predict(model, X_tr))
    test = _metrics(y_te, _predict(model, X_te))
    test_ranking = _ranking_metrics(y_te, *_score_matrix(model, X_te))
    return _row(name, "naive", train, cv, test, cv_ranking, test_ranking, filt.n_after_correlation_)

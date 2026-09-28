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
       "random": StratifiedShuffleSplit.
    2. Cross-validate on the training set: StratifiedGroupKFold / StratifiedKFold
       with ``cv_folds`` folds.
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
from dataclasses import dataclass
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
from sklearn.metrics import accuracy_score, balanced_accuracy_score, matthews_corrcoef
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


def evaluate(
    X: NDArray[Any],
    y: Sequence[str],
    groups: Sequence[str | None],
    params: ModelParams,
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
    train_idx, test_idx = _outer_split(X, labels, group_ids, params)

    rows = []
    for name in names:
        rows.append(_leak_free(name, X, labels, group_ids, train_idx, test_idx, params))
        if params.leakage_audit:
            rows.append(_naive_protocol(name, X, labels, params))
    scores = pd.DataFrame(rows, columns=list(SCORE_COLUMNS))

    ranked = scores[scores["protocol"] == "leak_free"].assign(
        _key=lambda f: f["cv_mcc"].fillna(-math.inf)
    )
    best = ranked.sort_values(["_key", "algorithm"], ascending=[False, True]).iloc[0]["algorithm"]
    return ModelResult(scores, str(best), test_idx, train_idx)


# -- protocol pieces ---------------------------------------------------------


def _outer_split(
    X: NDArray[Any], y: NDArray[Any], groups: NDArray[Any], params: ModelParams
) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    if params.split == "scaffold":
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
    y: NDArray[Any], groups: NDArray[Any] | None, params: ModelParams
) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
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
    n_features: int,
) -> dict[str, Any]:
    cv_arr = np.array(cv, dtype=float)
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
) -> dict[str, Any]:
    X_tr, y_tr, g_tr = X[train_idx], y[train_idx], groups[train_idx]
    cv = []
    for fit_rows, eval_rows in _cv_folds(y_tr, g_tr, params):
        filt, model = _fit_leak_free(name, X_tr[fit_rows], y_tr[fit_rows], params)
        cv.append(_metrics(y_tr[eval_rows], _predict(model, filt.transform(X_tr[eval_rows]))))
    filt, model = _fit_leak_free(name, X_tr, y_tr, params)
    train = _metrics(y_tr, _predict(model, filt.transform(X_tr)))
    test = _metrics(y[test_idx], _predict(model, filt.transform(X[test_idx])))
    return _row(name, "leak_free", train, cv, test, filt.n_after_correlation_)


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
    for fit_rows, eval_rows in _cv_folds(y_tr, None, params):
        model = _fit(name, X_tr[fit_rows], y_tr[fit_rows], params.seed)
        cv.append(_metrics(y_tr[eval_rows], _predict(model, X_tr[eval_rows])))
    model = _fit(name, X_tr, y_tr, params.seed)
    train = _metrics(y_tr, _predict(model, X_tr))
    test = _metrics(y_te, _predict(model, X_te))
    return _row(name, "naive", train, cv, test, filt.n_after_correlation_)

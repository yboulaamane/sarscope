"""Multiclass QSAR bake-off, with the validation done in an order that cannot leak.

The fourteen algorithms and their hyperparameters are the paper's Table 1,
which follows scikit-learn's "classifier comparison" example (hence GaussianNB
and QDA with defaults). ``random_state=seed`` wherever the estimator takes one.

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

**The paper's protocol** (``leakage_audit``) is: select features on all data,
oversample all data, then split 80:20 and cross-validate. On BRAF that
oversampling turns 3,952 molecules into 4,872 (Fig. 3: 4 x 1,218 potent), so
920 rows are copies. After a random split, copies of one molecule sit in both
train and test, and in both sides of every CV fold, so the model is partly
graded on molecules it trained on. Running both protocols on the same data
and reporting the gap turns "this leaks" from an assertion into a number.

**Metrics:** accuracy, balanced accuracy, and multiclass MCC
(``sklearn.metrics.matthews_corrcoef``). The paper also reports micro-averaged
recall, but in single-label multiclass that is identical to accuracy by
definition, so it is not repeated. "train" is resubstitution on the training
set, reported because the paper reports it, never used for ranking.

**Choosing the best model uses CV MCC, not test.** Picking the winner by test
score is a second leak; the test set is scored once, for the chosen model and
for comparison.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sarscope.params import ModelParams

#: name -> factory(seed) -> unfitted estimator. Filled in with the paper's
#: Table 1 when this module is implemented:
#:
#:   nearest_neighbors   KNeighborsClassifier(n_neighbors=4, weights="uniform")
#:   linear_svm          SVC(kernel="linear", C=0.25)
#:   polynomial_svm      SVC(kernel="poly", C=0.025)
#:   rbf_svm             SVC(kernel="rbf", C=2, gamma=2)
#:   gaussian_process    GaussianProcessClassifier(1.0 * RBF(1.0))
#:   gradient_boosting   GradientBoostingClassifier(n_estimators=200)
#:   decision_tree       DecisionTreeClassifier(max_depth=5, min_samples_split=8)
#:   extra_trees         ExtraTreesClassifier(n_estimators=200, min_samples_split=7)
#:   random_forest       RandomForestClassifier(n_estimators=200, max_depth=15)
#:   neural_net          MLPClassifier(alpha=0.5, max_iter=1500)
#:   adaboost            AdaBoostClassifier(n_estimators=200)
#:   naive_bayes         GaussianNB()
#:   qda                 QuadraticDiscriminantAnalysis()
#:   sgd                 SGDClassifier(loss="squared_error", penalty="l2")
ALGORITHMS: dict[str, Callable[[int], Any]] = {}

#: Columns of ModelResult.scores.
SCORE_COLUMNS: tuple[str, ...] = (
    "algorithm",
    "protocol",  # "leak_free" or "paper"
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
    and also under the paper's protocol if ``params.leakage_audit``.

    ``groups`` is the Murcko scaffold per row (None for acyclic), used only when
    ``params.split == "scaffold"``. The single name "all" expands to every key
    of ALGORITHMS. Unknown names are a ValueError raised before any fitting.
    """
    raise NotImplementedError

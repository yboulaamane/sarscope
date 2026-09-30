"""Specification for sarscope.analysis.model."""

import numpy as np
import pytest

from sarscope.analysis import model
from sarscope.analysis.model import ALGORITHMS, SCORE_COLUMNS, evaluate
from sarscope.params import ModelParams

pytestmark = pytest.mark.science

TABLE_1 = {
    "nearest_neighbors",
    "linear_svm",
    "polynomial_svm",
    "rbf_svm",
    "gaussian_process",
    "gradient_boosting",
    "decision_tree",
    "extra_trees",
    "random_forest",
    "neural_net",
    "adaboost",
    "naive_bayes",
    "qda",
    "sgd",
}

FAST = ("nearest_neighbors", "extra_trees")


def test_all_fourteen_algorithms_are_registered():
    assert set(ALGORITHMS) == TABLE_1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("nearest_neighbors", {"n_neighbors": 4, "weights": "uniform"}),
        ("linear_svm", {"kernel": "linear", "C": 0.25}),
        ("polynomial_svm", {"kernel": "poly", "C": 0.025}),
        ("rbf_svm", {"kernel": "rbf", "C": 2, "gamma": 2}),
        ("gradient_boosting", {"n_estimators": 200}),
        ("decision_tree", {"max_depth": 5, "min_samples_split": 8}),
        ("extra_trees", {"n_estimators": 200, "min_samples_split": 7}),
        ("random_forest", {"n_estimators": 200, "max_depth": 15}),
        ("neural_net", {"alpha": 0.5, "max_iter": 1500}),
        ("adaboost", {"n_estimators": 200}),
        ("sgd", {"loss": "squared_error", "penalty": "l2"}),
    ],
)
def test_table_1_hyperparameters(name, expected):
    params = ALGORITHMS[name](42).get_params()
    assert {k: params[k] for k in expected} == expected


def test_seed_is_passed_through():
    assert ALGORITHMS["extra_trees"](7).get_params()["random_state"] == 7


def dataset(signal: bool, seed: int = 0):
    """320 molecules, 4 imbalanced classes, 64 bits, 40 scaffolds.

    With ``signal``, bits 8k..8k+7 are on exactly for class k: a perfect
    indicator per class (the filter keeps one bit of each block, and drops the
    rarest class's as low-variance). Prototyped on three seeds with the default
    hyperparameters and a scaffold split: Extra Trees MCC 0.92-0.94,
    nearest neighbours 0.43-0.50 (the 32 random bits dominate its distances).
    """
    rng = np.random.default_rng(seed)
    y = np.array(["a"] * 200 + ["b"] * 60 + ["c"] * 40 + ["d"] * 20)
    rng.shuffle(y)
    X = rng.integers(0, 2, (len(y), 64)).astype(np.uint8)
    if signal:
        for k, label in enumerate("abcd"):
            X[:, k * 8 : k * 8 + 8] = 0
            X[y == label, k * 8 : k * 8 + 8] = 1
    groups = [f"s{i % 40}" for i in range(len(y))]
    return X, list(y), groups


def fast_params(**kw) -> ModelParams:
    return ModelParams(algorithms=FAST, cv_folds=3, **kw)


def test_result_shape():
    X, y, groups = dataset(signal=True)
    result = evaluate(X, y, groups, fast_params())
    assert list(result.scores.columns) == list(SCORE_COLUMNS)
    assert set(zip(result.scores["algorithm"], result.scores["protocol"], strict=True)) == {
        (a, p) for a in FAST for p in ("leak_free", "naive")
    }
    assert result.best_algorithm in FAST
    assert not set(result.train_index) & set(result.test_index)
    assert len(result.train_index) + len(result.test_index) == len(y)
    assert 0 <= result.scores.loc[0, "test_roc_auc_ovr"] <= 1
    assert 0 <= result.scores.loc[0, "test_pr_auc_ovr"] <= 1
    assert set(result.test_class_metrics["activity_class"]) == set(y)
    assert int(result.test_confusion.to_numpy().sum()) == len(result.test_index)


def test_roc_and_pr_metrics_use_scores_and_show_undefined_classes():
    truth = np.array(["inactive", "active", "inactive", "active"])
    classes = ("active", "inactive")
    scores = np.array([[0.1, 0.9], [0.9, 0.1], [0.2, 0.8], [0.8, 0.2]])
    assert model._ranking_metrics(truth, classes, scores) == (1.0, 1.0, 2)
    diagnostics = model._class_diagnostics(truth, truth, classes, scores)
    assert set(diagnostics["prevalence"]) == {0.5}
    assert set(diagnostics["f1"]) == {1.0}
    roc, ap, count = model._ranking_metrics(np.array(["active"] * 4), classes, scores)
    assert np.isnan(roc) and np.isnan(ap) and count == 0


def test_decision_function_is_used_when_probabilities_are_unavailable():
    X = np.array([[0.0], [0.1], [0.9], [1.0]])
    y = np.array(["inactive", "inactive", "active", "active"])
    fitted = ALGORITHMS["linear_svm"](42).fit(X, y)
    assert not hasattr(fitted, "predict_proba")
    classes, scores = model._score_matrix(fitted, X)
    assert scores is not None and scores.shape == (4, 2)
    assert model._ranking_metrics(y, classes, scores) == (1.0, 1.0, 2)


def test_missing_ranking_scores_leave_auc_undefined():
    class LabelsOnly:
        classes_ = np.array(["active", "inactive"])

    classes, scores = model._score_matrix(LabelsOnly(), np.zeros((3, 2)))
    assert scores is None
    roc, ap, count = model._ranking_metrics(
        np.array(["active", "inactive", "active"]), classes, scores
    )
    assert np.isnan(roc) and np.isnan(ap) and count == 0


def test_no_audit_means_leak_free_only():
    X, y, groups = dataset(signal=True)
    result = evaluate(X, y, groups, fast_params(leakage_audit=False))
    assert set(result.scores["protocol"]) == {"leak_free"}


def test_scaffold_split_keeps_each_scaffold_on_one_side():
    X, y, groups = dataset(signal=True)
    result = evaluate(X, y, groups, fast_params(split="scaffold"))
    train = {groups[i] for i in result.train_index}
    test = {groups[i] for i in result.test_index}
    assert not train & test


def test_time_cv_uses_expanding_year_folds_without_future_training_rows():
    years = np.repeat([2017, 2018, 2019, 2020], 8)
    labels = np.tile(["active", "inactive"], 16)
    params = fast_params(split="time", leakage_audit=False)
    folds = list(model._cv_folds(labels, None, params, years))
    assert len(folds) == 3
    for (fit, validation), year in zip(folds, [2018, 2019, 2020], strict=True):
        assert set(years[fit]) == set(range(2017, year))
        assert set(years[validation]) == {year}
        assert years[fit].max() < years[validation].min()


def test_time_split_classification_scores_only_newer_compounds():
    rng = np.random.default_rng(31)
    X = rng.integers(0, 2, (100, 20), dtype=np.uint8)
    labels = np.tile(["active", "inactive"], 50)
    years = np.repeat([2017, 2018, 2019, 2020, 2021], 20)
    params = ModelParams(
        algorithms=("extra_trees",),
        split="time",
        time_cutoff=2020,
        cv_folds=3,
        leakage_audit=True,
    )
    result = evaluate(X, labels, [f"s{i}" for i in range(100)], params, years)
    assert years[result.train_index].max() == 2020
    assert set(years[result.test_index]) == {2021}
    assert set(result.scores["protocol"]) == {"leak_free"}


def test_source_split_holds_out_only_marked_compounds():
    rng = np.random.default_rng(51)
    X = rng.integers(0, 2, (100, 24), dtype=np.uint8)
    labels = np.tile(["potent", "active", "intermediate", "inactive"], 25)
    mask = np.array([False] * 80 + [True] * 20)
    params = ModelParams(
        algorithms=("extra_trees",),
        split="source",
        source_test_id=7,
        cv_folds=3,
        leakage_audit=True,
    )
    result = evaluate(X, labels, [f"s{i}" for i in range(100)], params, source_test=mask)
    assert set(result.train_index) == set(range(80))
    assert set(result.test_index) == set(range(80, 100))
    assert set(result.scores["protocol"]) == {"leak_free"}


def test_time_cv_explains_insufficient_years():
    labels = np.array(["a", "b"] * 15)
    params = fast_params(split="time")
    with pytest.raises(ValueError, match="at least 4 distinct training years"):
        list(model._cv_folds(labels, None, params, np.repeat([2018, 2019, 2020], 10)))


def test_learns_real_signal():
    X, y, groups = dataset(signal=True)
    scores = evaluate(X, y, groups, fast_params()).scores
    leak_free = scores[scores["protocol"] == "leak_free"].set_index("algorithm")
    assert leak_free.loc["extra_trees", "test_mcc"] > 0.8
    assert leak_free.loc["nearest_neighbors", "test_mcc"] > 0.3


def test_the_naive_protocol_scores_noise_as_signal():
    # Labels are pure noise: the honest MCC is about zero. Oversampling before
    # splitting puts copies of training molecules in the test set. Prototyped
    # on five seeds: Extra Trees leak-free 0.00 vs naive order 0.98-0.99.
    X, y, groups = dataset(signal=False)
    scores = evaluate(X, y, groups, fast_params(split="random")).scores
    et = scores[scores["algorithm"] == "extra_trees"].set_index("protocol")
    assert abs(et.loc["leak_free", "test_mcc"]) < 0.2
    assert et.loc["naive", "test_mcc"] > et.loc["leak_free", "test_mcc"] + 0.5


def test_best_model_is_chosen_on_cv_not_test():
    X, y, groups = dataset(signal=True)
    result = evaluate(X, y, groups, fast_params())
    leak_free = result.scores[result.scores["protocol"] == "leak_free"]
    assert (
        result.best_algorithm
        == leak_free.sort_values(["cv_mcc", "algorithm"], ascending=[False, True]).iloc[0][
            "algorithm"
        ]
    )


def test_all_expands_to_every_registered_algorithm(monkeypatch):
    monkeypatch.setattr(model, "ALGORITHMS", {k: ALGORITHMS[k] for k in FAST})
    X, y, groups = dataset(signal=True)
    result = evaluate(X, y, groups, ModelParams(algorithms=("all",), cv_folds=3))
    assert set(result.scores["algorithm"]) == set(FAST)


def test_unknown_algorithm_fails_before_fitting():
    X, y, groups = dataset(signal=True)
    with pytest.raises(ValueError, match="no_such_model"):
        evaluate(X, y, groups, ModelParams(algorithms=("extra_trees", "no_such_model")))

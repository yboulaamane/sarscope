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
    rarest class's as low-variance). Prototyped on three seeds with the paper's
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
        (a, p) for a in FAST for p in ("leak_free", "paper")
    }
    assert result.best_algorithm in FAST
    assert not set(result.train_index) & set(result.test_index)
    assert len(result.train_index) + len(result.test_index) == len(y)


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


def test_learns_real_signal():
    X, y, groups = dataset(signal=True)
    scores = evaluate(X, y, groups, fast_params()).scores
    leak_free = scores[scores["protocol"] == "leak_free"].set_index("algorithm")
    assert leak_free.loc["extra_trees", "test_mcc"] > 0.8
    assert leak_free.loc["nearest_neighbors", "test_mcc"] > 0.3


def test_the_papers_protocol_scores_noise_as_signal():
    # Labels are pure noise: the honest MCC is about zero. Oversampling before
    # splitting puts copies of training molecules in the test set. Prototyped
    # on five seeds: Extra Trees leak-free 0.00 vs paper order 0.98-0.99.
    X, y, groups = dataset(signal=False)
    scores = evaluate(X, y, groups, fast_params(split="random")).scores
    et = scores[scores["algorithm"] == "extra_trees"].set_index("protocol")
    assert abs(et.loc["leak_free", "test_mcc"]) < 0.2
    assert et.loc["paper", "test_mcc"] > et.loc["leak_free", "test_mcc"] + 0.5


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

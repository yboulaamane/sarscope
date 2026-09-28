import numpy as np
import pytest

from sarscope.analysis.regression import REGRESSION_SCORE_COLUMNS, evaluate_regression
from sarscope.params import FeatureParams, ModelParams

pytestmark = pytest.mark.science


def test_continuous_models_report_r2_rmse_and_spearman():
    rng = np.random.default_rng(4)
    X = rng.integers(0, 2, (120, 48), dtype=np.uint8)
    y = 5.0 + 1.2 * X[:, 0] + 0.8 * X[:, 1] + rng.normal(0, 0.1, len(X))
    labels = np.where(y >= 6.5, "potent", np.where(y >= 5.8, "active", "inactive"))
    groups = [f"s{i % 20}" for i in range(len(X))]
    params = ModelParams(
        regression_algorithms=("random_forest",),
        cv_folds=3,
        leakage_audit=False,
        features=FeatureParams(variance_threshold=0.0),
    )
    result = evaluate_regression(X, y, groups, labels, params)
    assert list(result.scores) == list(REGRESSION_SCORE_COLUMNS)
    assert result.best_algorithm == "random_forest"
    assert len(result.test_predictions) == len(result.test_index)
    assert result.scores.iloc[0]["test_rmse"] < 0.5
    assert result.scores.iloc[0]["test_spearman"] > 0.7


def test_time_split_reserves_only_later_compounds():
    rng = np.random.default_rng(8)
    X = rng.integers(0, 2, (80, 24), dtype=np.uint8)
    y = np.linspace(5, 9, 80)
    labels = np.tile(["a", "b", "c", "d"], 20)
    years = np.repeat([2017, 2018, 2019, 2020], 20)
    params = ModelParams(
        regression_algorithms=("extra_trees",),
        split="time",
        time_cutoff=2019,
        cv_folds=3,
        features=FeatureParams(variance_threshold=0.0),
    )
    result = evaluate_regression(X, y, [f"s{i}" for i in range(80)], labels, params, years)
    assert set(years[result.train_index]) == {2017, 2018, 2019}
    assert set(years[result.test_index]) == {2020}

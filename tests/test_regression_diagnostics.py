from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sarscope.analysis.diagnostics import bootstrap_regression, novelty_summary, regression_summary
from sarscope.analysis.regression import evaluate_regression
from sarscope.params import FeatureParams, ModelParams


def result():
    truth = np.array([5, 6, 7, 8, 6, 7], dtype=float)
    prediction = truth + np.array([0, 0.1, -0.4, 1.2, -0.7, 0.2])
    return SimpleNamespace(
        test_truth=truth,
        test_predictions=prediction,
        baseline_test_predictions=np.full(len(truth), 6.3),
    )


def test_regression_summary_fold_errors_and_baseline_are_numeric():
    metrics = regression_summary(result()).set_index("metric")["value"]
    assert metrics["Within 3-fold"] == pytest.approx(4 / 6)
    assert metrics["Within 10-fold"] == pytest.approx(5 / 6)
    assert metrics["Test MAE"] == pytest.approx(2.6 / 6)
    assert metrics["Typical fold error"] == pytest.approx(10**0.3)
    assert metrics["RMSE gain versus baseline"] > 0


def test_block_bootstrap_is_deterministic_paired_and_reports_scaffold_count():
    groups = ["a", "a", "b", "b", None, None]
    first = bootstrap_regression(result(), groups, repeats=60)
    second = bootstrap_regression(result(), groups, repeats=60)
    pd.testing.assert_frame_equal(first, second)
    assert first["scaffold_blocks"].eq(4).all()  # each acyclic compound is a separate block
    assert (first["lower_95"] <= first["upper_95"]).all()
    same = result()
    same.baseline_test_predictions = same.test_predictions.copy()
    delta = bootstrap_regression(same, groups, repeats=60).iloc[-1]
    assert delta["lower_95"] == delta["upper_95"] == 0
    with pytest.raises(ValueError, match="independent"):
        bootstrap_regression(result(), ["same"] * 6)


def test_novelty_summary_preserves_counts_and_error_scale():
    profile = pd.DataFrame(
        {
            "max_training_similarity": [0, 0.3, 0.4, 0.6, 0.8, 1],
            "absolute_error": [1, 0.5, 0.4, 0.3, 0.2, 0],
        }
    )
    summary = novelty_summary(profile)
    assert summary["n"].sum() == len(profile)
    assert summary.iloc[0]["rmse"] == pytest.approx(np.sqrt((1 + 0.25) / 2))


def test_baseline_is_training_mean_not_test_mean_and_not_selected_instead_of_requested_models():
    rng = np.random.default_rng(11)
    X = rng.integers(0, 2, size=(80, 10))
    y = X[:, 0] + rng.normal(size=80)
    params = ModelParams(
        features=FeatureParams(variance_threshold=0),
        split="random",
        cv_folds=3,
        regression_algorithms=("ridge",),
    )
    fitted = evaluate_regression(X, y, [f"s{i}" for i in range(80)], ["a", "b"] * 40, params)
    assert fitted.best_algorithm == "ridge"
    assert set(fitted.scores["algorithm"]) == {"ridge", "mean_baseline"}
    np.testing.assert_allclose(fitted.baseline_test_predictions, y[fitted.train_index].mean())
    assert {"cv_mae", "test_mae"} <= set(fitted.scores)

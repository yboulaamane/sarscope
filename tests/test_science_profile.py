"""Specification for sarscope.analysis.profile."""

import numpy as np
import pandas as pd
import pytest
from scipy.stats import mannwhitneyu

from sarscope.analysis.profile import describe_groups, property_pca

pytestmark = pytest.mark.science

COLS = ("x", "y", "z")


@pytest.fixture
def table() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 120
    frame = pd.DataFrame(
        {
            "x": rng.normal(0, 1, n),
            "y": rng.gamma(2.0, 2.0, n),
            "z": rng.normal(5, 2, n),
            "group": [1] * 70 + [2] * 50,
        }
    )
    frame.loc[frame["group"] == 1, "x"] += 1.0
    frame["y"] += 0.5 * frame["x"]
    return frame


def test_stats_match_pandas(table):
    stats = describe_groups(table, COLS).stats
    assert list(stats.columns) == ["n", "max", "min", "mean", "median", "skew", "kurtosis"]
    g1 = table.loc[table["group"] == 1, "y"]
    row = stats.loc[("y", 1)]
    assert row["n"] == 70
    assert row["mean"] == pytest.approx(g1.mean())
    assert row["median"] == pytest.approx(g1.median())
    assert row["skew"] == pytest.approx(g1.skew())
    assert row["kurtosis"] == pytest.approx(g1.kurt())  # Fisher excess, bias-corrected


def test_kurtosis_is_excess_so_it_can_be_negative():
    # A uniform distribution has excess kurtosis -1.2; the paper reports negatives.
    frame = pd.DataFrame({"x": np.linspace(0, 1, 200), "group": [1, 2] * 100})
    assert describe_groups(frame, ("x",)).stats.loc[("x", 1), "kurtosis"] < 0


def test_p_values_are_two_sided_mann_whitney(table):
    p = describe_groups(table, COLS).p_values
    g1, g2 = (table.loc[table["group"] == g, "x"] for g in (1, 2))
    assert p["x"] == pytest.approx(mannwhitneyu(g1, g2, alternative="two-sided").pvalue)
    assert p["x"] < 1e-4


def test_constant_property_gives_nan_p_value(table):
    table = table.assign(c=3.0)
    assert np.isnan(describe_groups(table, ("x", "c")).p_values["c"])


def test_requires_exactly_two_groups(table):
    with pytest.raises(ValueError):
        describe_groups(table.assign(group=1), COLS)


def test_pca_standardises_so_units_do_not_matter(table):
    a = property_pca(table, COLS)
    b = property_pca(table.assign(x=table["x"] * 1000.0), COLS)
    pd.testing.assert_frame_equal(a.loadings, b.loadings, atol=1e-8)


def test_pca_sign_convention_and_shapes(table):
    result = property_pca(table, COLS, n_components=3)
    assert list(result.loadings.columns) == ["PC1", "PC2", "PC3"]
    assert list(result.loadings.index) == list(COLS)
    for pc in result.loadings.columns:
        column = result.loadings[pc]
        assert column[column.abs().idxmax()] > 0
    assert result.scores.shape == (len(table), 3)
    assert list(result.scores.index) == list(table.index)
    assert result.cumulative.is_monotonic_increasing
    assert result.cumulative.iloc[-1] == pytest.approx(1.0)


def test_pca_components_capped_at_number_of_properties(table):
    assert property_pca(table, ("x", "y"), n_components=5).loadings.shape == (2, 2)

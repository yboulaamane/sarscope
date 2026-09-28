"""Descriptor statistics by activity group (Table 2) and property PCA (Table 3).

**Skewness and kurtosis are pandas' ``Series.skew()`` / ``Series.kurt()``**:
bias-corrected, and kurtosis is Fisher *excess* kurtosis (normal = 0). Two
reasons: the paper states it computed these in pandas, and it reports a
negative kurtosis for Group 1 NumHDonors, which only excess kurtosis can
produce. ``scipy.stats.kurtosis`` defaults to ``bias=True`` and gives different
numbers, so do not substitute it.

**The PCA standardises first** (z-score each property, then PCA). The paper does
not say so, but its Table 3 does: PC1 loads MW 0.498, TPSA 0.485, RB 0.484.
Unscaled, MW's variance (hundreds of Da squared) would swamp everything and PC1
would be MW alone.

Principal-component signs are arbitrary, so fix them: flip each component so
its largest-magnitude loading is positive. Otherwise two runs, or two scikit-
learn versions, can print mirror-image tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from sarscope.analysis.descriptors import PAPER_DESCRIPTORS


@dataclass
class GroupProfile:
    #: Index: MultiIndex (property, group). Columns: n, max, min, mean,
    #: median, skew, kurtosis.
    stats: pd.DataFrame
    #: Index: property. Two-sided Mann-Whitney U p-value, Group 1 vs Group 2.
    p_values: pd.Series


@dataclass
class PcaResult:
    #: One row per molecule (same index as the input), columns PC1..PCk.
    scores: pd.DataFrame
    #: Index: property. Columns PC1..PCk. Sign-fixed as described above.
    loadings: pd.DataFrame
    #: Index PC1..PCk. Fraction of variance per component.
    explained: pd.Series

    @property
    def cumulative(self) -> pd.Series:
        return self.explained.cumsum()


def describe_groups(
    table: pd.DataFrame,
    columns: Sequence[str] = PAPER_DESCRIPTORS,
    group_col: str = "group",
) -> GroupProfile:
    """Table 2. Exactly two groups must be present, else ValueError.

    p-values from ``scipy.stats.mannwhitneyu(g1, g2, alternative="two-sided")``.
    A property that is constant across both groups has no defined test; report
    NaN for it rather than raising.
    """
    raise NotImplementedError


def property_pca(
    table: pd.DataFrame,
    columns: Sequence[str] = PAPER_DESCRIPTORS,
    n_components: int = 3,
) -> PcaResult:
    """Table 3 and the Fig. 5 scatter. ``n_components`` is capped at len(columns)."""
    raise NotImplementedError

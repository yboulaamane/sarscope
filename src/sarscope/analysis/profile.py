"""Descriptor statistics by activity group, and the property PCA.

**Skewness and kurtosis are pandas' ``Series.skew()`` / ``Series.kurt()``**:
bias-corrected, and kurtosis is Fisher *excess* kurtosis, so a normal
distribution scores 0 and a flat one scores negative.
``scipy.stats.kurtosis`` defaults to ``bias=True`` and returns different
numbers, so the two are not interchangeable; this module documents which it
uses because a reader comparing against another tool will need to know.

**The PCA standardises first** (z-score each property, then PCA). Without it,
molecular weight's variance - hundreds of daltons squared against single digits
for hydrogen-bond counts - would swamp everything and PC1 would be molecular
weight alone.

Principal-component signs are arbitrary, so fix them: flip each component so
its largest-magnitude loading is positive. Otherwise two runs, or two scikit-
learn versions, can print mirror-image tables.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from sarscope.analysis.descriptors import CORE_DESCRIPTORS


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
    columns: Sequence[str] = CORE_DESCRIPTORS,
    group_col: str = "group",
) -> GroupProfile:
    """Per-group descriptor statistics. Exactly two groups, else ValueError.

    p-values from ``scipy.stats.mannwhitneyu(g1, g2, alternative="two-sided")``.
    A property that is constant across both groups has no defined test; report
    NaN for it rather than raising.
    """
    groups = sorted(table[group_col].unique())
    if len(groups) != 2:
        raise ValueError(f"need exactly two groups in {group_col!r}, found {groups}")

    rows = {}
    p_values = {}
    for prop in columns:
        samples = [table.loc[table[group_col] == g, prop].astype(float) for g in groups]
        for g, values in zip(groups, samples, strict=True):
            rows[(prop, g)] = {
                "n": len(values),
                "max": values.max(),
                "min": values.min(),
                "mean": values.mean(),
                "median": values.median(),
                "skew": values.skew(),
                "kurtosis": values.kurt(),
            }
        pooled = pd.concat(samples)
        if pooled.nunique() < 2:
            p_values[prop] = np.nan
        else:
            p_values[prop] = float(
                mannwhitneyu(samples[0], samples[1], alternative="two-sided").pvalue
            )

    stats = pd.DataFrame.from_dict(rows, orient="index")
    stats.index = pd.MultiIndex.from_tuples(stats.index, names=["property", "group"])
    return GroupProfile(stats=stats, p_values=pd.Series(p_values, name="p_value"))


def property_pca(
    table: pd.DataFrame,
    columns: Sequence[str] = CORE_DESCRIPTORS,
    n_components: int = 3,
) -> PcaResult:
    """Loadings and scores. ``n_components`` is capped at len(columns)."""
    k = min(n_components, len(columns))
    X = StandardScaler().fit_transform(table[list(columns)].astype(float).to_numpy())
    pca = PCA(n_components=k).fit(X)

    components = pca.components_.copy()
    for i, row in enumerate(components):
        if row[np.argmax(np.abs(row))] < 0:
            components[i] = -row
    names = [f"PC{i + 1}" for i in range(k)]
    # Scores from the sign-fixed components, so plot and table always agree.
    scores = (X - pca.mean_) @ components.T
    return PcaResult(
        scores=pd.DataFrame(scores, index=table.index, columns=names),
        loadings=pd.DataFrame(components.T, index=list(columns), columns=names),
        explained=pd.Series(pca.explained_variance_ratio_, index=names),
    )

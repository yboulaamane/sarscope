"""Explicit molecule selection for the optional browser ML stage."""

from __future__ import annotations

import pandas as pd


def select_ml_table(table: pd.DataFrame, *, drop_intermediate: bool = False) -> pd.DataFrame:
    """Return an ML-only copy; chemical-space/SAR curation stays unchanged."""
    if "activity_class" not in table:
        raise ValueError("ML table has no activity_class column")
    selected = table.loc[table["activity_class"] != "intermediate"] if drop_intermediate else table
    if selected.empty:
        raise ValueError("dropping intermediate leaves no compounds for ML")
    return selected.reset_index(drop=True).copy()

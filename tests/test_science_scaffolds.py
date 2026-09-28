"""Specification for sarscope.analysis.scaffolds."""

import numpy as np
import pandas as pd
import pytest

from sarscope.analysis.scaffolds import (
    DIVERSITY_COLUMNS,
    ENRICHMENT_COLUMNS,
    add_scaffolds,
    diversity_table,
    enrichment_table,
)

pytestmark = pytest.mark.science

Z95 = 1.959963984540054


def test_add_scaffolds():
    table = pd.DataFrame({"smiles": ["c1ccccc1CC", "CCCCO", "c1ccncc1C"]})
    out = add_scaffolds(table)
    assert out["murcko"].tolist() == ["c1ccccc1", None, "c1ccncc1"]
    assert out["skeleton"].tolist() == ["C1CCCCC1", None, "C1CCCCC1"]
    assert "murcko" not in table.columns


def test_diversity_table_by_hand():
    table = pd.DataFrame(
        {
            "activity_class": ["potent"] * 4 + ["inactive"] * 3,
            "murcko": ["A", "A", "B", None, "A", "C", "D"],
            "skeleton": ["S1", "S1", "S1", None, "S1", "S2", "S2"],
        }
    )
    out = diversity_table(table, classes=("potent", "inactive"))
    assert list(out.columns) == list(DIVERSITY_COLUMNS)
    assert list(out.index) == ["complete", "potent", "inactive"]

    complete = out.loc["complete"]
    assert (complete["N"], complete["acyclic"], complete["Ns"]) == (7, 1, 4)
    assert (complete["Nss"], complete["Ncsk"]) == (3, 2)  # B, C, D singletons
    assert complete["Ns/N"] == pytest.approx(4 / 7)
    assert complete["Nss/Ns"] == pytest.approx(3 / 4)

    potent = out.loc["potent"]
    assert (potent["N"], potent["Ns"], potent["Nss"], potent["Ncsk"]) == (4, 2, 1, 1)
    inactive = out.loc["inactive"]
    # "A" has one inactive member, so it is a singleton *within* this class.
    assert (inactive["Ns"], inactive["Nss"]) == (3, 3)


def braf_like(n_all_active: int, singleton: bool = True) -> pd.DataFrame:
    """N = 3952 with A = 2298 Group 1 molecules, like the paper's BRAF set."""
    groups = [1] * 2298 + [2] * 1654
    scaffolds = [f"filler{i % 400}" for i in range(3952)]
    for i in range(n_all_active):
        scaffolds[i] = "SERIES"
    if singleton:
        scaffolds[n_all_active] = "LONE"
    return pd.DataFrame(
        {"murcko": scaffolds, "group": groups, "pactivity": np.linspace(9, 5, 3952)}
    )


def test_all_group1_scaffold_reaches_the_papers_maximum_ef():
    table = enrichment_table(braf_like(30)).set_index("scaffold")
    assert table.loc["SERIES", "ef"] == pytest.approx(3952 / 2298)
    assert round(table.loc["SERIES", "ef"], 3) == pytest.approx(1.720, abs=1e-3)  # paper: 1.719


def test_wilson_lower_bound_separates_a_series_from_one_lucky_compound():
    table = enrichment_table(braf_like(30)).set_index("scaffold")
    base = 2298 / 3952
    # For a = n, the Wilson lower bound reduces to 1 / (1 + z^2 / n).
    assert table.loc["SERIES", "ef_lower"] == pytest.approx(1 / (1 + Z95**2 / 30) / base)
    assert table.loc["LONE", "ef_lower"] == pytest.approx(1 / (1 + Z95**2 / 1) / base)
    assert table.loc["LONE", "ef"] == table.loc["SERIES", "ef"]


def test_columns_and_deterministic_order():
    table = enrichment_table(braf_like(30))
    assert list(table.columns) == list(ENRICHMENT_COLUMNS)
    assert table.iloc[0]["scaffold"] == "SERIES"
    keys = list(zip(-table["ef_lower"], -table["n"], table["scaffold"], strict=True))
    assert keys == sorted(keys)


def test_min_size():
    table = enrichment_table(braf_like(30), min_size=2)
    assert "LONE" not in set(table["scaffold"])


def test_acyclic_molecules_count_in_the_baseline_but_get_no_row():
    table = pd.DataFrame(
        {"murcko": ["A", "A", None, None], "group": [1, 1, 2, 2], "pactivity": [8, 8, 5, 5]}
    )
    out = enrichment_table(table).set_index("scaffold")
    assert list(out.index) == ["A"]
    assert out.loc["A", "ef"] == pytest.approx(1.0 / (2 / 4))


def test_no_group1_is_an_error():
    table = pd.DataFrame({"murcko": ["A", "B"], "group": [2, 2], "pactivity": [5, 5]})
    with pytest.raises(ValueError):
        enrichment_table(table)

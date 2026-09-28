"""Specification for sarscope.analysis.rgroups."""

import pandas as pd
import pytest

from sarscope.analysis.rgroups import (
    SUBSTITUENT_COLUMNS,
    decompose_scaffold,
    decompose_top_scaffolds,
)
from sarscope.analysis.scaffolds import add_scaffolds, enrichment_table

pytestmark = pytest.mark.science

SCAFFOLD = "c1ccc(-c2ccccc2)cc1"


def series() -> pd.DataFrame:
    """Twelve biphenyls: a fluorine series and a methoxy series on one ring.

    Built so the answer is known by hand. At the substituted position, the
    six F-bearing molecules have potencies 8.0-8.5 (median 8.25) and the six
    others 6.0-6.5 (median 6.25), so fluorine's delta there is +2.0.
    """
    rows = []
    for i, (sub, pact) in enumerate(
        [
            ("F", 8.0), ("F", 8.1), ("F", 8.2), ("F", 8.3), ("F", 8.4), ("F", 8.5),
            ("OC", 6.0), ("OC", 6.1), ("OC", 6.2), ("C", 6.3), ("C", 6.4), ("C", 6.5),
        ]
    ):  # fmt: skip
        rows.append(
            {
                "molecule_id": f"M{i}",
                "smiles": f"{sub}c1ccc(-c2ccccc2)cc1",
                "pactivity": pact,
                "activity_class": "potent" if pact >= 8 else "intermediate",
                "group": 1 if pact >= 7 else 2,
            }
        )
    return add_scaffolds(pd.DataFrame(rows))


def test_decomposes_every_member():
    sar = decompose_scaffold(series(), SCAFFOLD)
    assert sar is not None
    assert (sar.n_molecules, sar.n_unmatched) == (12, 0)
    assert sar.scaffold == SCAFFOLD
    assert "*" in sar.core  # the core carries attachment points
    assert len(sar.members) == 12
    assert set(sar.members["molecule_id"]) == {f"M{i}" for i in range(12)}


def test_members_carry_one_column_per_position():
    sar = decompose_scaffold(series(), SCAFFOLD)
    assert sar.positions  # at least one varying position
    assert set(sar.members.columns) >= {"molecule_id", "pactivity", "activity_class"}
    for position in sar.positions:
        assert sar.members[position].notna().all()


def test_the_known_fluorine_delta():
    sar = decompose_scaffold(series(), SCAFFOLD)
    subs = sar.substituents
    assert list(subs.columns) == list(SUBSTITUENT_COLUMNS)

    fluorine = subs[subs["substituent"].str.startswith("F")]
    assert len(fluorine) == 1
    row = fluorine.iloc[0]
    assert row["n"] == 6
    assert row["median_pactivity"] == pytest.approx(8.25)
    assert row["median_other"] == pytest.approx(6.25)
    assert row["delta"] == pytest.approx(2.0)
    assert row["min_pactivity"] == pytest.approx(8.0)
    assert row["max_pactivity"] == pytest.approx(8.5)
    assert row["p_value"] < 0.01


def test_hydrogen_is_kept_and_labelled():
    # Unsubstituted biphenyl alongside the series: its substituent is H.
    table = pd.concat(
        [
            series(),
            add_scaffolds(
                pd.DataFrame(
                    [
                        {
                            "molecule_id": f"H{i}",
                            "smiles": "c1ccc(-c2ccccc2)cc1",
                            "pactivity": 5.0 + i / 10,
                            "activity_class": "inactive",
                            "group": 2,
                        }
                        for i in range(3)
                    ]
                )
            ),
        ],
        ignore_index=True,
    )
    sar = decompose_scaffold(table, SCAFFOLD)
    assert "H" in set(sar.substituents["substituent"])
    assert "H" in set(sar.members[sar.positions[0]])


def test_rare_substituents_are_dropped_from_statistics_but_not_members():
    table = pd.concat(
        [
            series(),
            add_scaffolds(
                pd.DataFrame(
                    [
                        {
                            "molecule_id": "RARE",
                            "smiles": "Clc1ccc(-c2ccccc2)cc1",
                            "pactivity": 9.9,
                            "activity_class": "potent",
                            "group": 1,
                        }
                    ]
                )
            ),
        ],
        ignore_index=True,
    )
    sar = decompose_scaffold(table, SCAFFOLD, min_substituent_count=2)
    assert not sar.substituents["substituent"].str.startswith("Cl").any()
    assert "RARE" in set(sar.members["molecule_id"])
    assert len(sar.members) == 13


def test_invariant_positions_are_dropped():
    sar = decompose_scaffold(series(), SCAFFOLD)
    for position in sar.positions:
        assert sar.members[position].nunique() > 1


def test_ordering_is_by_position_then_effect():
    sar = decompose_scaffold(series(), SCAFFOLD)
    subs = sar.substituents
    assert list(subs["position"]) == sorted(subs["position"])
    for _, block in subs.groupby("position"):
        assert list(block["delta"]) == sorted(block["delta"], reverse=True)


def test_unparseable_or_absent_scaffold_returns_none():
    assert decompose_scaffold(series(), "not a smiles") is None
    assert decompose_scaffold(series(), "c1ccncc1") is None


def test_top_scaffolds_prefers_populated_series():
    table = pd.concat(
        [
            series(),
            add_scaffolds(
                pd.DataFrame(
                    [
                        {
                            "molecule_id": f"T{i}",
                            "smiles": f"{s}c1ccncc1",
                            "pactivity": 9.0,
                            "activity_class": "potent",
                            "group": 1,
                        }
                        for i, s in enumerate(["F", "Cl", "C"])
                    ]
                )
            ),
        ],
        ignore_index=True,
    )
    # The pyridine scaffold is all-potent (highest enrichment) but has 3 members.
    enrichment = enrichment_table(table)
    results = decompose_top_scaffolds(table, enrichment, min_members=8)
    assert [r.scaffold for r in results] == [SCAFFOLD]


def test_top_scaffolds_respects_max():
    table = series()
    results = decompose_top_scaffolds(table, enrichment_table(table), max_scaffolds=0)
    assert results == []

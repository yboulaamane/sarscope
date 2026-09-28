"""Specification for sarscope.analysis.landscape."""

import pandas as pd
import pytest
from rdkit import DataStructs

from sarscope.analysis.landscape import (
    PAIR_COLUMNS,
    REGIONS,
    SasResult,
    cliff_generators,
    consensus,
    sas_map,
)
from sarscope.params import LandscapeParams

pytestmark = pytest.mark.science


def fp(bits: range | list[int]) -> DataStructs.ExplicitBitVect:
    v = DataStructs.ExplicitBitVect(128)
    for b in bits:
        v.SetBit(b)
    return v


@pytest.fixture
def four() -> SasResult:
    """Tanimoto: A-B 10/11, A-C 1.0, B-C 10/11, D shares nothing.

    A=9, B=6, C=5, D=9, so the six pairs are:
      A-B cliff (0.909, 3)     A-C identical      A-D scaffold_hop (0, 0)
      B-C smooth (0.909, 1)    B-D nondescript (0, 3)   C-D nondescript (0, 4)
    """
    fps = [fp(range(10)), fp(range(11)), fp(range(10)), fp(range(50, 60))]
    return sas_map(fps, [9.0, 6.0, 5.0, 9.0], ["m10", "m2", "m3", "m4"], LandscapeParams())


def test_regions(four):
    assert set(four.region_counts) == set(REGIONS)
    assert four.region_counts == {"cliff": 1, "smooth": 1, "scaffold_hop": 1, "nondescript": 2}


def test_cliff_pair_and_sali(four):
    assert list(four.cliffs.columns) == list(PAIR_COLUMNS)
    row = four.cliffs.iloc[0]
    assert (row["id_a"], row["id_b"]) == ("m2", "m10")  # (len, str) order
    assert row["similarity"] == pytest.approx(10 / 11)
    assert row["delta"] == pytest.approx(3.0)
    assert row["sali"] == pytest.approx(3.0 / (1 - 10 / 11))


def test_identical_fingerprints_are_set_aside_not_dropped(four):
    pairs = four.identical_pairs
    assert len(pairs) == 1
    assert (pairs.iloc[0]["id_a"], pairs.iloc[0]["id_b"]) == ("m3", "m10")
    assert pd.isna(pairs.iloc[0]["sali"])
    assert sum(four.region_counts.values()) == 5  # 6 pairs minus the identical one


def test_thresholds_are_strict():
    # Tanimoto exactly 0.9 (9/10) and delta exactly 2.0: not a cliff.
    fps = [fp(range(9)), fp(range(10))]
    result = sas_map(fps, [8.0, 6.0], ["a", "b"], LandscapeParams())
    assert result.region_counts["cliff"] == 0
    assert result.region_counts["scaffold_hop"] == 1


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError):
        sas_map([fp(range(3))], [1.0, 2.0], ["a"], LandscapeParams())


def cliffs_from(pairs: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame([(a, b, 0.95, 3.0, 60.0) for a, b in pairs], columns=list(PAIR_COLUMNS))


def test_generators_use_molecules_in_cliffs_as_the_population():
    hub_pairs = [("hub", f"x{i}") for i in range(10)]
    out = cliff_generators(cliffs_from(hub_pairs), n_sd=2.0)
    counts = pd.Series([10] + [1] * 10)
    assert out.attrs["threshold"] == pytest.approx(counts.mean() + 2 * counts.std())
    assert out.iloc[0]["molecule_id"] == "hub"
    assert out.set_index("molecule_id")["is_generator"].to_dict()["hub"]
    assert out["is_generator"].sum() == 1
    assert len(out) == 11


def test_consensus_intersects_pairs_and_generators():
    shared = [("hub", f"x{i}") for i in range(10)]
    a = SasResult("maccs", {}, cliffs_from([*shared, ("p", "q")]), cliffs_from([]))
    b = SasResult("pubchem", {}, cliffs_from([*shared, ("r", "s")]), cliffs_from([]))
    pairs, generators = consensus({"maccs": a, "pubchem": b})
    assert len(pairs) == 10
    assert ("p", "q") not in set(zip(pairs["id_a"], pairs["id_b"], strict=True))
    assert generators == ["hub"]


def test_consensus_of_nothing_is_an_error():
    with pytest.raises(ValueError):
        consensus({})

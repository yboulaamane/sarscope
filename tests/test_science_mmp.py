import pandas as pd
import pytest

from sarscope.analysis.mmp import matched_molecular_pairs

pytestmark = pytest.mark.science


def test_single_cut_mmp_finds_a_transformation_without_a_scaffold_input():
    table = pd.DataFrame(
        {
            "molecule_id": ["methyl", "ethyl", "methoxy"],
            "smiles": ["Cc1ccccc1", "CCc1ccccc1", "COc1ccccc1"],
            "pactivity": [6.0, 7.0, 8.0],
        }
    )
    pairs = matched_molecular_pairs(table)
    assert not pairs.empty
    assert {"context", "fragment_a", "fragment_b", "delta_pactivity_b_minus_a"} <= set(pairs)
    assert pairs["delta_pactivity_b_minus_a"].abs().max() == pytest.approx(2.0)

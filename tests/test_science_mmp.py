import pandas as pd
import pytest

from sarscope.analysis.mmp import matched_molecular_pairs, summarise_transformations

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


def test_directional_transformations_expose_support_conflicts_and_assay_context():
    pairs = pd.DataFrame(
        [
            {
                "id_a": "a",
                "id_b": "b",
                "context": "core1",
                "fragment_a": "*C",
                "fragment_b": "*O",
                "delta_pactivity_b_minus_a": 1.0,
            },
            {
                "id_a": "c",
                "id_b": "d",
                "context": "core2",
                "fragment_a": "*O",
                "fragment_b": "*C",
                "delta_pactivity_b_minus_a": 0.5,
            },
            {
                "id_a": "e",
                "id_b": "f",
                "context": "core2",
                "fragment_a": "*C",
                "fragment_b": "*O",
                "delta_pactivity_b_minus_a": -0.2,
            },
        ]
    )
    evidence = pd.DataFrame(
        {
            "molecule_id": list("abcdef"),
            "assay_chembl_id": ["X"] * 6,
        }
    )
    result = summarise_transformations(pairs, evidence)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["assay_context"] == "shared assay"
    assert row["distinct_contexts"] == 2
    assert row["supporting_pairs"] == 3
    assert row["positive_pairs"] == 1
    assert row["negative_pairs"] == 2

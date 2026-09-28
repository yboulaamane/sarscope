import pandas as pd
import pytest

from sarscope.compare import compare_tables


def test_compare_tables_reports_compound_ratios_and_scaffold_preference():
    a = pd.DataFrame(
        {
            "structure_key": ["x", "y"],
            "molecule_id_a": ["a1", "a2"],
            "smiles": ["Cc1ccccc1", "CCc1ccccc1"],
            "murcko": ["c1ccccc1", "c1ccccc1"],
            "pactivity_a": [8.0, 7.0],
            "activity_class_a": ["potent", "active"],
            "group_a": [1, 1],
        }
    )
    b = pd.DataFrame(
        {
            "structure_key": ["x", "z"],
            "molecule_id_b": ["b1", "b2"],
            "pactivity_b": [6.0, 5.0],
            "activity_class_b": ["intermediate", "inactive"],
            "group_b": [2, 2],
            "smiles": ["Cc1ccccc1", "Clc1ccccc1"],
            "murcko": ["c1ccccc1", "c1ccccc1"],
        }
    )
    compounds, scaffolds = compare_tables(a, b)
    assert len(compounds) == 1
    assert compounds.iloc[0]["selectivity_ratio_a_over_b"] == pytest.approx(100.0)
    assert compounds.iloc[0]["preferred_target"] == "A"
    assert len(scaffolds) == 1
    assert scaffolds.iloc[0]["active_fraction_delta_a_minus_b"] == pytest.approx(1.0)

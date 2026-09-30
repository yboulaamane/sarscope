import pandas as pd
import pytest

from sarscope.compare import compare_tables, run_compare
from sarscope.params import CurationParams, RunParams


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


def test_comparison_exposes_missingness_and_assay_quality(make_record):
    def record(activity_id, molecule_id, smiles, value, assay_id):
        return make_record(
            activity_id=activity_id,
            molecule_chembl_id=molecule_id,
            parent_molecule_chembl_id=molecule_id,
            canonical_smiles=smiles,
            standard_value=str(value),
            assay_chembl_id=assay_id,
            document_chembl_id="CHEMBL_DOC",
            confidence_score=9,
        )

    class Client:
        release = "ChEMBL_test"

        def target(self, target_id):
            return {
                "target_chembl_id": target_id,
                "pref_name": target_id,
                "organism": "Homo sapiens",
            }

        def activities(self, target_id, standard_types):
            if target_id == "CHEMBL1":
                return [
                    record(1, "M1", "Cc1ccccc1", 10, "CHEMBL_A"),
                    record(2, "M2", "Clc1ccccc1", 100, "CHEMBL_A"),
                ]
            return [
                record(3, "M1", "Cc1ccccc1", 1000, "CHEMBL_B"),
                record(4, "M3", "Fc1ccccc1", 100, "CHEMBL_B"),
            ]

    result = run_compare("CHEMBL1", "CHEMBL2", RunParams(), Client())
    assert len(result.compounds) == 1
    assert result.compounds.iloc[0]["assay_context"] == "same endpoint, format and document"
    coverage = result.activity_matrix.set_index("coverage")
    assert set(coverage.index) == {"both", "A only", "B only"}
    assert pd.isna(coverage.loc["A only", "pactivity_b"])
    assert coverage.loc["B only", "smiles"] == "Fc1ccccc1"

    narrowed = run_compare(
        "CHEMBL1",
        "CHEMBL2",
        RunParams(curation=CurationParams(assay_ids=("CHEMBL_A",))),
        Client(),
        secondary_assay_ids=("CHEMBL_B",),
    )
    assert len(narrowed.compounds) == 1
    assert narrowed.provenance["secondary_assay_ids"] == ("CHEMBL_B",)

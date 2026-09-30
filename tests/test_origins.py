"""ChEMBL integration origins are exposed without pretending to be new data."""

import pandas as pd

from sarscope.sources.origins import source_disagreement, source_name, source_summary


def test_source_names_are_conservative():
    assert source_name("7") == "PubChem BioAssay (integrated)"
    assert source_name(37) == "BindingDB (integrated)"
    assert source_name(None) == "Unknown ChEMBL source"
    assert source_name(999) == "ChEMBL source 999"


def test_summary_counts_cross_source_molecules_once_per_origin():
    evidence = pd.DataFrame(
        [
            {"record_id": "1", "molecule_id": "A", "src_id": 1, "assay_chembl_id": "X"},
            {"record_id": "2", "molecule_id": "A", "src_id": 7, "assay_chembl_id": "Y"},
            {"record_id": "3", "molecule_id": "B", "src_id": 7, "assay_chembl_id": "Y"},
            {"record_id": "4", "molecule_id": "B", "src_id": 7, "assay_chembl_id": "Y"},
        ]
    )
    summary = source_summary(evidence).set_index("src_id")
    assert summary.loc[1, "records"] == 1
    assert summary.loc[7, "records"] == 3
    assert summary.loc[7, "molecules"] == 2
    assert summary.loc[7, "shared_molecules"] == 1


def test_disagreement_uses_source_specific_medians():
    evidence = pd.DataFrame(
        [
            {"molecule_id": "A", "src_id": 1, "pactivity": 6.0, "standard_type": "IC50"},
            {"molecule_id": "A", "src_id": 1, "pactivity": 7.0, "standard_type": "IC50"},
            {"molecule_id": "A", "src_id": 37, "pactivity": 8.0, "standard_type": "Ki"},
            {"molecule_id": "B", "src_id": 7, "pactivity": 5.0, "standard_type": "IC50"},
        ]
    )
    result = source_disagreement(evidence)
    assert result["molecule_id"].tolist() == ["A"]
    assert result.iloc[0]["median_pactivity_spread"] == 1.5
    assert result.iloc[0]["endpoint_types"] == "IC50; Ki"

"""Shared fixtures. No test here touches the network unless marked ``network``."""

from __future__ import annotations

from typing import Any

import pytest


def chembl_record(**overrides: Any) -> dict[str, Any]:
    """A synthetic ChEMBL activity record with the API's real field types.

    Defaults describe a record every default filter keeps. Override fields to
    exercise one filter at a time. (Synthetic rather than real ChEMBL rows,
    because ChEMBL data is CC BY-SA and this repository is MIT.)
    """
    record: dict[str, Any] = {
        "activity_id": 1,
        "molecule_chembl_id": "CHEMBL1",
        "parent_molecule_chembl_id": "CHEMBL1",
        "canonical_smiles": "c1ccc2[nH]ccc2c1",
        "standard_type": "IC50",
        "standard_relation": "=",
        "standard_value": "100.0",  # a string, as the API returns it
        "standard_units": "nM",
        "pchembl_value": "7.00",
        "assay_type": "B",
        "assay_variant_mutation": None,
        "potential_duplicate": 0,
        "data_validity_comment": None,
        "bao_label": "single protein format",
        "document_year": 2015,
        "src_id": 1,
    }
    record.update(overrides)
    return record


@pytest.fixture
def make_record() -> Any:
    return chembl_record

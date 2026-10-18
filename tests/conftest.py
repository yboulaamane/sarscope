"""Shared fixtures. No test here touches the network unless marked ``network``."""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def restore_main_metadata() -> Iterator[None]:
    """Keep Streamlit AppTest's temporary main script out of later spawn tests.

    AppTest may leave __main__.__file__ pointing at an unguarded UI fragment.
    A subsequent multiprocessing spawn then executes that fragment in the child,
    without its session fixture. Restore test-process metadata, not worker logic.
    """
    module = sys.modules.get("__main__")
    absent = object()
    names = ("__file__", "__spec__", "__package__", "__loader__")
    original = {name: getattr(module, name, absent) for name in names}
    yield
    if module is not None:
        sys.modules["__main__"] = module
        for name, value in original.items():
            if value is absent:
                module.__dict__.pop(name, None)
            else:
                setattr(module, name, value)


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

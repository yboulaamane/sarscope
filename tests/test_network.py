"""Live ChEMBL checks. Deselected in CI; run with ``pytest -m network``."""

import pytest

from sarscope.sources.chembl import ChemblClient

pytestmark = pytest.mark.network


def test_braf_resolves():
    with ChemblClient() as client:
        target = client.target("CHEMBL5145")
    assert target["pref_name"] == "Serine/threonine-protein kinase B-raf"


def test_the_paper_target_id_is_not_braf():
    # The reference paper cites "target ID: 5651". That is STK35.
    with ChemblClient() as client:
        target = client.target("5651")
        n = client.count_activities("5651")
    assert "B-raf" not in target["pref_name"]
    assert n < 100

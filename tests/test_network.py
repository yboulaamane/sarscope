"""Live ChEMBL checks. Deselected in CI; run with ``pytest -m network``."""

import pytest

from sarscope.sources.chembl import ChemblClient

pytestmark = pytest.mark.network


def test_braf_resolves():
    with ChemblClient() as client:
        target = client.target("CHEMBL5145")
    assert target["pref_name"] == "Serine/threonine-protein kinase B-raf"


def test_a_near_miss_target_id_resolves_to_a_different_protein():
    # CHEMBL5651 is STK35, not BRAF. A transposed digit fetches another protein
    # rather than failing, which is why the CLI and app always print the name.
    with ChemblClient() as client:
        target = client.target("5651")
        n = client.count_activities("5651")
    assert "B-raf" not in target["pref_name"]
    assert n < 100

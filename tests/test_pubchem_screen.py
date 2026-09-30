"""Offline tests for the categorical PubChem workflow."""

import hashlib
import json

import httpx
import pytest

from sarscope.screen import curate_outcomes, read_screen_snapshot
from sarscope.sources.pubchem import PubChemClient, PubChemError


def _table(rows):
    columns = ["AID", "CID", "Activity Outcome", "Assay Name", "Assay Type"]
    return {"Table": {"Columns": {"Column": columns}, "Row": [{"Cell": row} for row in rows]}}


def test_pubchem_client_reads_outcomes_and_structures_without_potency():
    def respond(request):
        if "concise" in str(request.url):
            return httpx.Response(200, json=_table([["100", "12", "Active", "Test", "Screening"]]))
        return httpx.Response(
            200, json={"PropertyTable": {"Properties": [{"CID": 12, "SMILES": "CCO"}]}}
        )

    with PubChemClient(transport=httpx.MockTransport(respond)) as client:
        rows, meta = client.concise_assay(100)
        smiles = client.smiles_for_cids([12])
    table, audit = curate_outcomes(rows, smiles)
    assert meta["aid"] == "100"
    assert table.iloc[0]["outcome"] == "active"
    assert "pactivity" not in table.columns
    assert "Activity Value [uM]" not in table.columns
    assert audit["curated_compounds"] == 1


def test_ambiguous_and_contradictory_outcomes_are_excluded():
    rows = [
        {"CID": "1", "Activity Outcome": "Active"},
        {"CID": "2", "Activity Outcome": "Inactive"},
        {"CID": "3", "Activity Outcome": "Inconclusive"},
        {"CID": "4", "Activity Outcome": "Inactive"},
    ]
    table, audit = curate_outcomes(rows, {1: "CCO", 2: "CCO", 3: "CCC", 4: "CCN"})
    assert table["outcome"].tolist() == ["inactive"]
    assert audit["conflicting_structures"] == 1
    assert audit["nonbinary_outcome"] == 1


def test_assay_at_limit_is_rejected_not_silently_truncated():
    def respond(request):
        return httpx.Response(
            200, json=_table([["100", "12", "Active", "Test", "Screening"]] * 10_000)
        )

    with PubChemClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(PubChemError, match="safety limit"):
            client.concise_assay(100)


def test_wrong_aid_is_rejected():
    def respond(request):
        return httpx.Response(200, json=_table([["101", "12", "Active", "Test", "Screening"]]))

    with PubChemClient(transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(PubChemError, match="different AID"):
            client.concise_assay(100)


def test_offline_snapshot_verifies_calls_and_structures(tmp_path):
    rows = [{"AID": "100", "CID": "12", "Activity Outcome": "Active"}]
    smiles = {12: "CCO"}
    digest = lambda value: hashlib.sha256(  # noqa: E731 - compact test fixture
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    payload = {
        "assay": {
            "aid": "100",
            "raw_rows_sha256": digest(rows),
            "source_smiles_sha256": digest(smiles),
        },
        "rows": rows,
        "smiles_by_cid": smiles,
    }
    path = tmp_path / "source_snapshot.json"
    path.write_text(json.dumps(payload))
    aid, read_rows, read_smiles, _ = read_screen_snapshot(path)
    assert (aid, read_rows, read_smiles) == (100, rows, smiles)
    payload["smiles_by_cid"][12] = "CCC"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="structure snapshot hash mismatch"):
        read_screen_snapshot(path)

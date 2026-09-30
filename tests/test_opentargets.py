import json

import httpx
import pytest

from sarscope.sources.opentargets import OpenTargetsError, disease_targets, search_diseases


def test_disease_search_is_bounded_and_does_not_choose_a_term():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["variables"] == {"query": "melanoma", "size": 10}
        assert 'entityNames: ["disease"]' in payload["query"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "search": {
                        "total": 2,
                        "hits": [
                            {"id": "MONDO_0005105", "name": "melanoma", "entity": "disease"},
                            {"id": "MONDO_0006486", "name": "uveal melanoma", "entity": "disease"},
                        ],
                    }
                }
            },
        )

    result = search_diseases(" melanoma ", transport=httpx.MockTransport(handler))
    assert len(result["hits"]) == result["total"] == 2


def test_associations_record_direct_evidence_and_only_reviewed_accessions():
    def handler(request):
        payload = json.loads(request.content)
        assert payload["variables"] == {"id": "MONDO_0005105", "size": 25}
        assert "enableIndirect: false" in payload["query"]
        return httpx.Response(
            200,
            json={
                "data": {
                    "disease": {
                        "id": "MONDO_0005105",
                        "name": "melanoma",
                        "associatedTargets": {
                            "count": 100,
                            "rows": [
                                {
                                    "score": 0.8,
                                    "target": {
                                        "id": "ENSG00000157764",
                                        "approvedSymbol": "BRAF",
                                        "approvedName": "B-raf",
                                        "proteinIds": [
                                            {"id": "P15056", "source": "uniprot_swissprot"},
                                            {"id": "P15056", "source": "uniprot_swissprot"},
                                            {"id": "OLD", "source": "uniprot_obsolete"},
                                            {"id": "ENSP1", "source": "ensembl_PRO"},
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                }
            },
        )

    result = disease_targets("MONDO_0005105", transport=httpx.MockTransport(handler))
    assert result["rows"][0]["accessions"] == ["P15056"]
    assert result["rows"][0]["score"] == 0.8
    assert result["total"] == 100
    assert result["enable_indirect"] is False
    assert result["retrieved_utc"]


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(503),
        httpx.Response(200, json={"errors": [{"message": "schema changed"}], "data": {}}),
        httpx.Response(200, json={"data": None}),
        httpx.Response(200, text="bad JSON"),
        httpx.Response(200, json={"data": {"search": {"hits": [], "total": None}}}),
    ],
)
def test_upstream_errors_are_explained(response):
    with pytest.raises(OpenTargetsError):
        search_diseases("melanoma", transport=httpx.MockTransport(lambda _: response))


def test_timeout_is_explained():
    def handler(request):
        raise httpx.ReadTimeout("timeout", request=request)

    with pytest.raises(OpenTargetsError, match="unavailable"):
        search_diseases("melanoma", transport=httpx.MockTransport(handler))


def test_retired_disease_is_not_an_empty_success():
    transport = httpx.MockTransport(lambda _: httpx.Response(200, json={"data": {"disease": None}}))
    with pytest.raises(OpenTargetsError, match="no longer available"):
        disease_targets("OLD_1", transport=transport)


@pytest.mark.parametrize("query", ["", " ", "A" * 201])
def test_invalid_search_does_not_reach_api(query):
    def handler(_):
        pytest.fail("Invalid search reached the network")

    with pytest.raises(ValueError):
        search_diseases(query, transport=httpx.MockTransport(handler))

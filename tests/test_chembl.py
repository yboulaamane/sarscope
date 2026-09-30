import gzip
import json

import httpx
import pytest

from sarscope.sources.chembl import (
    ChemblClient,
    ChemblError,
    TargetNotFoundError,
    normalise_target_id,
)

STATUS = {"chembl_db_version": "ChEMBL_37", "chembl_release_date": "2026-05-01"}
BASE = "/chembl/api/data"


def paged_api(n_records: int, page_size: int, *, total: int | None = None, fail_first: int = 0):
    """A fake ChEMBL: status, one target, and paged activities.

    ``fail_first`` makes the first N requests return 503 to exercise retries.
    Returns (handler, calls) where calls records every requested URL.
    """
    calls: list[str] = []
    failures = {"left": fail_first}
    records = [{"activity_id": i, "molecule_chembl_id": f"CHEMBL{i}"} for i in range(n_records)]

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        if failures["left"] > 0:
            failures["left"] -= 1
            return httpx.Response(503)
        path = request.url.path
        if path == f"{BASE}/status.json":
            return httpx.Response(200, json=STATUS)
        if path == f"{BASE}/target/CHEMBL5145.json":
            return httpx.Response(
                200, json={"target_chembl_id": "CHEMBL5145", "pref_name": "B-raf"}
            )
        if path.startswith(f"{BASE}/target/"):
            return httpx.Response(404)
        if path == f"{BASE}/activity.json":
            # Like the real API, the server caps the page size whatever was asked.
            offset = int(request.url.params.get("offset", 0))
            chunk = records[offset : offset + page_size]
            end = offset + len(chunk)
            nxt = (
                f"{BASE}/activity.json?limit={page_size}&offset={end}" if end < n_records else None
            )
            meta = {"total_count": n_records if total is None else total, "next": nxt}
            return httpx.Response(200, json={"activities": chunk, "page_meta": meta})
        return httpx.Response(404)

    return handler, calls


def client(handler, **kwargs) -> ChemblClient:
    return ChemblClient(transport=httpx.MockTransport(handler), sleep=lambda _: None, **kwargs)


@pytest.mark.parametrize("raw", ["CHEMBL5145", "chembl5145", " 5145 ", "5145"])
def test_normalise_target_id(raw):
    assert normalise_target_id(raw) == "CHEMBL5145"


@pytest.mark.parametrize("raw", ["", "BRAF", "CHEMBL", "CHEMBL12a"])
def test_normalise_rejects_non_ids(raw):
    with pytest.raises(ValueError):
        normalise_target_id(raw)


def test_follows_every_page():
    handler, calls = paged_api(n_records=25, page_size=10)
    records = client(handler).activities("CHEMBL5145")
    assert [r["activity_id"] for r in records] == list(range(25))
    assert sum("activity.json" in c for c in calls) == 3


def test_first_request_carries_the_filters():
    handler, calls = paged_api(n_records=5, page_size=10)
    client(handler).activities("5145", ["Ki", "IC50", "IC50"])
    first = httpx.URL(next(c for c in calls if "activity.json" in c))
    assert first.params["target_chembl_id"] == "CHEMBL5145"
    assert first.params["standard_type__in"] == "IC50,Ki"


def test_count_mismatch_is_an_error():
    handler, _ = paged_api(n_records=5, page_size=10, total=6)
    with pytest.raises(ChemblError, match="expected 6"):
        client(handler).activities("CHEMBL5145")


def test_retries_server_errors():
    handler, calls = paged_api(n_records=3, page_size=10, fail_first=2)
    assert client(handler).status()["chembl_db_version"] == "ChEMBL_37"
    assert len(calls) == 3


def test_gives_up_after_retries():
    handler, _ = paged_api(n_records=3, page_size=10, fail_first=99)
    with pytest.raises(ChemblError, match="503"):
        client(handler, retries=2).status()


def test_transport_errors_are_retried_then_reported():
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        raise httpx.ConnectError("offline", request=request)

    with pytest.raises(ChemblError, match="unreachable"):
        client(handler, retries=1).status()
    assert attempts["n"] == 2


def test_unknown_target():
    handler, _ = paged_api(n_records=0, page_size=10)
    with pytest.raises(TargetNotFoundError, match="CHEMBL9999999"):
        client(handler).target("CHEMBL9999999")


def test_cache_is_written_keyed_by_release_and_reused(tmp_path):
    handler, calls = paged_api(n_records=12, page_size=5)
    first = client(handler, cache_dir=tmp_path).activities("CHEMBL5145")
    cache_file = tmp_path / "ChEMBL_37" / "CHEMBL5145_IC50.json.gz"
    with gzip.open(cache_file, "rt") as fh:
        assert json.load(fh) == first

    n_activity_calls = sum("activity.json" in c for c in calls)
    second = client(handler, cache_dir=tmp_path).activities("CHEMBL5145")
    assert second == first
    assert sum("activity.json" in c for c in calls) == n_activity_calls
    assert not list(tmp_path.rglob("*.tmp"))


def test_activity_progress_reports_each_page_and_cached_load(tmp_path):
    handler, _ = paged_api(n_records=12, page_size=5)
    updates: list[tuple[int, int]] = []
    with client(handler, cache_dir=tmp_path) as chembl:
        chembl.activities("CHEMBL5145", progress=lambda done, total: updates.append((done, total)))
    assert updates == [(5, 12), (10, 12), (12, 12)]

    updates.clear()
    with client(handler, cache_dir=tmp_path) as chembl:
        chembl.activities("CHEMBL5145", progress=lambda done, total: updates.append((done, total)))
    assert updates == [(12, 12)]


def test_count_activities_uses_a_single_row_request():
    handler, calls = paged_api(n_records=42, page_size=10)
    assert client(handler).count_activities("CHEMBL5145") == 42
    assert httpx.URL(calls[-1]).params["limit"] == "1"


def test_assay_confidence_paginates_separately_from_activities():
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == f"{BASE}/assay.json"
        if len(calls) == 1:
            assert request.url.params["only"] == "assay_chembl_id,confidence_score"
            return httpx.Response(
                200,
                json={
                    "assays": [{"assay_chembl_id": "A", "confidence_score": 9}],
                    "page_meta": {"total_count": 2, "next": f"{BASE}/assay.json?offset=1"},
                },
            )
        return httpx.Response(
            200,
            json={
                "assays": [{"assay_chembl_id": "B", "confidence_score": 7}],
                "page_meta": {"total_count": 2, "next": None},
            },
        )

    with client(handler) as c:
        assert c.assay_confidences("5145") == {"A": 9, "B": 7}
    assert len(calls) == 2


def test_name_search_preserves_ambiguous_targets_and_reports_total():
    def handler(request):
        assert request.url.path == f"{BASE}/target/search.json"
        assert request.url.params["q"] == "BRAF"
        assert request.url.params["organism"] == "Homo sapiens"
        assert request.url.params["limit"] == "25"
        return httpx.Response(
            200,
            json={
                "targets": [
                    {
                        "target_chembl_id": "CHEMBL4106189",
                        "pref_name": "BRAF/CRAF",
                        "organism": "Homo sapiens",
                        "target_type": "PROTEIN COMPLEX",
                    },
                    {
                        "target_chembl_id": "CHEMBL5145",
                        "pref_name": "B-raf",
                        "organism": "Homo sapiens",
                        "target_type": "SINGLE PROTEIN",
                    },
                ],
                "page_meta": {"total_count": 40},
            },
        )

    with client(handler) as c:
        rows, total = c.search_targets(" BRAF ")
    assert [row["target_chembl_id"] for row in rows] == ["CHEMBL5145", "CHEMBL4106189"]
    assert total == 40


def test_name_search_all_species_and_empty_result():
    def handler(request):
        assert "organism" not in request.url.params
        return httpx.Response(200, json={"targets": [], "page_meta": {"total_count": 0}})

    with client(handler) as c:
        assert c.search_targets("unknown", organism=None) == ([], 0)


def test_accession_mapping_is_exact_human_single_protein_and_bounded():
    calls = []

    def handler(request):
        calls.append(request)
        assert request.url.path == f"{BASE}/target.json"
        assert request.url.params["target_components__accession__in"] == "P15056,Q02750"
        assert request.url.params["target_type"] == "SINGLE PROTEIN"
        assert request.url.params["organism"] == "Homo sapiens"
        assert "q" not in request.url.params
        return httpx.Response(200, json={"targets": [], "page_meta": {"total_count": 0}})

    with client(handler) as c:
        assert c.targets_for_accessions([]) == ([], 0)
        assert calls == []
        assert c.targets_for_accessions(["P15056", "P15056", "Q02750"]) == ([], 0)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(400, json={"error": "bad query"}),
        httpx.Response(200, json={"targets": None}),
        httpx.Response(200, text="not JSON"),
    ],
)
def test_search_failure_is_actionable(response):
    with client(lambda _: response) as c, pytest.raises(ChemblError, match="search failed"):
        c.search_targets("BRAF")


@pytest.mark.parametrize("query,limit", [("", 25), ("A" * 201, 25), ("BRAF", 100)])
def test_search_rejects_bad_input_without_network(query, limit):
    def handler(_):
        pytest.fail("Invalid search should not reach the network")

    with client(handler) as c, pytest.raises(ValueError):
        c.search_targets(query, limit=limit)

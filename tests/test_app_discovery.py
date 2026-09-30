"""Exercise discovery through the actual app, with both upstream APIs mocked."""

import json
from pathlib import Path

import httpx
import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"
TARGET = {
    "target_chembl_id": "CHEMBL5145",
    "pref_name": "B-raf",
    "organism": "Homo sapiens",
    "target_type": "SINGLE PROTEIN",
}


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def test_curation_stage_reports_load_and_standardization_progress(monkeypatch, make_record):
    import streamlit_app as app_module

    statuses = []
    loaded = []
    standardized = []
    record = make_record()

    def cached_fetch(target_id, types, _progress=None):
        assert target_id == "CHEMBL5145"
        assert types == ("IC50",)
        if _progress is not None:
            _progress(1, 1)
        return [record]

    monkeypatch.setattr(app_module, "fetch", cached_fetch)
    stage = app_module.run_curation_stage(
        "CHEMBL5145",
        ("IC50",),
        app_module.RunParams(),
        "ChEMBL_test",
        "B-raf",
        "Homo sapiens",
        on_status=statuses.append,
        on_download_progress=lambda done, total: loaded.append((done, total)),
        on_standardize_progress=lambda done, total: standardized.append((done, total)),
    )
    assert loaded == [(1, 1)]
    assert standardized[-1] == (1, 1)
    assert stage.timings["load_records"] >= 0
    assert stage.timings["curation"] >= 0
    assert len(stage.table) == 1
    assert any("Loaded 1 raw records" in message for message in statuses)


@pytest.fixture
def app_api(monkeypatch, make_record):
    st.cache_data.clear()
    calls = []

    def send(client, request, **kwargs):
        calls.append(request)
        path = request.url.path
        if path.endswith("/graphql"):
            payload = json.loads(request.content)
            if "DiseaseSearch" in payload["query"]:
                data = {
                    "search": {
                        "total": 1,
                        "hits": [{"id": "MONDO_0005105", "name": "melanoma", "entity": "disease"}],
                    }
                }
            else:
                data = {
                    "disease": {
                        "id": "MONDO_0005105",
                        "name": "melanoma",
                        "associatedTargets": {
                            "count": 1,
                            "rows": [
                                {
                                    "score": 0.8,
                                    "target": {
                                        "id": "ENSG00000157764",
                                        "approvedSymbol": "BRAF",
                                        "approvedName": "B-raf",
                                        "proteinIds": [
                                            {"id": "P15056", "source": "uniprot_swissprot"}
                                        ],
                                    },
                                }
                            ],
                        },
                    }
                }
            return httpx.Response(200, json={"data": data}, request=request)
        if path.endswith("target/search.json") or path.endswith("target.json"):
            if request.url.params.get("q") == "fail":
                return httpx.Response(400, json={"error": "bad search"}, request=request)
            data = {"targets": [TARGET], "page_meta": {"total_count": 1}}
        elif path.endswith("target/CHEMBL5145.json"):
            data = TARGET
        elif path.endswith("status.json"):
            data = {"chembl_db_version": "ChEMBL_test"}
        elif path.endswith("activity.json"):
            data = {"page_meta": {"total_count": 1}}
            if request.url.params.get("limit") != "1":
                data["activities"] = [make_record()]
                data["page_meta"]["next"] = None
        else:
            pytest.fail(f"Unexpected request: {request.url}")
        return httpx.Response(200, json=data, request=request)

    monkeypatch.setattr(httpx.Client, "send", send)
    yield calls
    st.cache_data.clear()


def test_startup_and_mode_changes_make_no_network_requests(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    for mode in ["Gene / protein", "Disease", "ChEMBL ID"]:
        widget(app, "radio", "Find a target by").set_value(mode).run()
        assert not app.exception
    assert not app_api


def test_curation_button_shows_phase_timings(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app, "radio", "Find a target by").set_value("ChEMBL ID").run()
    widget(app, "text_input", "ChEMBL target ID").set_value("CHEMBL5145").run()
    widget(app, "button", "Check target").click().run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    assert not app.exception
    assert any("Activity load:" in item.value for item in app.caption)


def test_pubchem_workflow_is_separate_and_idle_until_requested(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app, "radio", "Workflow").set_value("PubChem qualitative screen").run()
    assert not app.exception
    assert not app_api
    assert not any(button.label == "Check target" for button in app.button)
    assert any(button.label == "Run qualitative screen" for button in app.button)


def test_gene_search_requires_selection_and_invalidates_old_results(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app, "radio", "Find a target by").set_value("Gene / protein").run()
    widget(app, "text_input", "Gene or protein name").set_value("BRAF").run()
    assert not app_api
    widget(app, "button", "Search").click().run()
    assert len(app_api) == 1
    assert widget(app, "button", "Check target").disabled
    widget(app, "selectbox", "Choose a ChEMBL target").select("CHEMBL5145").run()
    assert len(app_api) == 1
    widget(app, "button", "Check target").click().run()
    assert not app.exception
    assert any(item.value == "B-raf · CHEMBL5145" for item in app.subheader)
    count = len(app_api)
    widget(app, "checkbox", "Human targets only").uncheck().run()
    assert not app.subheader
    assert len(app_api) == count
    widget(app, "text_input", "Gene or protein name").set_value("fail").run()
    widget(app, "button", "Search").click().run()
    assert not app.exception
    assert any("search failed" in item.value for item in app.error)
    assert not app.subheader
    assert "target_lookup" not in app.session_state


def test_disease_to_gene_to_exact_chembl_target(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app, "radio", "Find a target by").set_value("Disease").run()
    widget(app, "text_input", "Disease or phenotype").set_value("melanoma").run()
    widget(app, "button", "Search").click().run()
    assert len(app_api) == 1
    widget(app, "selectbox", "Choose the disease term").select("MONDO_0005105").run()
    assert len(app_api) == 1
    widget(app, "button", "Find associated genes").click().run()
    widget(app, "selectbox", "Choose a gene to resolve in ChEMBL").select("ENSG00000157764").run()
    assert len(app_api) == 2
    widget(app, "button", "Find ChEMBL targets for this gene").click().run()
    assert app_api[-1].url.params["target_components__accession__in"] == "P15056"
    widget(app, "selectbox", "Choose a ChEMBL target").select("CHEMBL5145").run()
    widget(app, "button", "Check target").click().run()
    assert not app.exception
    assert any(item.value == "B-raf · CHEMBL5145" for item in app.subheader)
    widget(app, "text_input", "Disease or phenotype").set_value("asthma").run()
    assert not app.subheader

"""Exercise discovery through the actual app, with both upstream APIs mocked."""

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pandas as pd
import pytest

from sarscope.analysis.landscape import SasResult
from sarscope.params import RunParams

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


def test_editable_cutoffs_invalidate_curation_and_show_group_meanings(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app, "button", "Check target").click().run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    assert not app.exception
    assert {"Potent", "Active", "Intermediate", "Inactive"} <= {item.label for item in app.metric}
    assert widget(app, "metric", "Inactive").value == "0"
    assert any("Group 2 = intermediate + inactive" in item.value for item in app.caption)
    old = app.session_state["curation_stage"][1].table["activity_class"].tolist()
    widget(app, "number_input", "Potent minimum pActivity").set_value(9.0).run()
    assert not any(item.label == "Intermediate" for item in app.metric)
    widget(app, "number_input", "Active minimum pActivity").set_value(8.0).run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    assert not app.exception
    stage = app.session_state["curation_stage"][1]
    assert stage.params.classes.bounds[0] == ("potent", 9.0)
    assert stage.table["activity_class"].tolist() != old
    widget(app, "number_input", "Active minimum pActivity").set_value(9.0).run()
    assert any("Class cutoffs must be ordered" in item.value for item in app.error)


def test_variant_filter_is_generic_and_does_not_force_braf_mutation(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    selector = widget(app, "selectbox", "Protein variant")
    assert selector.value == "No mutation annotation"
    assert "V600E" not in selector.options
    selector.set_value("Specific mutation").run()
    assert not app.exception
    assert any("Enter an exact mutation annotation" in item.value for item in app.info)
    widget(app, "text_input", "Exact mutation annotation").set_value("T790M").run()
    assert not app.exception
    widget(app, "button", "Check target").click().run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    assert not app.exception
    assert any("no measurements remain" in item.value for item in app.error)
    widget(app, "selectbox", "Protein variant").set_value("No mutation annotation").run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    assert not app.exception
    assert app.session_state["curation_stage"][1].params.curation.variant is None
    assert any("not confirmation of wild-type" in item.value for item in app.caption)


def test_cliff_browser_reaches_pairs_beyond_thirty_and_filters_ids(app_api):
    app = AppTest.from_file(str(APP), default_timeout=20).run()
    widget(app, "button", "Check target").click().run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    pairs = pd.DataFrame(
        {
            "id_a": ["CHEMBL1"] * 61,
            "id_b": [f"CHEMBL{i + 2}" for i in range(61)],
            "similarity": [0.95] * 61,
            "delta": [4.0 - i / 100 for i in range(61)],
            "sali": [80 - i for i in range(61)],
        }
    )
    sas = SasResult("ecfp4", {"cliff": 61}, pairs, pairs.iloc[:0])
    app.session_state["landscape_stage"] = SimpleNamespace(
        params=RunParams(),
        table=app.session_state["curation_stage"][1].table,
        landscapes={"ecfp4": sas},
        consensus_cliffs=pairs,
        consensus_generators=[],
    )
    app.run()
    assert not app.exception
    widget(app, "number_input", "Cliff-pair page").set_value(3).run()
    assert not app.exception
    options = widget(app, "selectbox", "Cliff pair").options
    assert len(options) == 11
    assert "CHEMBL52" in options[0]
    widget(app, "selectbox", "Cliff pair").select(options[-1]).run()
    assert not app.exception
    widget(app, "text_input", "Filter pairs by molecule ID").set_value("CHEMBL62").run()
    assert not app.exception
    assert len(widget(app, "selectbox", "Cliff pair").options) == 1
    assert widget(app, "number_input", "Cliff-pair page").value == 1
    widget(app, "text_input", "Filter pairs by molecule ID").set_value("no matches").run()
    assert not app.exception
    assert any("No cliff pairs match" in item.value for item in app.info)


def test_umap_is_opt_in_and_controls_mark_existing_projection_stale(app_api, monkeypatch):
    import numpy as np

    from sarscope.analysis import chemical_space

    original = chemical_space.ecfp4_umap
    calls = []

    def traced(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(chemical_space, "ecfp4_umap", traced)
    pytest.importorskip("umap")
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    widget(app, "button", "Check target").click().run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    stage = app.session_state["curation_stage"][1]
    stage.table = pd.DataFrame(
        {
            "molecule_id": [f"m{i}" for i in range(6)],
            "smiles": [
                "Cc1ccccc1",
                "CCc1ccccc1",
                "COc1ccccc1",
                "Nc1ccccc1",
                "Clc1ccccc1",
                "Fc1ccccc1",
            ],
            "pactivity": np.linspace(5, 9, 6),
            "activity_class": ["inactive", "intermediate", "active", "potent", "active", "active"],
            "group": [2, 2, 1, 1, 1, 1],
        }
    )
    app.run()
    assert not calls
    widget(app, "button", "Run ECFP4 UMAP").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert app.session_state["structural_space"].settings["metric"] == "jaccard"
    widget(app, "radio", "Color structural space by").set_value("Continuous potency").run()
    assert not app.exception
    widget(app, "slider", "UMAP minimum distance").set_value(0.3).run()
    assert not app.exception
    assert any("UMAP controls have changed" in item.value for item in app.info)
    assert len(calls) == 1


def test_descriptor_regression_ui_and_optional_bootstrap_are_user_controlled(app_api):
    import numpy as np

    from sarscope.curate import assign_classes
    from sarscope.params import ClassScheme

    app = AppTest.from_file(str(APP), default_timeout=30).run()
    widget(app, "button", "Check target").click().run()
    widget(app, "button", "1 · Curate CHEMBL5145").click().run()
    stage = app.session_state["curation_stage"][1]
    smiles = [
        core.format(sub)
        for core in ["c1ccc({})cc1", "c1ccnc({})c1", "C1CCC({})CC1", "c1cc({})sc1"]
        for sub in ["C", "CC", "CCC", "CO", "CN", "Cl", "F", "OC", "C(F)(F)F", "C#N"]
    ]
    potency = pd.Series(np.tile([5.5, 6.5, 7.5, 8.5], 10))
    labels, groups = assign_classes(potency, ClassScheme())
    stage.table = pd.DataFrame(
        {
            "molecule_id": [f"m{i}" for i in range(40)],
            "smiles": smiles,
            "pactivity": potency,
            "activity_class": labels,
            "group": groups,
            "document_year": 2018,
        }
    )
    stage.curation.table = stage.table
    app.run()
    widget(app, "button", "Run scaffold analysis").click().run()
    assert not app.exception
    widget(app, "multiselect", "Prediction tasks").set_value(
        ["Continuous pActivity regression"]
    ).run()
    widget(app, "selectbox", "Molecular representation").set_value("RDKit 2D descriptors").run()
    widget(app, "selectbox", "Validation split").set_value("random").run()
    widget(app, "slider", "CV folds").set_value(3).run()
    widget(app, "multiselect", "Regression algorithms").set_value(["ridge"]).run()
    names = ["MolWt", "MolLogP", "TPSA", "NumHDonors"]
    widget(app, "multiselect", "Descriptors to include").set_value(names).run()
    assert "ml_stage" not in app.session_state
    widget(app, "button", "Run selected ML").click().run()
    assert not app.exception
    fitted = app.session_state["ml_stage"]
    assert fitted.regression.best_algorithm == "ridge"
    assert fitted.models is None
    assert fitted.prediction_bundle.feature_names == tuple(names)
    assert "regression_bootstrap" not in app.session_state
    assert any(item.label == "Within 3-fold" for item in app.metric)
    widget(app, "button", "Estimate metric confidence intervals").click().run()
    assert not app.exception
    assert "regression_bootstrap" in app.session_state
    widget(app, "multiselect", "Descriptors to include").set_value(["MolWt", "TPSA"]).run()
    assert not app.exception
    assert any("ML feature/model choices changed" in item.value for item in app.warning)


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

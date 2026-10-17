"""Design remains opt-in and stale settings never display old proposals/scores."""

from types import SimpleNamespace

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

SCRIPT = """
import pandas as pd
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analogue_design_ui import show_analogue_design
table=add_scaffolds(pd.DataFrame([
 {"molecule_id":"M1","smiles":"Cc1cc(F)ncc1","pactivity":6.,"activity_class":"series"},
 {"molecule_id":"M2","smiles":"Cc1cc(Cl)ncc1","pactivity":7.,"activity_class":"series"},
 {"molecule_id":"M3","smiles":"CCc1cc(F)ncc1","pactivity":6.5,"activity_class":"series"},
]))
series=decompose_scaffold(table,table.murcko.iloc[0],min_substituent_count=1)
evidence=table[["molecule_id","pactivity"]].assign(
 record_id=["1","2","3"],assay_chembl_id="A1",standard_type="IC50",
 standard_relation="=",assay_variant_mutation=None)
show_analogue_design([series],table,evidence)
"""


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


def test_no_generation_or_fitting_on_load_and_small_model_disabled():
    app = AppTest.from_string(SCRIPT, default_timeout=20).run()
    assert not app.exception
    assert "analogue_design_result" not in app.session_state
    assert "analogue_design_model" not in app.session_state
    assert widget(app, "button", "Fit scoped 2D model").disabled
    assert widget(app, "button", "Score generated analogues").disabled


def test_generation_shows_measured_evidence_and_is_invalidated_by_parent_changes():
    app = AppTest.from_string(SCRIPT, default_timeout=20).run()
    widget(app, "selectbox", "Parent compound").set_value("M3").run()
    widget(app, "button", "Generate analogue proposals").click().run()
    assert not app.exception
    result = app.session_state["analogue_design_result"][1]
    assert len(result.candidates) == 2
    assert any(item.label == "Inspect proposed analogue" for item in app.selectbox)
    assert len(app.get("download_button")) == 2
    assert len(app.get("vega_lite_chart")) == 1
    assert "analogue_design_model" not in app.session_state
    widget(app, "selectbox", "Parent compound").set_value("M1").run()
    assert not app.exception
    assert not any(item.label == "Inspect proposed analogue" for item in app.selectbox)
    assert any("old results are hidden" in item.value for item in app.info)


def test_limits_and_locked_positions_invalidate_proposals_without_automatic_rebuild():
    app = AppTest.from_string(SCRIPT, default_timeout=20).run()
    widget(app, "button", "Generate analogue proposals").click().run()
    assert not app.exception
    old = app.session_state["analogue_design_result"]
    widget(app, "slider", "Maximum proposals").set_value(1).run()
    assert not app.exception
    assert app.session_state["analogue_design_result"][0] == old[0]
    assert not any(item.label == "Inspect proposed analogue" for item in app.selectbox)
    widget(app, "multiselect", "Editable positions").set_value([]).run()
    assert widget(app, "button", "Generate analogue proposals").disabled


def test_upload_entry_needs_no_target_download_or_alignment():
    app = AppTest.from_string(
        "from sarscope.analogue_design_ui import show_design_upload\nshow_design_upload()",
        default_timeout=20,
    ).run()
    assert not app.exception
    assert any("No 3D alignment" in item.value for item in app.caption)
    assert any("molecule_id,smiles,pactivity" in item.value for item in app.code)
    assert not app.button


def test_failed_model_and_stale_scores_cannot_enable_scoring(monkeypatch):
    import sarscope.analogue_design_ui as ui
    from sarscope.analogue_design import DesignModel

    def fake_fit(table, evidence, params, settings, **kwargs):
        return DesignModel(
            None,
            SimpleNamespace(scores=pd.DataFrame({"test_r2": [0.04]})),
            ui.scope_signature(table, evidence),
            pd.DataFrame({"check": ["test R2"], "value": [0.04], "passed": [False]}),
            {},
        )

    monkeypatch.setattr(ui, "fit_design_model", fake_fit)
    extra = """
table=pd.concat([table,pd.DataFrame([
 {"molecule_id":f"X{i}","smiles":"C"*(i+1),"pactivity":5.+i*.01,
  "activity_class":"series"} for i in range(37)])],ignore_index=True)
evidence=table[["molecule_id","pactivity"]].assign(
 record_id=table.molecule_id,assay_chembl_id="A1",standard_type="IC50",
 standard_relation="=",assay_variant_mutation=None)
"""
    script = SCRIPT.replace(
        "show_analogue_design([series],table,evidence)",
        extra + "\nshow_analogue_design([series],table,evidence)",
    )
    app = AppTest.from_string(script, default_timeout=20).run()
    widget(app, "button", "Generate analogue proposals").click().run()
    assert "analogue_design_result" in app.session_state
    widget(app, "button", "Fit scoped 2D model").click().run()
    assert not app.exception
    assert widget(app, "button", "Score generated analogues").disabled
    assert any("Model checks failed" in item.value for item in app.warning)
    # Changing estimator hides the mismatched old model and leaves scoring disabled.
    widget(app, "selectbox", "Design estimator").set_value("ridge").run()
    assert not app.exception
    assert widget(app, "button", "Score generated analogues").disabled
    assert not any("Model checks failed" in item.value for item in app.warning)


def test_optional_example_has_guided_defaults_and_runs_end_to_end_without_network(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx.Client, "send", lambda *args, **kwargs: pytest.fail("Offline example")
    )
    app = AppTest.from_string(
        "from sarscope.analogue_design_ui import show_design_upload\nshow_design_upload()",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert not widget(app, "checkbox", "Use measured EGFR example").value
    assert not app.button  # downloads only; no preparation, fitting or generation on load
    widget(app, "checkbox", "Use measured EGFR example").check().run()
    assert not app.exception
    assert (
        widget(app, "selectbox", "Shared scaffold to prepare").value == "c1ccc(Cc2ncnc3ccccc23)cc1"
    )
    widget(app, "button", "Prepare design series").click().run()
    assert not app.exception
    assert widget(app, "selectbox", "Parent compound").value == "CHEMBL522134"
    assert "upload_analogue_design_design_result" not in app.session_state
    widget(app, "button", "Generate analogue proposals").click().run()
    assert not app.exception
    result = app.session_state["upload_analogue_design_design_result"][1]
    assert result.candidates.known_ids.eq("").sum() == 6
    assert result.provenance["source"]["assay"] == "CHEMBL944276"
    assert not result.provenance["source"].get("user_declared_context", False)
    widget(app, "button", "Fit scoped 2D model").click().run()
    assert not app.exception
    assert not widget(app, "button", "Score generated analogues").disabled
    widget(app, "button", "Score generated analogues").click().run()
    assert not app.exception
    scored = app.session_state["upload_analogue_design_design_scored"][1]
    assert scored.candidates.predicted_pactivity.notna().sum() == 7
    # Switching back to own data cannot leave an example or predictions visible.
    widget(app, "checkbox", "Use measured EGFR example").uncheck().run()
    assert not app.exception
    assert not any(item.label == "Parent compound" for item in app.selectbox)

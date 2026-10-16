"""Streamlit workflow checks; browser WebGL rendering is not exercised by AppTest."""

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
from test_field_qsar import synthetic_result as _synthetic_result  # noqa: E402


@pytest.fixture(scope="module")
def field_result():
    return _synthetic_result.__wrapped__()


SCRIPT = """
import streamlit as st
import sarscope.field_qsar_ui as ui
dataset = st.session_state.fixture.prepared.dataset
ui.show_field_qsar(dataset.table, dataset.measurements, key='test_field')
"""


def test_upload_entry_does_not_require_chembl_or_run_models():
    app = AppTest.from_string(
        "from sarscope.field_qsar_ui import show_field_qsar\nshow_field_qsar()"
    ).run()
    assert not app.exception
    assert any("SEH steroid benchmark agreement" in item.value for item in app.warning)
    assert any(
        "Native end-to-end validation remains incomplete" in item.value for item in app.warning
    )
    assert not any(button.label == "Run experimental 3D QSAR" for button in app.button)
    assert "field_qsar_result" not in app.session_state


def test_prepare_review_run_and_setting_changes_hide_old_results(field_result, monkeypatch):
    import sarscope.field_qsar_ui as ui

    monkeypatch.setattr(ui, "_assay_dataset", lambda *args: field_result.prepared.dataset)
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    assert not app.exception
    assert "test_field_prepared" not in app.session_state
    next(b for b in app.button if b.label == "Prepare and review alignment").click().run()
    assert not app.exception
    run = next(b for b in app.button if b.label == "Run experimental 3D QSAR")
    assert run.disabled
    next(
        c for c in app.checkbox if c.label == "I reviewed the overlay and alignment assumptions"
    ).check()
    next(c for c in app.selectbox if c.label == "Training-label permutations").select(0)
    app.run()
    next(b for b in app.button if b.label == "Run experimental 3D QSAR").click().run()
    assert not app.exception
    assert "test_field_active_result" in app.session_state
    assert any("not validated" in w.value for w in app.warning)
    next(b for b in app.button if b.label == "Prepare 3D-QSAR evidence download").click().run()
    assert not app.exception
    assert "test_field_archive" in app.session_state
    next(c for c in app.selectbox if c.label == "Partial charges").select("mmff94").run()
    assert not app.exception
    assert "test_field_active_result" not in app.session_state
    assert any("Earlier results are hidden" in i.value for i in app.info)


def test_viewer_escapes_uploaded_script_fragments():
    from sarscope.field_qsar_ui import viewer_html

    content = viewer_html(["</script><script>alert('x')</script>"])
    assert "</script><script>alert" not in content
    assert "\\u003c/script>" in content


def test_training_stability_is_button_gated_and_excludes_heldout_rows(field_result, monkeypatch):
    from types import SimpleNamespace

    import pandas as pd

    import sarscope.field_sensitivity as sensitivity

    calls = []

    def fake(data, settings, model):
        calls.append(data.table.molecule_id.tolist())
        assert data.source["scope"] == "training_only"
        return SimpleNamespace(
            summary=pd.DataFrame(
                [
                    {
                        "variant": "pose_test",
                        "kind": "relative_pose",
                        "completed": True,
                        "engineering_stability_flag": False,
                        "oof_change_p95": 0.7,
                        "oof_change_max": 1.2,
                    }
                ]
            ),
            predictions=pd.DataFrame(),
            contributions=pd.DataFrame(),
            provenance={"held_out_labels_used": False},
        )

    monkeypatch.setattr(sensitivity, "run_sensitivity", fake)
    script = """
import streamlit as st
from sarscope.field_qsar_ui import show_training_stability
show_training_stability(st.session_state.fixture, 'test_stability')
"""
    app = AppTest.from_string(script)
    app.session_state["fixture"] = field_result
    app.run()
    assert not app.exception and not calls
    next(b for b in app.button if b.label == "Check training-only stability").click().run()
    assert not app.exception
    assert calls == [
        field_result.prepared.dataset.table.iloc[
            field_result.regression.train_index
        ].molecule_id.tolist()
    ]
    assert any("stability flags failed" in item.value for item in app.warning)


def test_reference_field_definition_exposes_se_only_and_explicit_overlap_policy(
    field_result, monkeypatch
):
    import sarscope.field_qsar_ui as ui

    monkeypatch.setattr(ui, "_assay_dataset", lambda *args: field_result.prepared.dataset)
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    next(b for b in app.button if b.label == "Prepare and review alignment").click().run()
    next(s for s in app.selectbox if s.label == "Field definition").select("tripos52_se").run()
    assert not app.exception
    selected = next(s for s in app.multiselect if s.label == "Fields included in the model")
    assert selected.value == ["steric", "electrostatic"]
    next(s for s in app.selectbox if s.label == "Molecular fields").select("comfa").run()
    assert not app.exception
    assert (
        next(s for s in app.selectbox if s.label == "Missing electrostatic descriptors").value
        == "training_mean"
    )
    assert (
        next(s for s in app.selectbox if s.label == "Electrostatic plateau (± kcal/mol)").value
        == 30
    )
    assert any("No vendor-equivalence claim" in item.value for item in app.info)


def test_empty_fields_are_warned_about_and_not_offered_as_contours(field_result):
    import io
    import zipfile
    from dataclasses import replace

    from sarscope.field_qsar_report import field_qsar_archive

    summary = field_result.contours.summary.copy()
    summary.loc[summary.field.eq("electrostatic"), "retained_columns"] = 0
    changed = replace(field_result, contours=replace(field_result.contours, summary=summary))
    script = """
import streamlit as st
from sarscope.field_qsar_ui import show_field_result
show_field_result(st.session_state.fixture, 'test_empty_fields')
"""
    app = AppTest.from_string(script)
    app.session_state["fixture"] = changed
    app.run()
    assert not app.exception
    assert any("Filtering retained no columns for" in w.value for w in app.warning)
    choices = next(s for s in app.selectbox if s.label == "Contour field")
    assert "electrostatic" not in choices.options
    with zipfile.ZipFile(io.BytesIO(field_qsar_archive(changed))) as archive:
        assert b"Filtering retained no columns for" in archive.read("report.html")

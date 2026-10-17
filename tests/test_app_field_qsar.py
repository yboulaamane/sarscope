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
    assert any("To enable Run, tick" in item.value for item in app.info)
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


def test_invalid_scaffold_split_explains_reason_and_disables_run(field_result, monkeypatch):
    import sarscope.field_qsar_ui as ui

    monkeypatch.setattr(ui, "_assay_dataset", lambda *args: field_result.prepared.dataset)
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    app.button(key="test_field_prepare").click().run()
    app.checkbox(key="test_field_reviewed").check().run()
    app.selectbox(key="test_field_split").select("scaffold").run()
    assert not app.exception
    assert app.button(key="test_field_run").disabled
    assert any(
        "All compounds share one scaffold. Use a random split." in w.value for w in app.warning
    )
    assert not any("Run the model for these settings" in i.value for i in app.info)


def test_model_error_replaces_stale_run_prompt_and_is_cleared_on_setting_change(
    field_result, monkeypatch
):
    import sarscope.field_qsar_ui as ui

    monkeypatch.setattr(ui, "_assay_dataset", lambda *args: field_result.prepared.dataset)

    def fail(*args):
        raise ValueError("Overlap: review Missing electrostatic descriptors")

    monkeypatch.setattr(ui, "run_field_qsar_stage", fail)
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    app.button(key="test_field_prepare").click().run()
    app.checkbox(key="test_field_reviewed").check().run()
    app.button(key="test_field_run").click().run()
    assert not app.exception
    assert any("Missing electrostatic descriptors" in e.value for e in app.error)
    assert not any("Run the model for these settings" in i.value for i in app.info)
    app.run()
    assert app.error  # Remains visible on ordinary rerun, not only the failed button event.
    app.selectbox(key="test_field_charge").select("mmff94").run()
    assert not app.error
    assert any("Earlier results are hidden" in i.value for i in app.info)


def test_field_caption_and_overlap_setting_follow_selected_method(field_result, monkeypatch):
    import sarscope.field_qsar_ui as ui

    monkeypatch.setattr(ui, "_assay_dataset", lambda *args: field_result.prepared.dataset)
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    app.button(key="test_field_prepare").click().run()
    app.selectbox(key="test_field_method").select("comfa").run()
    assert not app.exception
    assert any("Lennard–Jones steric energy" in c.value for c in app.caption)
    assert app.selectbox(key="test_field_missing_sarscope").value == "refuse"
    # The policy must not disappear into the collapsed model-settings expander.
    expander = next(e for e in app.expander if e.label == "Field and model settings")
    assert not any(s.label == "Missing electrostatic descriptors" for s in expander.selectbox)
    spacing = app.selectbox(key="test_field_spacing")
    assert spacing.options == ["1.0", "1.5", "2.0", "2.5", "3.0"]
    assert spacing.value == 2.0


def test_prediction_plot_axes_colours_reference_and_permutation_label(field_result):
    import json

    script = """
import streamlit as st
from sarscope.field_qsar_ui import show_field_result
show_field_result(st.session_state.fixture, 'plot_fix')
"""
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    assert not app.exception
    charts = app.get("vega_lite_chart")
    predicted = json.loads(charts[0].proto.spec)
    assert predicted["layer"][0]["mark"]["strokeDash"] == [6, 4]
    scatter = predicted["layer"][1]["encoding"]
    assert scatter["x"]["type"] == "quantitative"
    assert scatter["x"]["axis"]["tickCount"] == 6
    assert scatter["x"]["scale"]["domain"] == scatter["y"]["scale"]["domain"]
    assert len(set(scatter["color"]["scale"]["range"])) == 2
    residuals = json.loads(charts[1].proto.spec)
    assert residuals["encoding"]["x"]["type"] == "quantitative"
    assert any("Green vertical line: observed pooled outer-CV" in c.value for c in app.caption)
    assert any("not interchangeable" in c.value for c in app.caption)
    assert any("zero-based index" in c.value for c in app.caption)


def test_viewer_escapes_uploaded_script_fragments():
    from sarscope.field_qsar_ui import viewer_html

    content = viewer_html(["</script><script>alert('x')</script>"])
    assert "</script><script>alert" not in content
    assert "\\u003c/script>" in content


@pytest.mark.parametrize("compound_count", [55, 125])
def test_scaffold_change_resets_compounds_but_ordinary_reruns_preserve_edits(
    field_result, monkeypatch, compound_count
):
    from dataclasses import replace

    import pandas as pd

    import sarscope.field_qsar_ui as ui

    original = field_result.prepared.dataset
    table = pd.concat([original.table] * 7, ignore_index=True).iloc[:compound_count].copy()
    table["molecule_id"] = [f"selection_test_{i}" for i in range(compound_count)]
    dataset = replace(
        original,
        table=table,
        measurements=table[["molecule_id", "smiles", "pactivity"]].copy(),
        molecules=[original.molecules[i % len(original.molecules)] for i in range(compound_count)],
    )
    current = {"dataset": dataset}
    monkeypatch.setattr(ui, "_assay_dataset", lambda *args: current["dataset"])
    # Controlled ten-member groups isolate widget state from scaffold chemistry.
    monkeypatch.setattr(
        ui,
        "add_scaffolds",
        lambda table: table.assign(murcko=[f"series_{i // 10}" for i in range(len(table))]),
    )
    app = AppTest.from_string(SCRIPT, default_timeout=30)
    app.session_state["fixture"] = field_result
    app.run()
    assert not app.exception
    assert app.selectbox(key="test_field_scaffold").value == "All compounds"
    assert len(app.multiselect(key="test_field_compounds").value) == (
        compound_count if compound_count <= 100 else 0
    )
    app.selectbox(key="test_field_scaffold").select("series_0").run()
    first_scaffold = "series_0"
    chosen = app.multiselect(key="test_field_compounds").value[1:]
    app.multiselect(key="test_field_compounds").set_value(chosen).run()
    app.run()
    assert app.multiselect(key="test_field_compounds").value == chosen
    app.button(key="test_field_select_all").click().run()
    assert len(app.multiselect(key="test_field_compounds").value) == 10
    assert any("alignment and modelling controls appear" in i.value for i in app.info)
    app.selectbox(key="test_field_scaffold").select("All compounds").run()
    assert not app.exception
    expected = table.molecule_id.tolist() if compound_count <= 100 else []
    assert app.multiselect(key="test_field_compounds").value == expected
    if expected:
        app.multiselect(key="test_field_compounds").set_value(expected[1:]).run()
        app.run()
        assert app.multiselect(key="test_field_compounds").value == expected[1:]
        current["dataset"] = replace(dataset, source={"kind": "replacement upload"})
        app.run()
        assert app.multiselect(key="test_field_compounds").value == expected
    app.selectbox(key="test_field_scaffold").select(first_scaffold).run()
    assert not app.exception
    assert len(app.multiselect(key="test_field_compounds").value) == 10


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
                    },
                    {
                        "variant": "common_translation",
                        "kind": "common_frame",
                        "completed": True,
                        "engineering_stability_flag": True,
                        "oof_change_p95": 1e-14,
                        "oof_change_max": 1e-14,
                    },
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
    import json

    chart = json.loads(app.get("vega_lite_chart")[0].proto.spec)
    assert chart["mark"]["type"] == "point"
    assert chart["encoding"]["y"]["axis"]["labelLimit"] >= 250
    assert any("point at zero" in c.value for c in app.caption)


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

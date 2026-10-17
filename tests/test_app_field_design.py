"""The example and post-fit design controls live inside the 3D workflow."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
from test_field_qsar import synthetic_result as _synthetic_result  # noqa: E402


def widget(app, kind, label):
    return next(item for item in getattr(app, kind) if item.label == label)


@pytest.fixture(scope="module")
def fitted():
    return _synthetic_result.__wrapped__()


def test_real_example_is_at_3d_upload_stage_and_nothing_runs_automatically(monkeypatch):
    import httpx

    monkeypatch.setattr(httpx.Client, "send", lambda *args, **kwargs: pytest.fail("Offline demo"))
    app = AppTest.from_string(
        "from sarscope.field_qsar_ui import show_field_qsar\nshow_field_qsar(key='own3d')",
        default_timeout=30,
    ).run()
    assert not app.exception
    assert not widget(app, "checkbox", "Use measured 3D-QSAR example").value
    assert not any(b.label == "Run experimental 3D QSAR" for b in app.button)
    assert "own3d_result" not in app.session_state
    widget(app, "checkbox", "Use measured 3D-QSAR example").check().run()
    assert not app.exception
    assert len(widget(app, "multiselect", "Compounds in this 3D-QSAR series").value) == 75
    assert widget(app, "selectbox", "Reference compound").value == "CCR5_01"
    assert widget(app, "selectbox", "Alignment route").value == "prealigned"
    assert "own3d_prepared" not in app.session_state
    assert "own3d_result" not in app.session_state
    assert not any("Fit scoped 2D" in b.label for b in app.button)
    assert any("0.499" in item.value and "remains failed" in item.value for item in app.warning)
    widget(app, "selectbox", "Measured 3D example").select("egfr").run()
    assert not app.exception
    assert len(widget(app, "multiselect", "Compounds in this 3D-QSAR series").value) == 50
    assert widget(app, "selectbox", "Reference compound").value == "CHEMBL29197"


def test_example_reset_is_explicit_and_does_not_prepare_or_train():
    app = AppTest.from_string(
        "from sarscope.field_qsar_ui import show_field_qsar\nshow_field_qsar(key='reset3d')",
        default_timeout=30,
    ).run()
    widget(app, "checkbox", "Use measured 3D-QSAR example").check().run()
    widget(app, "button", "Prepare and review alignment").click().run()
    assert not app.exception
    widget(app, "selectbox", "Molecular fields").select("comfa").run()
    widget(app, "selectbox", "Grid spacing (Å)").select(3.0).run()
    widget(app, "button", "Restore example model defaults").click().run()
    assert not app.exception
    assert "reset3d_prepared" not in app.session_state
    assert "reset3d_result" not in app.session_state
    assert not any(b.label == "Run experimental 3D QSAR" for b in app.button)
    assert widget(app, "selectbox", "Reference compound").value == "CCR5_01"


def test_3d_design_uses_active_field_model_and_requires_exploratory_consent(fitted):
    script = """
import streamlit as st
from sarscope.field_design_ui import show_field_design
show_field_design(st.session_state.fixture, 'design3d')
"""
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["fixture"] = fitted
    app.run()
    assert not app.exception
    assert not any("Fit scoped 2D" in b.label for b in app.button)
    widget(app, "button", "Prepare R-positions for 3D design").click().run()
    assert not app.exception
    widget(app, "button", "Generate proposals for this 3D model").click().run()
    assert not app.exception
    assert not any(
        s.value == "Proposed experiments, not confirmed improvements" for s in app.subheader
    )
    assert widget(app, "button", "Align and predict with this 3D model").disabled
    consent = widget(
        app, "checkbox", "I understand: show exploratory predictions from this unvalidated 3D model"
    )
    consent.check().run()
    assert not app.exception
    widget(app, "button", "Align and predict with this 3D model").click().run()
    assert not app.exception
    prediction = app.session_state["design3d_field_design_predictions"][1]
    assert prediction.audit["prediction_model"] == "3D field PLS"
    assert prediction.audit["2d_fallback"] is False
    assert not prediction.contributions.empty
    assert any("not candidate prediction minus the measured parent" in c.value for c in app.caption)
    assert any(s.value == "Proposed experiments, not confirmed improvements" for s in app.subheader)
    widget(
        app, "checkbox", "I understand: show exploratory predictions from this unvalidated 3D model"
    ).uncheck().run()
    assert not any("Prediction engine: the displayed" in c.value for c in app.caption)
    assert widget(app, "button", "Align and predict with this 3D model").disabled


def test_design_needs_pose_consent_even_with_green_scores_and_new_evidence_hides_old_predictions(
    fitted,
):
    from dataclasses import replace

    from test_field_interpretation import passing_evidence

    checks = fitted.checks.copy()
    checks["passed"] = True  # Artificial fixture isolates the pose-evidence gate.
    result = replace(
        fitted, checks=checks, provenance=dict(fitted.provenance), training_stability=None
    )
    app = AppTest.from_string(
        "import streamlit as st\nfrom sarscope.field_design_ui import show_field_design\n"
        "show_field_design(st.session_state.fixture, 'pose_design')",
        default_timeout=30,
    )
    app.session_state["fixture"] = result
    app.run()
    widget(app, "button", "Prepare R-positions for 3D design").click().run()
    widget(app, "button", "Generate proposals for this 3D model").click().run()
    assert widget(app, "button", "Align and predict with this 3D model").disabled
    app.checkbox(key="pose_design_field_design_exploratory").check().run()
    widget(app, "button", "Align and predict with this 3D model").click().run()
    assert not app.exception
    assert any("Prediction engine: the displayed" in c.value for c in app.caption)
    saved = app.session_state["fixture"]
    saved.training_stability = passing_evidence(saved)
    saved.training_stability["summary"].loc[0, "oof_change_max"] = 1.2
    app.run()
    assert not app.exception
    assert not any("Prediction engine: the displayed" in c.value for c in app.caption)
    assert any("Alignment-sensitive" in w.value for w in app.warning)

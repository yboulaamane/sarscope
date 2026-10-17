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
    assert len(widget(app, "multiselect", "Compounds in this 3D-QSAR series").value) == 50
    assert widget(app, "selectbox", "Reference compound").value == "CHEMBL29197"
    assert widget(app, "selectbox", "Alignment route").value == "prealigned"
    assert "own3d_prepared" not in app.session_state
    assert "own3d_result" not in app.session_state
    assert not any("Fit scoped 2D" in b.label for b in app.button)


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
    widget(
        app, "checkbox", "I understand: show exploratory predictions from this unvalidated 3D model"
    ).uncheck().run()
    assert not any("Prediction engine: the displayed" in c.value for c in app.caption)
    assert widget(app, "button", "Align and predict with this 3D model").disabled

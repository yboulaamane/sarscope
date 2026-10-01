"""The browser domain panel displays the fitted boundary, not a reliability promise."""

import json

import numpy as np
import pytest

from sarscope.analysis.domain import DomainResult

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def domain_fixture(dimensions=2):
    return DomainResult(
        np.array([True, False]),
        np.array([[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [1.0, 1.0]])[:, :dimensions],
        np.array([[0.0, 0.0], [3.0, 0.0]])[:, :dimensions],
    )


def test_domain_chart_preserves_ids_flags_and_training_only_boundary():
    from streamlit_app import applicability_domain_chart

    chart, points = applicability_domain_chart(
        domain_fixture(), list("abcd"), ["inside", "outside"]
    )
    spec = chart.to_dict()
    bounds = spec["datasets"][spec["layer"][0]["data"]["name"]]
    assert bounds == [{"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0}]
    assert points.molecule_id.tolist() == ["a", "b", "c", "d", "inside", "outside"]
    queries = points[points.status != "Training compound"]
    assert queries.inside_pca_box.tolist() == [True, False]
    assert queries.PC1.tolist() == [0.0, 3.0]


def test_one_component_is_an_interval_without_fabricated_pc2():
    from streamlit_app import applicability_domain_chart

    chart, points = applicability_domain_chart(domain_fixture(1), list("abcd"), ["i", "o"])
    assert "PC2" not in points
    assert "field" not in chart.to_dict()["layer"][1]["encoding"]["y"]


def test_plot_sampling_does_not_change_export_or_full_boundary():
    from streamlit_app import applicability_domain_chart

    domain = DomainResult(
        np.ones(5500, dtype=bool),
        np.column_stack([np.arange(6000), np.arange(6000)]),
        np.zeros((5500, 2)),
    )
    chart, points = applicability_domain_chart(
        domain, [f"t{i}" for i in range(6000)], [f"q{i}" for i in range(5500)]
    )
    spec = chart.to_dict()  # Must not exceed Altair's default row limit.
    assert len(points) == 11500
    assert sorted(len(data) for data in spec["datasets"].values()) == [1, 4000]
    bounds = spec["datasets"][spec["layer"][0]["data"]["name"]][0]
    assert bounds["x_max"] == 5999


def panel_app(*, regression=False, dimensions=2, empty=False, all_inside=False):
    script = f"""
from types import SimpleNamespace
import numpy as np
import pandas as pd
from streamlit_app import MlStage, show_applicability_domain
from sarscope.analysis.domain import DomainResult
from sarscope.params import RunParams
selected = SimpleNamespace(train_index=np.arange(4), test_index=np.array([4,5]))
domain = DomainResult(np.array([True,False]),
    np.array([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])[:, :{dimensions}],
    np.array([[0.,0.],[3.,0.]])[:, :{dimensions}])
profile = pd.DataFrame({{"molecule_id":["i","o"], "max_training_similarity":[0.2,0.9],
    "training_similarity_cutoff":[0.5,0.5], "in_training_domain":[False,True],
    "pactivity":[7.,7.], "predicted_pactivity":[5.8,6.8], "absolute_error":[1.2,0.2]}})
if {all_inside}:
    domain.in_domain[1] = True
    domain.query_scores[1, 0] = 0.5
if {empty}:
    domain.in_domain = np.array([], dtype=bool)
    domain.query_scores = np.empty((0, {dimensions}))
stage = MlStage(RunParams(), pd.DataFrame({{"molecule_id":list("abcd")+["i","o"]}}),
    None if {regression} else selected, selected if {regression} else None, domain,
    regression_test_predictions=profile if {regression} else pd.DataFrame())
show_applicability_domain(stage)
"""
    return AppTest.from_string(script, default_timeout=20).run()


@pytest.mark.parametrize("dimensions", [1, 2])
def test_classification_domain_panel_has_plot_counts_and_interpretation(dimensions):
    app = panel_app(dimensions=dimensions)
    assert not app.exception
    assert next(m for m in app.metric if m.label == "Inside PCA box").value == "1 / 2"
    assert len(app.get("vega_lite_chart")) == 1
    assert any("accuracy" in item.value for item in app.warning)
    assert any("not refitted" in item.value for item in app.markdown)
    assert app.get("download_button")


def test_regression_domain_panel_separates_similarity_rule_and_actual_errors():
    app = panel_app(regression=True)
    assert not app.exception
    charts = app.get("vega_lite_chart")
    assert len(charts) == 2
    spec = json.loads(charts[1].proto.spec)
    assert spec["layer"][1]["mark"]["type"] == "rule"
    assert any(
        "0.500" in item.value and "separate heuristic" in item.value for item in app.markdown
    )
    errors = app.dataframe[0].value.set_index("PCA domain status")
    assert errors.loc["Test · inside PCA box", "MAE (log units)"] == pytest.approx(1.2)
    assert errors.loc["Test · outside PCA box", "MAE (log units)"] == pytest.approx(0.2)


def test_empty_test_set_reports_undefined_coverage_without_crashing():
    app = panel_app(empty=True)
    assert not app.exception
    assert any("undefined" in item.value for item in app.info)
    assert not app.metric


def test_full_pca_coverage_does_not_hide_absent_outside_comparison():
    app = panel_app(regression=True, all_inside=True)
    assert not app.exception
    coverage = next(m for m in app.metric if m.label.startswith("Test compounds inside"))
    assert coverage.value == "100.0%"
    errors = app.dataframe[0].value.set_index("PCA domain status")
    assert errors.loc["Test · outside PCA box", "Test compounds"] == 0
    assert np.isnan(errors.loc["Test · outside PCA box", "MAE (log units)"])
    assert any("cannot be assessed" in item.value for item in app.caption)
    assert any(
        "Structural similarity coverage: 1 / 2 (50.0%)" in item.value for item in app.markdown
    )

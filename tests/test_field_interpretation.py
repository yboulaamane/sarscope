"""Interpretation contracts on artificial data, not evidence of biological validity."""

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from test_field_design import designed as _designed
from test_field_qsar import synthetic_result as _synthetic_result

from sarscope.field_design import (
    field_design_archive,
    field_model_signature,
    predict_field_analogues,
)
from sarscope.field_interpretation import (
    POSE_VARIANTS,
    interpretation_review,
    interpretation_signature,
    training_pose_ranges,
)
from sarscope.field_qsar_report import field_qsar_archive


@pytest.fixture(scope="module")
def fitted():
    return _synthetic_result.__wrapped__()


def passing_evidence(result):
    rows = [
        {
            "variant": name,
            "kind": "relative_pose",
            "completed": True,
            "engineering_stability_flag": True,
            "oof_change_p95": 0.05,
            "oof_change_max": 0.1,
        }
        for name in POSE_VARIANTS
    ]
    rows += [
        {
            "variant": name,
            "kind": "numerical_control",
            "completed": True,
            "engineering_stability_flag": True,
            "oof_change_p95": 0.0,
            "oof_change_max": 0.0,
        }
        for name in ("baseline", "common_translation")
    ]
    ids = result.prepared.dataset.table.iloc[result.regression.train_index].molecule_id
    predictions = pd.DataFrame(
        [
            {"variant": name, "molecule_id": pid, "oof_prediction": float(i + shift / 10)}
            for shift, name in enumerate(POSE_VARIANTS)
            for i, pid in enumerate(ids)
        ]
    )
    return {
        "fitted_model_signature": field_model_signature(result),
        "summary": pd.DataFrame(rows),
        "predictions": predictions,
        "contributions": pd.DataFrame(),
        "provenance": {"held_out_rows_used": 0, "held_out_labels_used": False},
    }


def checked_fixture(fitted):
    # Isolate the new gate; production native-validation checks still remain failed.
    checks = fitted.checks.copy()
    checks["passed"] = True
    return replace(
        fitted, checks=checks, provenance=dict(fitted.provenance), training_stability=None
    )


def test_unassessed_sensitivity_is_not_a_pass_even_with_green_score_checks(fitted):
    result = checked_fixture(fitted)
    review = interpretation_review(result)
    assert review["pose_status"] == "not_assessed"
    assert review["requires_exploratory_consent"]
    design = _designed.__wrapped__(result)
    output = predict_field_analogues(result, design, route="rigid_mcs")
    assert output.design.candidates.predicted_pactivity.isna().all()
    assert output.design.candidates.prediction_status.str.contains("pose sensitivity").all()
    assert output.design.candidates.model_pose_status.eq("not_assessed").all()


def test_passing_bounded_checks_are_not_binding_pose_or_contour_certification(fitted):
    result = checked_fixture(fitted)
    result.training_stability = passing_evidence(result)
    review = interpretation_review(result)
    assert review["pose_status"] == "within_tested_limits"
    assert review["complete_pose_protocol"]
    assert not review["requires_exploratory_consent"]
    assert "does not validate" in review["pose_message"]
    assert "contour-location stability" in review["limits"]


@pytest.mark.parametrize(
    "failure",
    [
        "missing_pose",
        "unfinished",
        "nan",
        "flag_string",
        "bad_control",
        "unbound",
        "wrong_model",
        "heldout",
        "empty",
    ],
)
def test_partial_invalid_or_foreign_evidence_cannot_get_a_pass(fitted, failure):
    result = checked_fixture(fitted)
    evidence = passing_evidence(result)
    summary = evidence["summary"]
    if failure == "missing_pose":
        evidence["summary"] = summary.iloc[1:]
    elif failure == "unfinished":
        summary.loc[0, "completed"] = False
    elif failure == "nan":
        summary.loc[0, "oof_change_max"] = np.nan
    elif failure == "flag_string":
        summary["completed"] = "True"
    elif failure == "bad_control":
        summary.loc[4, "oof_change_max"] = 0.1
    elif failure == "unbound":
        evidence.pop("fitted_model_signature")
    elif failure == "wrong_model":
        evidence["fitted_model_signature"] = "different"
    elif failure == "heldout":
        evidence["provenance"]["held_out_labels_used"] = True
    else:
        evidence["summary"] = summary.iloc[:0]
    result.training_stability = evidence
    review = interpretation_review(result)
    assert review["pose_status"] == "incomplete"
    assert not review["complete_pose_protocol"]
    assert review["requires_exploratory_consent"]


def test_pose_failures_are_visible_in_predictions_and_evidence_without_refitting(fitted):
    result = checked_fixture(fitted)
    result.training_stability = passing_evidence(result)
    result.training_stability["summary"].loc[0, "oof_change_max"] = 1.2
    review = interpretation_review(result)
    assert review["pose_status"] == "sensitive"
    assert review["failed_pose_checks"] == 1
    design = _designed.__wrapped__(result)
    before = field_model_signature(result)
    output = predict_field_analogues(result, design, route="rigid_mcs", allow_exploratory=True)
    assert field_model_signature(result) == before
    assert output.audit["interpretation_review"]["pose_status"] == "sensitive"
    assert output.audit["2d_fallback"] is False and output.audit["refit"] is False
    assert output.design.candidates.model_pose_status.eq("sensitive").all()
    assert output.design.candidates.interpretation_warning.str.contains("Alignment-sensitive").all()
    with zipfile.ZipFile(io.BytesIO(field_qsar_archive(result))) as archive:
        assert json.loads(archive.read("interpretation.json"))["pose_status"] == "sensitive"
        assert b"Alignment-sensitive" in archive.read("report.html")
        assert b"Same-split 2D baseline" in archive.read("report.html")


def test_new_diagnostics_invalidate_old_candidate_export_even_if_model_is_unchanged(fitted):
    result = checked_fixture(fitted)
    design = _designed.__wrapped__(result)
    old = predict_field_analogues(result, design, route="rigid_mcs", allow_exploratory=True)
    before = field_model_signature(result)
    old_review = interpretation_signature(result)
    result.training_stability = passing_evidence(result)
    assert field_model_signature(result) == before
    assert interpretation_signature(result) != old_review
    with pytest.raises(ValueError, match="Stability evidence changed"):
        field_design_archive(old, result)


def test_latest_failed_attempt_preserves_previous_evidence_but_downgrades_its_label(fitted):
    result = checked_fixture(fitted)
    result.training_stability = passing_evidence(result)
    result.provenance["training_stability_attempt_error"] = "timeout"
    assert interpretation_review(result)["pose_status"] == "incomplete"
    assert len(result.training_stability["summary"]) == 5


def test_ranges_exclude_test_compounds_and_are_not_candidate_uncertainty(fitted):
    result = checked_fixture(fitted)
    result.training_stability = passing_evidence(result)
    test_id = result.prepared.dataset.table.iloc[result.regression.test_index[0]].molecule_id
    predictions = result.training_stability["predictions"]
    result.training_stability["predictions"] = pd.concat(
        [
            predictions,
            pd.DataFrame(
                [{"variant": POSE_VARIANTS[0], "molecule_id": test_id, "oof_prediction": 1e6}]
            ),
        ],
        ignore_index=True,
    )
    ranges = training_pose_ranges(result)
    assert test_id not in set(ranges.molecule_id)
    assert len(ranges) == len(result.regression.train_index)
    assert ranges.completed_pose_checks.eq(3).all()
    np.testing.assert_allclose(ranges.pose_oof_range, 0.2)


def test_field_dominance_is_an_interpretation_warning_not_a_claim_of_biological_importance(fitted):
    summary = fitted.contours.summary.copy()
    summary["absolute_contribution_percent"] = 0.0
    summary.loc[summary.index[0], "absolute_contribution_percent"] = 99.2
    result = replace(fitted, contours=replace(fitted.contours, summary=summary))
    review = interpretation_review(result)
    assert review["dominant_field"]["percent"] == 99.2
    assert "not causal" in review["dominant_field"]["interpretation"]
    assert review["baseline"]["field_rmse"] == pytest.approx(
        fitted.statistics.set_index("algorithm").loc["field_pls", "test_rmse"]
    )


def test_real_sensitivity_engine_attaches_interpretation_without_changing_the_displayed_fit(fitted):
    from sarscope.field_sensitivity import run_sensitivity, training_dataset
    from sarscope.params import ModelParams

    result = checked_fixture(fitted)
    before = field_model_signature(result)
    data = training_dataset(result.prepared, result.regression.train_index.tolist())
    evidence = run_sensitivity(
        data, result.prepared.params, ModelParams(split="random", cv_folds=3)
    )
    result.training_stability = {
        "fitted_model_signature": before,
        "summary": evidence.summary,
        "predictions": evidence.predictions,
        "contributions": evidence.contributions,
        "provenance": evidence.provenance,
    }
    review = interpretation_review(result)
    assert field_model_signature(result) == before
    assert len(evidence.summary) == 10
    assert review["pose_status"] in {"sensitive", "within_tested_limits"}
    assert review["complete_pose_protocol"]
    assert len(training_pose_ranges(result)) == len(result.regression.train_index)
    assert evidence.provenance["held_out_rows_used"] == 0
    assert not evidence.provenance["held_out_labels_used"]
    with zipfile.ZipFile(io.BytesIO(field_qsar_archive(result))) as archive:
        record = json.loads(archive.read("training_stability/provenance.json"))
        assert record["fitted_model_signature"] == before


def test_passing_pose_checks_cannot_hide_failed_charge_or_grid_diagnostics(fitted):
    result = checked_fixture(fitted)
    result.training_stability = passing_evidence(result)
    summary = result.training_stability["summary"]
    extra = {
        "variant": "mmff94",
        "kind": "charge",
        "completed": True,
        "engineering_stability_flag": False,
        "oof_change_p95": 0.7,
        "oof_change_max": 1.3,
    }
    result.training_stability["summary"] = pd.concat(
        [summary, pd.DataFrame([extra])], ignore_index=True
    )
    review = interpretation_review(result)
    assert review["pose_status"] == "within_tested_limits"
    assert review["other_sensitivity_failures"] == 1
    assert review["requires_exploratory_consent"]
    assert "charge/grid" in review["interpretation_message"]

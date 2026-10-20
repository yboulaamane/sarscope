"""Offline integrity/metric tests for the retrospective positive worked example."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

from sarscope.field_primary_example import primary_mmff_settings

EVIDENCE = Path(__file__).resolve().parents[1] / (
    "benchmarks/field_qsar/end-to-end-demo-20261019-v2"
)


def test_demo_manifest_and_scope_do_not_relabel_a_retrospective_case():
    for name, expected in json.loads((EVIDENCE / "manifest.sha256.json").read_text()).items():
        assert hashlib.sha256((EVIDENCE / name).read_bytes()).hexdigest() == expected
    scope = json.loads((EVIDENCE / "scope.json").read_text())
    assert scope["positive_fixed_alignment_worked_example"]
    assert not scope["new_blinded_validation"]
    assert not scope["csv_route_success"]
    assert not scope["pose_stress_qualified"]
    assert not scope["package_certified"]
    assert not list(EVIDENCE.rglob("*.sdf"))


def test_positive_demo_scores_recalculate_without_removing_test_compounds():
    folder = EVIDENCE / "thrombin_source_alignment"
    summary = json.loads((folder / "assessment.json").read_text())
    predictions = pd.read_csv(folder / "predictions.csv")
    train = predictions[predictions.split.eq("train")]
    test = predictions[predictions.split.eq("held_out")]
    assert len(predictions) == 88 and (len(train), len(test)) == (59, 29)
    assert np.isclose(r2_score(test.pactivity, test.prediction), summary["test_r2"])
    assert np.isclose(
        root_mean_squared_error(test.pactivity, test.prediction), summary["test_rmse"]
    )
    q2 = r2_score(train.pactivity, train.training_oof_prediction)
    assert np.isclose(q2, summary["nested_q2"])
    null = pd.read_csv(folder / "permutations.csv")
    assert len(null) == 99
    assert (1 + (null.q2 >= q2).sum()) / 100 == summary["permutation_p"]
    assert q2 > 0.5 and summary["test_r2"] > 0.6
    assert summary["test_rmse"] < summary["rf_test_rmse"] < summary["mean_rmse"]
    assert summary["held_out_label_invariance"]
    assert summary["numeric_replay_max_error"] < 1e-8
    assert not summary["all_validation_checks_pass"]


def test_fresh_prediction_audit_and_pose_failures_are_both_visible():
    folder = EVIDENCE / "thrombin_source_alignment"
    replay = json.loads((folder / "fresh_held_out_prediction_audit.json").read_text())
    assert replay["held_out"] == 29 and replay["passed"]
    assert replay["test_properties_removed_before_descriptor_calculation"]
    assert not replay["model_refit"] and not replay["grid_rebuilt"]
    assert replay["raw_field_max_error"] == 0
    stability = pd.read_csv(folder / "training_sensitivity.csv")
    pose = stability[stability.kind.eq("relative_pose")]
    assert len(pose) == 3 and pose.completed.all()
    assert not pose.engineering_stability_flag.any()
    assert stability[stability.kind.eq("grid_phase")].engineering_stability_flag.all()
    report = (EVIDENCE / "report.html").read_text()
    assert "Small-pose stress checks failed" in report


def test_failed_csv_route_and_unchanged_bounded_profile_are_retained():
    frame = pd.read_csv(EVIDENCE / "summary.csv").set_index("case")
    assert len(frame) == 2
    assert frame.loc["ccr5_generated_from_csv", "status"] == "failed"
    assert "no conformer" in frame.loc["ccr5_generated_from_csv", "error"]
    profile = primary_mmff_settings()
    assert profile.component_sample_ratio == 5 and profile.component_cap == 4
    assert profile.permutations == 99
    assert profile.field_definition == "mmff94_mif"
    assert profile.preserve_input_hydrogens
    assert not profile.allow_charge_fallback

"""Do not let a positive tutorial overwrite the adverse frozen validation evidence."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "benchmarks/field_qsar/ccr5-demo-validation-20261017"


def test_complete_retrospective_evidence_is_hash_verified_and_not_certified():
    # Deliberately required evidence, not a skip that hides removed failures.
    manifest = json.loads((RECORD / "manifest.sha256.json").read_text())
    assert len(manifest) >= 80
    for name, digest in manifest.items():
        path = (RECORD / name).resolve()
        assert path.is_relative_to(RECORD.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    freeze = json.loads((RECORD / "freeze.json").read_text())
    runner = RECORD.parent / "validate_ccr5_demo.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == freeze["runner_sha256"]
    protocol = RECORD.parent / "ccr5-demo-validation-protocol.json"
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == freeze["protocol_sha256"]
    assert json.loads(protocol.read_text()) == json.loads((RECORD / "protocol.json").read_text())
    assert freeze["previously_observed_demo_metrics"]
    assert not freeze["independent_external_validation"]
    assessment = json.loads((RECORD / "assessment.json").read_text())
    assert assessment["predictive_release"].startswith("NO-GO")
    assert not assessment["checks"]["independent_validation"]
    assert not assessment["checks"]["nested_q2"]
    assert not assessment["checks"]["all_training_stability_flags"]
    assert assessment["checks"]["post_alignment_leakage_controls"]
    assert not assessment["automatic_preset_changes"]


@pytest.mark.parametrize("name", ["primary", "historical-63-12", "scaffold"])
def test_exported_metrics_recalculate_from_actual_held_out_rows(name):
    data = pd.read_csv(RECORD / name / "predictions.csv")
    held_out = data.query("split == 'held_out'")
    stats = pd.read_csv(RECORD / name / "statistics.csv").set_index("algorithm")
    assert len(data) == 75 and data.molecule_id.is_unique
    for algorithm, column in [
        ("field_pls", "prediction"),
        ("random_forest", "baseline_prediction"),
    ]:
        assert r2_score(held_out.pactivity, held_out[column]) == pytest.approx(
            stats.loc[algorithm, "test_r2"], abs=1e-10
        )
        rmse = np.sqrt(np.mean((held_out.pactivity - held_out[column]) ** 2))
        assert rmse == pytest.approx(stats.loc[algorithm, "test_rmse"], abs=1e-10)
    train = data.query("split == 'train'")
    q2 = (
        1
        - ((train.pactivity - train.training_oof_prediction) ** 2).sum()
        / ((train.pactivity - train.pactivity.mean()) ** 2).sum()
    )
    assert q2 == pytest.approx(stats.loc["field_pls", "cv_pooled_q2"], abs=1e-10)


def test_permutations_intervals_and_failure_records_are_not_cherry_picked():
    assessment = json.loads((RECORD / "assessment.json").read_text())
    permutations = pd.read_csv(RECORD / "primary/permutations.csv")
    assert len(permutations) == 199
    p = (1 + (permutations.q2 >= assessment["primary"]["cv_pooled_q2"]).sum()) / 200
    assert p == assessment["primary"]["permutation_pvalue"] == 0.005
    repeats = pd.read_csv(RECORD / "additional-splits.csv")
    assert repeats.seed.tolist() == [11, 29, 47, 83, 131]
    assert repeats.test_r2.min() < 0.2
    assert assessment["historical_re_evaluation"]["test_r2"] < 0
    assert pd.read_csv(RECORD / "scaffold/statistics.csv").iloc[0].test_r2 < 0
    intervals = pd.read_csv(RECORD / "bootstrap.csv")
    assert intervals.scaffold_blocks.eq(5).all()
    assert intervals.repeats.eq(2000).all()
    r2 = intervals.query("comparison == '3d_vs_mean' and metric == 'R²'").iloc[0]
    assert r2.lower_95 < 0 < r2.upper_95
    stability = pd.read_csv(RECORD / "stability.csv")
    assert len(stability) == 10 and stability.completed.all()
    assert (~stability.engineering_stability_flag).sum() == 2
    control = stability.query("variant == 'common_translation'").iloc[0]
    assert control.oof_change_max < 1e-8


def test_separate_source_check_has_an_executable_identity_and_records_charge_changes():
    record = json.loads((RECORD.parent / "ccr5-source-audit-20261017.json").read_text())
    runner = RECORD.parent / "verify_ccr5_source_preparation.py"
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == record["runner_sha256"]
    assert record["records_checked"] == record["formal_charge_changed_records"] == 75
    assert record["activities_unchanged"]
    assert record["heavy_atom_elements_bonds_and_coordinates_identical"]
    assert not record["experimental_assay_or_pose_validation"]

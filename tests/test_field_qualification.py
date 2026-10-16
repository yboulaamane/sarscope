"""Freeze/one-shot guards are audit controls, not scientific certification."""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from sarscope import field_qualification as qualification
from sarscope import provenance
from sarscope.field_campaign import _json, _manifest
from sarscope.field_sensitivity import sensitivity_variants
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/field_qsar"


def evidence(tmp_path):
    spec = json.loads((ROOT / "public-se-qualification-protocol.json").read_text())
    protocol = tmp_path / "protocol.json"
    _json(protocol, spec)
    folder = tmp_path / "sensitivity"
    folder.mkdir()
    params = qualification.as_plain(replace(FieldQsarParams(**spec["fields"]), permutations=0))
    profile = spec["qualification"]["profile"]
    rows = []
    for name in spec["qualification"]["training_cohorts"]:
        target = folder / name / profile
        target.mkdir(parents=True)
        _json(
            target / "provenance.json",
            {
                "held_out_rows_used": 0,
                "held_out_labels_used": False,
                "settings": params,
                "model": qualification.as_plain(ModelParams(**spec["model"])),
                "workflow_sha256": qualification.workflow_hashes(),
            },
        )
        rows.extend(
            {
                "dataset": name,
                "profile": profile,
                "variant": v["name"],
                "kind": v["kind"],
                "completed": True,
                "engineering_stability_flag": v["kind"] != "relative_pose",
            }
            for v in sensitivity_variants()
        )
    pd.DataFrame(rows).to_csv(folder / "summary.csv", index=False)
    _manifest(folder)
    return protocol, folder


def test_freeze_retains_pose_warnings_and_receipt_prevents_a_second_evaluation(tmp_path):
    protocol, folder = evidence(tmp_path)
    freeze = tmp_path / "freeze.json"
    record = qualification.freeze_candidate(folder, protocol, freeze)
    assert record["sampling_gates_passed"]
    assert record["pose_warning_count"] == 9
    assert not record["package_or_model_certified"]
    assert not record["fresh_labels_downloaded_for_freeze"]
    assert "tripos_reference.py" in " ".join(record["workflow_sha256"])
    with pytest.raises(FileExistsError):
        qualification.freeze_candidate(folder, protocol, freeze)
    assert qualification.claim_evaluation(freeze, protocol, tmp_path / "out") == record
    with pytest.raises(FileExistsError):
        qualification.claim_evaluation(freeze, protocol, tmp_path / "different-out")


@pytest.mark.parametrize("failure", ["phase", "heldout", "source", "settings", "missing"])
def test_freeze_refuses_failed_or_stale_gates_before_test_download(tmp_path, failure):
    protocol, folder = evidence(tmp_path)
    path = folder / "AChE/public_se_gaussian_1A/provenance.json"
    record = json.loads(path.read_text())
    table = pd.read_csv(folder / "summary.csv")
    if failure == "phase":
        table.loc[table.variant.eq("phase_x"), "engineering_stability_flag"] = False
    elif failure == "heldout":
        record["held_out_labels_used"] = True
    elif failure == "source":
        record["workflow_sha256"]["analysis/fields.py"] = "0" * 64
    elif failure == "settings":
        record["settings"]["charge_model"] = "mmff94"
    else:
        table = table.iloc[:-1]
    table.to_csv(folder / "summary.csv", index=False)
    _json(path, record)
    _manifest(folder)
    with pytest.raises(ValueError):
        qualification.freeze_candidate(folder, protocol, tmp_path / "freeze.json")
    assert not (tmp_path / "freeze.json").exists()


def test_freeze_rejects_tampered_manifest_and_evaluation_rejects_protocol_drift(tmp_path):
    protocol, folder = evidence(tmp_path)
    freeze = tmp_path / "freeze.json"
    qualification.freeze_candidate(folder, protocol, freeze)
    protocol.write_text(protocol.read_text() + "\n")
    with pytest.raises(ValueError, match="Protocol differs"):
        qualification.claim_evaluation(freeze, protocol, tmp_path / "out")
    assert not freeze.with_suffix(".evaluation-started.json").exists()
    (folder / "summary.csv").write_text("tampered")
    with pytest.raises(ValueError, match="hash/path"):
        qualification.freeze_candidate(folder, protocol, tmp_path / "second.json")


def test_current_protocol_keeps_single_primary_and_known_limitations():
    spec = json.loads((ROOT / "public-se-qualification-protocol.json").read_text())
    assert spec["methods"][0]["name"] == "gaussian_se_primary"
    assert spec["fields"]["field_subset"] == ["steric", "electrostatic"]
    assert spec["datasets"][0]["test"]["valid"] == 22
    assert "not a protein" in spec["datasets"][0]["family"]
    assert "No SARscope ATA" in spec["interpretation"]["data_status"]
    assert len(provenance.file_sha256(ROOT / "public-se-qualification-protocol.json")) == 64


def test_frozen_public_se_attempt_retains_negative_outcomes_and_sampling_limitations():
    folder = ROOT / "public-se-assessment-20261016-v2"
    if not folder.exists():
        pytest.skip("Frozen local qualification attempt not present.")
    for name, digest in json.loads((folder / "manifest.sha256.json").read_text()).items():
        assert provenance.file_sha256(folder / name) == digest
    result = json.loads((folder / "assessment.json").read_text())
    assert result["freeze_verified_against_current_workflow"]
    assert result["sampling"]["phase_checks_passed"] == 12
    assert result["sampling"]["relative_pose_flags_failed"] == 5
    assert result["cross_environment"]["max_prediction_difference"] < 1e-10
    assert not result["package_certified"]
    assert result["predictive_release_decision"].startswith("NO-GO")
    for row in result["held_out_models"]:
        assert row["test_r2"] < 0
        assert row["test_rmse"] > row["mean_baseline_rmse"]
        assert row["test_rmse"] > row["rf_baseline_rmse"]
        assert not row["passes_predeclared_predictive_thresholds"]
    assert "ATA is now consumed" in result["test_set_status"]

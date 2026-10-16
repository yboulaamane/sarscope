"""Campaign accounting and frozen evidence checks; no live downloads in CI."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_field_qsar import molecule

from sarscope import field_campaign
from sarscope.field_qsar import FieldDataset

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/field_qsar"


def test_partition_audit_refuses_duplicate_graphs_with_different_names():
    molecules = [molecule("CCO"), molecule("CCC"), molecule("OCC")]
    dataset = FieldDataset(
        pd.DataFrame({"molecule_id": ["a", "b", "alias_of_a"]}),
        molecules,
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )
    with pytest.raises(ValueError, match="Duplicate encoded structures"):
        field_campaign.audit_partition(dataset, 2)
    dataset.molecules[2] = molecule("CCN")
    audit = field_campaign.audit_partition(dataset, 2)
    assert audit.split.tolist() == ["train", "train", "held_out"]
    assert audit.graph_sha256.nunique() == 3
    dataset.table.loc[2, "molecule_id"] = "a"
    with pytest.raises(ValueError, match="identifiers"):
        field_campaign.audit_partition(dataset, 2)


def test_campaign_rejects_bad_cv_configuration_and_preserves_prior_evidence(tmp_path):
    protocol = json.loads((ROOT / "native-validation-protocol.json").read_text())
    protocol["model"]["regression_cv"] = "kfold"
    with pytest.raises(ValueError, match="regression_cv"):
        field_campaign.run_campaign(protocol, {}, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    protocol["model"]["regression_cv"] = "default"
    prior = tmp_path / "existing"
    prior.mkdir()
    (prior / "evidence.json").write_text("{}")
    with pytest.raises(FileExistsError, match="retain previous evidence"):
        field_campaign.run_campaign(protocol, {}, prior)
    assert (prior / "evidence.json").read_text() == "{}"


def test_source_failure_accounts_for_every_declared_case_without_certification(tmp_path):
    protocol = json.loads((ROOT / "native-validation-protocol.json").read_text())
    payloads = {
        source["name"]: {"train": b"tampered", "test": b"tampered"}
        for source in protocol["datasets"]
    }
    result = field_campaign.run_campaign(protocol, payloads, tmp_path / "failed")
    assert result["expected_cases"] == result["failed_cases"] == 6
    assert result["completed_cases"] == 0
    assert not result["package_certified"]
    summary = pd.read_csv(tmp_path / "failed/summary.csv")
    assert len(summary) == 6
    assert not summary.completed.any()
    assert summary.error.str.contains("hash differs").all()
    assert summary.stage.eq("input/preparation").all()


def test_frozen_native_campaign_keeps_three_positive_or_negative_results_and_three_abstentions():
    folder = ROOT / "native-validation-20261016-v2"
    if not folder.exists():
        pytest.skip("Local native campaign evidence not present.")
    manifest = json.loads((folder / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    result = json.loads((folder / "assessment.json").read_text())
    assert (result["expected_cases"], result["completed_cases"], result["failed_cases"]) == (
        6,
        3,
        3,
    )
    assert not result["package_certified"]
    assert not result["prospective_or_blinded"]
    assert not result["historical_preprocessing_or_test_mean_correction"]
    assert (
        result["runner_sha256"]
        == hashlib.sha256(Path(field_campaign.__file__).read_bytes()).hexdigest()
    )
    summary = pd.read_csv(folder / "summary.csv")
    assert len(summary) == 6
    assert summary.loc[summary.method.eq("comsia"), "completed"].all()
    assert not summary.loc[summary.method.eq("comfa"), "completed"].any()
    for source in ("AChE", "CCR5", "THR"):
        case = folder / source / "comsia"
        permutations = pd.read_csv(case / "permutations.csv")
        assert len(permutations) == 99
        predictions = pd.read_csv(case / "predictions.csv")
        test = predictions[predictions.split.eq("held_out")]
        observed = test.pactivity.to_numpy()
        predicted = test.prediction.to_numpy()
        expected = 1 - np.sum((observed - predicted) ** 2) / np.sum(
            (observed - observed.mean()) ** 2
        )
        assert summary.loc[
            summary.dataset.eq(source) & summary.method.eq("comsia"), "test_r2"
        ].iloc[0] == pytest.approx(expected)
        checks = pd.read_csv(case / "checks.csv")
        assert not checks.passed.all()
        assert len(pd.read_csv(case / "bootstrap.csv")) == 4

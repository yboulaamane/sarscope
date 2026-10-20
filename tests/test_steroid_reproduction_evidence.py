"""Recalculate the retained paper repetition; no network or upstream program."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/field_qsar/steroid-alignment-reproduction-20261019"


def test_compact_evidence_manifest_and_source_boundary():
    manifest = json.loads((EVIDENCE / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        path = (EVIDENCE / name).resolve()
        assert path.is_relative_to(EVIDENCE.resolve())
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    assert not list(EVIDENCE.rglob("*.sdf"))
    assert not list(EVIDENCE.rglob("*.npz"))


def test_published_internal_metrics_reproduce_but_test_failure_is_retained():
    folder = EVIDENCE / "historical_repeat"
    result = json.loads((folder / "assessment.json").read_text())
    table = pd.read_csv(folder / "predictions.csv")
    train, test = table[table.split == "train"], table[table.split == "test"]
    assert (len(train), len(test)) == (21, 10)
    q2 = r2_score(train.pactivity, train.loo_prediction)
    fit = r2_score(train.pactivity, train.prediction)
    assert abs(q2 - result["q2"]) < 1e-12
    assert abs(fit - result["train_r2"]) < 1e-12
    assert abs(q2 - 0.609) < 0.005 and abs(fit - 0.917) < 0.005
    assert result["selected_components"] == 3
    assert abs(r2_score(test.pactivity, test.prediction) - result["test_r2"]) < 1e-12
    assert result["test_r2"] < 0
    assert not result["independent_predictive_validation"]
    curve = pd.read_csv(folder / "component_search.csv")
    assert curve.loc[curve.selection_q2.idxmax(), "components"] == 3


def test_native_metrics_recalculate_and_all_six_cases_are_reported():
    table = pd.read_csv(EVIDENCE / "summary.csv")
    assert len(table) == 6
    assert set(table.route) == {"prealigned", "rigid_mcs", "constrained"}
    assert set(table.profile) == {"gaussian_SEH", "mmff94_SE"}
    assert len(table[table.status == "alignment_failed"]) == 4
    for row in table[table.status == "ok"].itertuples():
        prediction = pd.read_csv(EVIDENCE / row.route / row.profile / "predictions.csv")
        test = prediction[prediction.split == "held_out"]
        assert len(test) == 10
        assert np.isclose(r2_score(test.pactivity, test.prediction), row.test_r2)
        assert np.isclose(root_mean_squared_error(test.pactivity, test.prediction), row.test_rmse)


def test_accounting_and_frozen_scope_do_not_upgrade_failures_to_validation():
    accounting = pd.read_csv(EVIDENCE / "source_accounting.csv")
    assert len(accounting) == 32
    assert (accounting.status == "retained").sum() == 31
    assert (accounting.status == "unreadable source record").sum() == 1
    protocol = json.loads((EVIDENCE / "protocol.json").read_text())
    assert protocol["all_failures_retained"]
    assert not protocol["predictive_qualification"]
    assert protocol["permutations"] == 0
    assert protocol["reference"]["index"] == 0
    assert protocol["reference"]["molecule_id"] == "aldosterone"
    for profile in protocol["profiles"].values():
        assert profile["component_sample_ratio"] == 5
        assert not profile["preserve_input_hydrogens"]
        assert profile["max_alignment_rmsd"] == 1.0

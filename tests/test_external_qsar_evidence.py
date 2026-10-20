"""Integrity and scope of the frozen external comparison; no external binaries."""

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "benchmarks/alignment/external-comparison-20261019"


def test_external_compact_manifest():
    manifest = json.loads((EVIDENCE / "manifest.sha256.json").read_text())
    for filename, expected in manifest.items():
        assert hashlib.sha256((EVIDENCE / filename).read_bytes()).hexdigest() == expected


def test_external_solver_comparison_covers_both_full_cohorts():
    frame = pd.read_csv(EVIDENCE / "solver_parity.csv").set_index("cohort")
    assert set(frame.index) == {"THR", "AChE"}
    assert frame.loc["THR", "n"] == 88
    assert frame.loc["AChE", "n"] == 111
    assert frame.parity_pass.all()
    assert (frame.max_absolute_prediction_difference < 5e-5).all()
    assert (frame.tolerance == 5e-4).all()


def test_external_alignment_comparison_preserves_all_routes():
    frame = pd.read_csv(EVIDENCE / "model_metrics.csv")
    assert len(frame) == 8
    assert set(frame.route) == {"original", "o3a_mmff", "o3a_crippen", "open3dalign_atom"}
    assert frame.status.eq("ok").all()
    original = frame[frame.route == "original"]
    assert original.original_reproduces.all()
    summary = json.loads((EVIDENCE / "summary.json").read_text())
    assert summary["permutations"] == 0
    assert "not field equivalence" in summary["scope"]
    assert max(summary["rf_prediction_max_changes"].values()) < 1e-12

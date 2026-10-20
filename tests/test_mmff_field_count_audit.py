"""Scalar-count evidence is derived, hash-bound and independent of model fitting."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from benchmarks.audit_mmff_field_counts import build_audit

EVIDENCE = (
    Path(__file__).resolve().parents[1] / "benchmarks/field_qsar/mmff-mif-independent-20261019"
)


def fixture_files(tmp_path, *, shape=(3, 16), train=None):
    array = tmp_path / "case/native_fields.npy"
    array.parent.mkdir()
    np.save(array, np.zeros(shape))
    protocol = {
        "cases": [
            {
                "name": "case",
                "train": [0, 1] if train is None else train,
                "test": [2],
                "external_grid": {"shape": [2, 2, 2]},
                "params": {"field_subset": ["steric", "electrostatic"]},
                "input_sha256": {
                    "native_fields.npy": hashlib.sha256(array.read_bytes()).hexdigest()
                },
            }
        ]
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    (tmp_path / "results.csv").write_text("case,status,n\ncase,ok,3\n")
    return path, array


def test_count_audit_uses_verified_matrix_size(tmp_path):
    protocol, _ = fixture_files(tmp_path)
    audit = build_audit(protocol, tmp_path)
    assert audit["total_scalar_values"] == 48
    assert audit["cases"][0]["matrix_shape"] == [3, 16]
    assert not audit["fields_recomputed"] and not audit["models_refitted"]


def test_count_audit_refuses_replaced_array(tmp_path):
    protocol, array = fixture_files(tmp_path)
    np.save(array, np.ones((3, 16)))
    with pytest.raises(ValueError, match="hash mismatch"):
        build_audit(protocol, tmp_path)


def test_count_audit_refuses_shape_mismatch(tmp_path):
    protocol, _ = fixture_files(tmp_path, shape=(3, 15))
    with pytest.raises(ValueError, match="membership mismatch"):
        build_audit(protocol, tmp_path)


def test_count_audit_refuses_duplicate_partition(tmp_path):
    protocol, _ = fixture_files(tmp_path, train=[0, 0])
    with pytest.raises(ValueError, match="duplicated molecule partition"):
        build_audit(protocol, tmp_path)


def test_paper_count_matches_hash_bound_frozen_grid_dimensions():
    protocol = json.loads((EVIDENCE / "protocol.json").read_text())
    audit = json.loads((EVIDENCE / "scalar-count-audit.json").read_text())
    assert (
        hashlib.sha256((EVIDENCE / "protocol.json").read_bytes()).hexdigest()
        == audit["protocol_sha256"]
    )
    assert (
        hashlib.sha256((EVIDENCE / "results.csv").read_bytes()).hexdigest()
        == audit["results_sha256"]
    )
    with (EVIDENCE / "results.csv").open(newline="") as handle:
        results = {row["case"]: row for row in csv.DictReader(handle)}
    counts = []
    assert len(audit["cases"]) == len(protocol["cases"]) == 4
    for entry, case in zip(audit["cases"], protocol["cases"], strict=True):
        assert entry["case"] == case["name"]
        assert entry["molecules"] == int(results[case["name"]]["n"])
        expected = (
            entry["molecules"]
            * len(case["params"]["field_subset"])
            * np.prod(case["external_grid"]["shape"])
        )
        assert entry["scalar_values"] == int(expected)
        assert entry["native_fields_sha256"] == case["input_sha256"]["native_fields.npy"]
        assert entry["hash_matches_frozen_protocol"]
        counts.append(entry["scalar_values"])
    assert sum(counts) == audit["total_scalar_values"] == 12_607_200

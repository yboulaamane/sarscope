"""The benchmark only runs from a verified fixed, family-diverse snapshot."""

import json

import pytest

from sarscope.benchmark import PROTOCOL, TARGETS, _json_bytes, _sha256, load_frozen, run_benchmark


def _fixture_manifest(tmp_path):
    entries = []
    for target_id, family, name in TARGETS:
        file = f"{target_id}.activities.json"
        data = _json_bytes([{"target_chembl_id": target_id}])
        (tmp_path / file).write_bytes(data)
        entries.append(
            {
                "target_id": target_id,
                "target_name": name,
                "family": family,
                "file": file,
                "sha256": _sha256(data),
                "raw_activity_count": 1,
            }
        )
    from dataclasses import asdict

    manifest = {
        "schema_version": 1,
        "chembl_release": "ChEMBL_test",
        "protocol": asdict(PROTOCOL),
        "targets": entries,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_frozen_panel_accepts_exact_three_family_manifest(tmp_path):
    _fixture_manifest(tmp_path)
    _, entries = load_frozen(tmp_path)
    assert [entry["family"] for entry, _ in entries] == [x[1] for x in TARGETS]


def test_frozen_panel_rejects_hash_drift(tmp_path):
    _fixture_manifest(tmp_path)
    (tmp_path / "CHEMBL217.activities.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_frozen(tmp_path)


def test_frozen_panel_rejects_family_relabeling(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    manifest["targets"][1]["family"] = "protein_kinase"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="panel changed"):
        load_frozen(tmp_path)


def test_frozen_panel_rejects_protocol_drift(tmp_path):
    manifest = _fixture_manifest(tmp_path)
    manifest["protocol"]["seed"] = 9
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="protocol differs"):
        load_frozen(tmp_path)


def test_benchmark_rejects_unregistered_validation(tmp_path):
    with pytest.raises(ValueError, match="validation must be"):
        run_benchmark(tmp_path, tmp_path / "out", validation="random")

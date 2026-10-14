"""Publication evidence cannot silently become an independent or expert benchmark."""

import json
import zipfile
from dataclasses import asdict

import pandas as pd
import pytest

from sarscope.benchmark import PROTOCOL, TARGETS
from sarscope.provenance import file_sha256
from sarscope.publication import (
    benchmark_audit,
    bundle_repository,
    direct_position_pairs,
    evaluate_cases,
    freeze_cases,
    load_cases,
    lock_environment,
    main,
    score_reviews,
)


def frozen_panel(tmp_path):
    snapshot = tmp_path / "frozen"
    snapshot.mkdir()
    entries, cases = [], []
    for index, (target, family, name) in enumerate(TARGETS):
        assay, document = f"ASSAY{index}", f"DOC{index}"
        records = [
            {
                "target_chembl_id": target,
                "assay_chembl_id": assay,
                "document_chembl_id": document,
                "assay_type": "B",
                "standard_type": "IC50",
                "standard_relation": "=",
                "standard_value": value,
                "standard_units": "nM",
                "assay_variant_mutation": None,
                "src_id": 1,
                "activity_id": f"{index}-{n}",
                "molecule_chembl_id": f"M{index}-{n}",
                "canonical_smiles": smiles,
            }
            for n, (smiles, value) in enumerate(
                [
                    ("Cc1ccccc1", 100),
                    ("Fc1ccccc1", 10),
                    ("CCc1ccccc1", 1000),
                ]
            )
        ]
        file = f"{target}.activities.json"
        (snapshot / file).write_text(json.dumps(records))
        entries.append(
            {
                "target_id": target,
                "family": family,
                "target_name": name,
                "file": file,
                "sha256": file_sha256(snapshot / file),
                "raw_activity_count": len(records),
            }
        )
        cases.append(
            {
                "case_id": f"case{index}",
                "target_id": target,
                "assay_id": assay,
                "document_id": document,
                "doi": "synthetic-test-not-a-paper",
            }
        )
    (snapshot / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "chembl_release": "ChEMBL_test",
                "protocol": asdict(PROTOCOL),
                "targets": entries,
            }
        )
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"schema_version": 1, "effect_threshold": 0.5, "cases": cases}))
    return snapshot, protocol


def test_case_freeze_and_offline_evaluation_are_traceable(tmp_path):
    frozen, protocol = frozen_panel(tmp_path)
    cases, result = tmp_path / "cases", tmp_path / "results"
    freeze_cases(frozen, protocol, cases)
    _, cohorts = load_cases(cases)
    assert len(cohorts) == 3
    evaluate_cases(cases, result)
    summary = pd.read_csv(result / "case-summary.csv")
    assert summary.reference_agreement.all()
    assert (summary.curated_molecules == 3).all()
    assert (
        json.loads((result / "evaluation.json").read_text())["expert_validation"] == "not performed"
    )
    assert (result / "review-template.csv").is_file()
    with pytest.raises(ValueError, match="overwrite"):
        evaluate_cases(cases, result)
    (cases / "case0.json").write_text("[]")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_cases(cases)


def test_case_protocol_refuses_path_traversal_and_wrong_document(tmp_path):
    frozen, protocol = frozen_panel(tmp_path)
    config = json.loads(protocol.read_text())
    config["cases"][0]["case_id"] = "../escape"
    protocol.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="safe directory"):
        freeze_cases(frozen, protocol, tmp_path / "cases")
    config["cases"][0]["case_id"] = "case0"
    config["cases"][0]["document_id"] = "wrong"
    protocol.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="mismatched document"):
        freeze_cases(frozen, protocol, tmp_path / "cases")


def test_direct_reference_has_explicit_scope_guards():
    from test_sar_map import data

    series, evidence = data()
    pairs = direct_position_pairs(series, evidence)
    assert pairs.query("position == 'R1'").delta_pactivity.tolist() == [1, 1, 1]
    evidence.loc[0, "assay_chembl_id"] = "other"
    with pytest.raises(ValueError, match="one assay"):
        direct_position_pairs(series, evidence)
    evidence["assay_chembl_id"] = "one"
    evidence.loc[0, "assay_variant_mutation"] = "V600E"
    with pytest.raises(ValueError, match="unannotated"):
        direct_position_pairs(series, evidence)


def test_blank_reviews_are_not_scored_and_missing_consensus_is_not_agreement(tmp_path):
    key, reviews = tmp_path / "key.csv", tmp_path / "reviews.csv"
    pd.DataFrame(
        [
            {"task_id": "T1", "grade": "strong", "tool_decision": "increase"},
            {"task_id": "T2", "grade": "mixed", "tool_decision": "mixed / small change"},
        ]
    ).to_csv(key, index=False)
    pd.DataFrame([{"task_id": "T1", "reviewer_id": "", "decision": ""}]).to_csv(
        reviews, index=False
    )
    with pytest.raises(ValueError, match="incomplete"):
        score_reviews(key, reviews, tmp_path / "result")
    rows = [
        {"task_id": "T1", "reviewer_id": "A", "decision": "insufficient"},
        {"task_id": "T1", "reviewer_id": "B", "decision": "insufficient"},
        {"task_id": "T2", "reviewer_id": "A", "decision": "mixed"},
    ]
    pd.DataFrame(rows).to_csv(reviews, index=False)
    score_reviews(key, reviews, tmp_path / "result")
    frame = pd.read_csv(tmp_path / "result/review-agreement.csv")
    assert bool(frame.iloc[0].unsupported_directional_claim)
    assert frame.iloc[1].consensus == "unresolved"
    assert pd.isna(frame.iloc[1].agreement)
    pd.DataFrame(rows + rows[:1]).to_csv(reviews, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        score_reviews(key, reviews, tmp_path / "other")


def test_bootstrap_is_reproducible_and_counts_acyclic_compounds_separately(tmp_path):
    path = tmp_path / "predictions.csv"
    pd.DataFrame(
        [
            {
                "target_id": target,
                "molecule_id": f"M{i}",
                "murcko": None,
                "pactivity": i + 5,
                "predicted_pactivity": i + 5.2,
                "baseline_pactivity": 6,
            }
            for target, _, _ in TARGETS
            for i in range(3)
        ]
    ).to_csv(path, index=False)
    first = benchmark_audit(path, tmp_path / "one", repeats=100)
    second = benchmark_audit(path, tmp_path / "two", repeats=100)
    assert first.read_bytes() == second.read_bytes()
    assert (pd.read_csv(first).scaffold_clusters == 3).all()
    assert "not repeated training" in json.loads((first.parent / "scope.json").read_text())["scope"]
    with pytest.raises(ValueError, match=">=100"):
        benchmark_audit(path, tmp_path / "bad", repeats=1)


def test_local_bundle_hashes_include_uncommitted_source_but_not_ignored_secrets(tmp_path):
    import subprocess

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / ".gitignore").write_text(".env\n")
    subprocess.run(["git", "add", ".gitignore"], cwd=repository, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Synthetic Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "Test fixture",
        ],
        cwd=repository,
        check=True,
    )
    (repository / ".env").write_text("private")
    (repository / "example.py").write_text("# synthetic source\n")
    archive = bundle_repository(repository, tmp_path / "archive.zip")
    with zipfile.ZipFile(archive) as zipped:
        assert "sarscope/.env" not in zipped.namelist()
        manifest = json.loads(zipped.read("sarscope/ARCHIVE-MANIFEST.json"))
        assert manifest["includes_uncommitted_files"]
        assert next(row for row in manifest["files"] if row["file"] == "example.py")[
            "sha256"
        ] == file_sha256(repository / "example.py")
    with pytest.raises(ValueError, match="new archive"):
        bundle_repository(repository, archive)


def test_cli_reports_integrity_error_without_fetching(tmp_path, capsys):
    assert main(["evaluate", "--cases", str(tmp_path), "--out", str(tmp_path / "out")]) == 1
    assert "error:" in capsys.readouterr().err


def test_environment_lock_has_archive_hashes_and_never_copies_private_urls(tmp_path):
    report, lock = tmp_path / "report.json", tmp_path / "requirements.lock"
    payload = {
        "environment": {
            "python_full_version": "3.13.13",
            "sys_platform": "linux",
            "platform_machine": "x86_64",
        },
        "install": [
            {
                "metadata": {"name": "example", "version": "1.0"},
                "download_info": {
                    "url": "https://secret@example.invalid/file.whl",
                    "archive_info": {"hashes": {"sha256": "a" * 64}},
                },
            }
        ],
    }
    report.write_text(json.dumps(payload))
    lock_environment(report, lock)
    assert "example==1.0 --hash=sha256:" in lock.read_text()
    assert "secret" not in lock.read_text()
    with pytest.raises(ValueError, match="overwrite"):
        lock_environment(report, lock)
    payload["install"][0]["download_info"]["archive_info"]["hashes"] = {}
    report.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="SHA-256"):
        lock_environment(report, tmp_path / "missing-hash.lock")


def test_recorded_commands_carry_no_machine_specific_path():
    """Published artifacts must not leak the author's filesystem.

    The commands are run with sys.executable, but recording that absolute path
    would put a home directory into the repository without making the record
    more reproducible.
    """
    import sys

    from sarscope.publication import _portable_command

    command = [sys.executable, "-m", "mmpdblib", "fragment", "paper/x/input.smi"]
    recorded = _portable_command(command)
    assert recorded[0] == "python"
    assert recorded[1:] == command[1:]
    assert sys.executable not in recorded


def test_committed_mmpdb_commands_are_portable():
    """Guards the checked-in comparison artifacts, not just the generator."""
    import json
    from pathlib import Path

    files = sorted(Path("paper/mmpdb-comparison").glob("*/commands.json"))
    if not files:
        pytest.skip("comparison artifacts are not present")
    for path in files:
        for command in json.loads(path.read_text()):
            assert command[0] == "python", f"{path} records an interpreter path"
            for part in command:
                assert not part.startswith("/"), f"{path} has an absolute path: {part}"

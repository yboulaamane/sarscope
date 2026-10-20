"""Offline integrity and retrieval checks for the published paper datasets."""

import csv
import hashlib
import io
import json
import math
import re
import tarfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import pytest

from examples.paper_data import fetch_3d_inputs as fetcher

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples/paper_data"


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_paper_data_manifest_covers_every_published_file():
    manifest = json.loads((DATA / "manifest.sha256.json").read_text())
    paths = {
        str(p.relative_to(DATA)): p
        for p in DATA.rglob("*")
        if p.is_file() and "__pycache__" not in p.parts and p.name != "manifest.sha256.json"
    }
    assert set(manifest) == set(paths)
    for name, path in paths.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[name], name
    assert sum(p.stat().st_size for p in paths.values()) < 6 * 1024 * 1024
    assert not any(p.suffix in {".sdf", ".npz", ".pdf", ".docx"} for p in paths.values())


def test_paper_example_documentation_links_resolve():
    for path in [ROOT / "examples/README.md", *DATA.rglob("*.md")]:
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text()):
            if not link.startswith(("https://", "http://", "#")):
                assert (path.parent / link.split("#")[0]).exists(), (path, link)


@pytest.mark.parametrize(
    "case,raw_count,curated_count,selected_count,pair_count",
    [("braf", 46, 44, 16, 29), ("drd2", 34, 30, 23, 36), ("esr1", 107, 98, 53, 198)],
)
def test_assay_examples_retain_final_scientific_selections(
    case, raw_count, curated_count, selected_count, pair_count
):
    raw = DATA / "sar/raw" / f"{case}.json"
    manifest = json.loads((raw.parent / "manifest.json").read_text())
    entry = next(c for c in manifest["cases"] if c["case_id"] == case)
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == entry["sha256"]
    assert len(json.loads(raw.read_text())) == raw_count
    compounds = rows(DATA / "sar" / case / "compounds.csv")
    selected = rows(DATA / "sar" / case / "selected_series.csv")
    pairs = rows(DATA / "sar" / case / "positional_pairs.csv")
    assert len(compounds) == curated_count
    assert len(selected) == selected_count
    assert len(pairs) == pair_count
    selected_ids = {r["molecule_id"] for r in selected}
    assert selected_ids == {r["molecule_id"] for r in compounds if r["selected_series"] == "True"}
    for pair in pairs:
        assert {pair["id_from"], pair["id_to"]} <= selected_ids
        assert (
            abs(
                float(pair["pactivity_to"])
                - float(pair["pactivity_from"])
                - float(pair["delta_pactivity"])
            )
            < 1e-12
        )
    if case == "drd2":
        assert not {"CHEMBL125455", "CHEMBL126493", "CHEMBL2109988"} & selected_ids


def test_2d_original_records_predictions_and_reconstructed_splits_match():
    directory = DATA / "qsar2d"
    predictions = rows(directory / "held_out_predictions.csv")
    metrics = rows(directory / "target_metrics.csv")
    prefix = "runs/benchmark-chembl37/"
    with tarfile.open(directory / "benchmark-chembl37.tar.gz") as archive:
        manifest_bytes = archive.extractfile(prefix + "frozen/manifest.json").read()
        assert hashlib.sha256(manifest_bytes).hexdigest() == (
            "f48c8aec796c8986de3938febbc95b99149355660e1c31315f98852e3395a493"
        )
        manifest = json.loads(manifest_bytes)
        for name in ("held_out_predictions.csv", "target_metrics.csv", "summary.json"):
            assert (
                archive.extractfile(prefix + "results/" + name).read()
                == (directory / name).read_bytes()
            )
        for entry, expected in zip(
            manifest["targets"], [(2917, 2333, 584), (492, 393, 99), (3258, 2607, 651)], strict=True
        ):
            raw = archive.extractfile(prefix + "frozen/" + entry["file"]).read()
            assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
            assert len(json.loads(raw)) == entry["raw_activity_count"]
            tid = entry["target_id"]
            compounds = rows(directory / f"{tid}_compounds_and_split.csv")
            counts = Counter(r["split"] for r in compounds)
            assert (len(compounds), counts["train"], counts["held_out"]) == expected
            held_out = {r["molecule_id"]: r for r in compounds if r["split"] == "held_out"}
            saved = [r for r in predictions if r["target_id"] == tid]
            assert set(held_out) == {r["molecule_id"] for r in saved}
            for record in saved:
                exported = held_out[record["molecule_id"]]
                assert exported["smiles"] == record["smiles"]
                assert abs(float(exported["pactivity"]) - float(record["pactivity"])) < 1e-12
            scaffolds = {
                split: {r["murcko"] for r in compounds if r["split"] == split} - {""}
                for split in ("train", "held_out")
            }
            assert not scaffolds["train"] & scaffolds["held_out"]
            truth = [float(r["pactivity"]) for r in saved]
            error = sum(
                (float(r["predicted_pactivity"]) - float(r["pactivity"])) ** 2 for r in saved
            )
            mean = sum(truth) / len(truth)
            score = next(r for r in metrics if r["target_id"] == tid)
            assert abs(math.sqrt(error / len(truth)) - float(score["test_rmse"])) < 1e-12
            assert abs(
                1 - error / sum((y - mean) ** 2 for y in truth) - float(score["test_r2"])
            ) < (1e-12)


@pytest.mark.parametrize(
    "label,train,test,test_label",
    [("THR", 59, 29, "held_out"), ("AChE", 74, 37, "held_out"), ("STEROIDS", 21, 10, "test")],
)
def test_3d_memberships_keep_historical_partitions(label, train, test, test_label):
    members = rows(DATA / "qsar3d" / f"{label}_source_membership.csv")
    assert Counter(r["split"] for r in members) == {"train": train, test_label: test}
    assert len({r["molecule_id"] for r in members}) == train + test
    assert all(math.isfinite(float(r["pactivity"])) for r in members)
    if label == "THR":
        assert "01" in {r["molecule_id"] for r in members}


def fake_archive(tmp_path):
    payloads = {"THR": b"original thrombin coordinates", "AChE": b"original ache coordinates"}
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for label, data in payloads.items():
            member = tarfile.TarInfo(f"{label}.sd")
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    outer = io.BytesIO()
    with ZipFile(outer, "w") as archive:
        archive.writestr("input.tar.gz", buffer.getvalue())
    payload = outer.getvalue()
    spec = [
        {
            "dataset": label,
            "archive_url": "https://example.org/original.zip",
            "archive_sha256": hashlib.sha256(payload).hexdigest(),
            "nested_tar": "input.tar.gz",
            "member": f"{label}.sd",
            "member_sha256": hashlib.sha256(data).hexdigest(),
        }
        for label, data in payloads.items()
    ]
    path = tmp_path / "sources.json"
    path.write_text(json.dumps(spec))
    return path, payload, payloads


def test_verified_retrieval_reuses_shared_archive_and_preserves_bytes(tmp_path, monkeypatch):
    spec, archive, original = fake_archive(tmp_path)
    calls = []

    def download(url, digest):
        calls.append((url, digest))
        return archive

    monkeypatch.setattr(fetcher, "fetch_bytes", download)
    result = fetcher.verified_inputs(("THR", "AChE"), spec)
    assert len(calls) == 1
    assert result == {f"{k}_original.sdf": v for k, v in original.items()}
    out = tmp_path / "output"
    paths = fetcher.write_inputs(out, result)
    assert fetcher.write_inputs(out, result) == paths
    assert {p.name: p.read_bytes() for p in paths} == result


def test_verified_retrieval_refuses_molecular_hash_drift(tmp_path, monkeypatch):
    spec, archive, _ = fake_archive(tmp_path)
    source = json.loads(spec.read_text())
    source[0]["member_sha256"] = "0" * 64
    spec.write_text(json.dumps(source))
    monkeypatch.setattr(fetcher, "fetch_bytes", lambda *_: archive)
    with pytest.raises(ValueError, match="input hash differs"):
        fetcher.verified_inputs(("THR",), spec)


def test_download_hash_and_size_checks_are_offline(monkeypatch):
    payload = b"source data"
    monkeypatch.setattr(fetcher, "urlopen", lambda *_args, **_kwargs: io.BytesIO(payload))
    assert fetcher.fetch_bytes(
        "https://example.org/input", hashlib.sha256(payload).hexdigest()
    ) == (payload)
    with pytest.raises(ValueError, match="Source hash differs"):
        fetcher.fetch_bytes("https://example.org/input", "0" * 64)
    monkeypatch.setattr(fetcher, "MAX_BYTES", 4)
    with pytest.raises(ValueError, match="size limit"):
        fetcher.fetch_bytes("https://example.org/input", "0" * 64)


def test_write_preflights_all_files_and_refuses_overwrite(tmp_path):
    (tmp_path / "B.sdf").write_bytes(b"existing")
    with pytest.raises(FileExistsError, match="different data"):
        fetcher.write_inputs(tmp_path, {"A.sdf": b"new A", "B.sdf": b"new B"})
    assert not (tmp_path / "A.sdf").exists()
    assert (tmp_path / "B.sdf").read_bytes() == b"existing"
    with pytest.raises(ValueError, match="plain filename"):
        fetcher.write_inputs(tmp_path, {"../outside.sdf": b"data"})

"""Frozen, attributed ChEMBL example. Loading never fetches, fits or generates."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from importlib.resources import files

import pandas as pd

from sarscope.field_qsar import FieldDataset, dataset_from_upload

EXAMPLE_FILES = (
    "egfr_ic50.csv",
    "measurements.csv",
    "source.json",
    "verification.json",
    "README.md",
)


def example_files() -> dict[str, bytes]:
    root = files("sarscope").joinpath("data", "analogue_design")
    contents = {name: root.joinpath(name).read_bytes() for name in EXAMPLE_FILES}
    source = json.loads(contents["source.json"])
    for name in ("egfr_ic50.csv", "measurements.csv"):
        if hashlib.sha256(contents[name]).hexdigest() != source["files_sha256"][name]:
            raise ValueError(f"Bundled example checksum mismatch: {name}.")
    return contents


def example_dataset() -> FieldDataset:
    contents = example_files()
    source = json.loads(contents["source.json"])
    raw = pd.read_csv(io.BytesIO(contents["egfr_ic50.csv"]))
    dataset = dataset_from_upload(
        raw,
        [],
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="IC50",
        source_sha256=source["files_sha256"]["egfr_ic50.csv"],
    )
    dataset.measurements = pd.read_csv(
        io.BytesIO(contents["measurements.csv"]), dtype={"record_id": str}
    )
    dataset.source = {**source, "canonical_tautomer": False}
    return dataset


def example_archive() -> bytes:
    contents = example_files()
    contents["manifest.sha256.json"] = json.dumps(
        {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()}, indent=2
    ).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in contents.items():
            archive.writestr(name, data)
    return stream.getvalue()

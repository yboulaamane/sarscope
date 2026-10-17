"""Offline, measured-compound example specifically for the 3D workflow."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from importlib.resources import files

import pandas as pd

from sarscope.field_qsar import FieldDataset, dataset_from_upload, upload_table


def field_example_files() -> dict[str, bytes]:
    root = files("sarscope").joinpath("data", "field_example")
    names = (
        "egfr_3d.csv",
        "egfr_3d_aligned.sdf",
        "measurements.csv",
        "source.json",
        "verification.json",
        "alignment.csv",
        "README.md",
    )
    contents = {name: root.joinpath(name).read_bytes() for name in names}
    metadata = json.loads(contents["source.json"])
    for name in ("egfr_3d.csv", "egfr_3d_aligned.sdf", "measurements.csv"):
        if hashlib.sha256(contents[name]).hexdigest() != metadata["files_sha256"][name]:
            raise ValueError(f"3D example checksum mismatch: {name}")
    return contents


def field_example_dataset() -> FieldDataset:
    contents = field_example_files()
    raw, molecules = upload_table(contents["egfr_3d_aligned.sdf"], ".sdf")
    source = json.loads(contents["source.json"])
    dataset = dataset_from_upload(
        raw,
        molecules,
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="IC50",
        source_sha256=source["files_sha256"]["egfr_3d_aligned.sdf"],
    )
    dataset.measurements = pd.read_csv(
        io.BytesIO(contents["measurements.csv"]), dtype={"record_id": str}
    )
    dataset.source = source
    return dataset


def field_example_archive() -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in field_example_files().items():
            archive.writestr(name, content)
    return stream.getvalue()

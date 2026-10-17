"""Offline, measured-compound example specifically for the 3D workflow."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from importlib.resources import files

import pandas as pd

from sarscope.field_qsar import FieldDataset, dataset_from_upload, upload_table


@dataclass(frozen=True)
class FieldExample:
    label: str
    folder: str
    csv: str
    sdf: str
    extra_files: tuple[str, ...] = ()


FIELD_EXAMPLES = {
    "ccr5": FieldExample(
        "CCR5 · 75 measured compounds · positive within-series prediction",
        "field_ccr5_example",
        "ccr5_3d.csv",
        "ccr5_3d_aligned.sdf",
        ("preparation.csv", "predictions.csv", "LICENSE.txt"),
    ),
    "egfr": FieldExample(
        "EGFR · 50 measured compounds · weak-model limitation case",
        "field_example",
        "egfr_3d.csv",
        "egfr_3d_aligned.sdf",
    ),
}


def field_example_files(name: str = "ccr5") -> dict[str, bytes]:
    if name not in FIELD_EXAMPLES:
        raise ValueError("Choose a registered measured 3D example: ccr5 or egfr.")
    spec = FIELD_EXAMPLES[name]
    root = files("sarscope").joinpath("data", spec.folder)
    names = (
        spec.csv,
        spec.sdf,
        "measurements.csv",
        "source.json",
        "verification.json",
        "alignment.csv",
        "README.md",
        *spec.extra_files,
    )
    contents = {name: root.joinpath(name).read_bytes() for name in names}
    metadata = json.loads(contents["source.json"])
    for filename, digest in metadata["files_sha256"].items():
        if filename not in contents or hashlib.sha256(contents[filename]).hexdigest() != digest:
            raise ValueError(f"3D example checksum mismatch: {filename}")
    return contents


def field_example_dataset(name: str = "ccr5") -> FieldDataset:
    contents = field_example_files(name)
    spec = FIELD_EXAMPLES[name]
    raw, molecules = upload_table(contents[spec.sdf], ".sdf")
    source = json.loads(contents["source.json"])
    dataset = dataset_from_upload(
        raw,
        molecules,
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint=source.get("endpoint", "IC50"),
        source_sha256=source["files_sha256"][spec.sdf],
    )
    dataset.measurements = pd.read_csv(
        io.BytesIO(contents["measurements.csv"]), dtype={"record_id": str, "molecule_id": str}
    )
    dataset.source = source
    return dataset


def field_example_archive(name: str = "ccr5") -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename, content in field_example_files(name).items():
            archive.writestr(filename, content)
    return stream.getvalue()

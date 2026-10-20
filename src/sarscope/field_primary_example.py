"""Opt-in, hash-verified primary THR example; no chemical data bundled or repaired.

The publisher's supporting data are CC BY-NC4.0, separate from the code
licence. Loading/download and modelling are separate user actions. Recorded
scores are regression-test expectations, never substitute model predictions.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
import time
from pathlib import Path
from zipfile import ZipFile

import httpx
from rdkit import Chem

from sarscope.field_qsar import FieldDataset, dataset_from_upload, upload_table
from sarscope.params import FieldQsarParams

EXAMPLE_ID = "primary_thrombin"
URL = "https://ndownloader.figshare.com/files/5158327"
ARCHIVE_SHA256 = "e69edbb64048a243db5bce8700b198ce9a60b6f8fe5c87f30d863d1f99790ecf"
SDF_SHA256 = "7e91820c51e6b9bb62e9b542cbc906cb0555cb461033fd1226f35d9ae9c1efff"
MAX_DOWNLOAD_BYTES = 4 * 1024 * 1024
TRAIN_IDS = (
    "42 64 27 17 86 48 06 13 12 63 76 38 41 04 53 84 85 88 75 83 22 08 28 03 40 "
    "47 15 43 25 36 24 18 50 02 26 23 33 66 68 59 45 87 34 80 37 57 32 19 78 09 "
    "79 73 62 74 60 69 55 52 77"
).split()
TEST_IDS = (
    "29 20 05 65 07 51 11 30 10 49 61 44 82 46 31 67 81 39 72 21 56 54 14 58 16 71 01 70 35".split()
)
RECORDED_SCORES = {
    "cv_pooled_q2": 0.6070519813940478,
    "test_r2": 0.5071821301111985,
    "test_rmse": 0.771570968908504,
}
RECORDED_MMFF_SCORES = {
    "cv_pooled_q2": 0.612159452062688,
    "test_r2": 0.6298298690930271,
    "test_rmse": 0.6687031071765973,
}


def primary_mmff_settings() -> FieldQsarParams:
    """One reproducible, explicit worked-example profile; not test-driven tuning."""
    return FieldQsarParams(
        method="comfa",
        field_definition="mmff94_mif",
        field_subset=("steric", "electrostatic"),
        charge_model="mmff94",
        minimum_distance=0.0001,
        electrostatic_cutoff=30.0,
        spacing=1.0,
        field_scaling="block_std",
        component_selection="max_q2",
        alignment="prealigned",
        assess_prealigned_core=False,
        preserve_input_hydrogens=True,
        permutations=99,
        permutation_timeout=600,
        max_grid_points=20000,
        max_memory_mb=512,
        seed=42,
    )


def fetch_primary_archive() -> bytes:
    """Explicit, bounded download; no side-effect on module import or page load."""
    chunks, size = [], 0
    deadline = time.monotonic() + 60
    with httpx.stream("GET", URL, timeout=30, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if time.monotonic() > deadline:
                raise ValueError("Primary-data download exceeded its 60-second total budget.")
            size += len(chunk)
            if size > MAX_DOWNLOAD_BYTES:
                raise ValueError("Primary-data download exceeds its 4 MiB limit.")
            chunks.append(chunk)
    payload = b"".join(chunks)
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Primary archive hash differs; source review required, no model loaded.")
    return payload


def primary_thrombin_sdf(payload: bytes) -> bytes:
    """Read only the named data member, retain all source states/activities/poses."""
    if len(payload) > MAX_DOWNLOAD_BYTES or hashlib.sha256(payload).hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Primary archive hash/size differs; stop and re-audit.")
    with ZipFile(io.BytesIO(payload)) as archive:
        nested = archive.read("qsar_sets/3dqsar_sd.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(nested), mode="r:gz") as tar:
        stream = tar.extractfile("./3dqsar_sd/thr_3dqsar.sd")
        if stream is None:
            raise ValueError("Primary thrombin source member missing.")
        source = stream.read()
    if hashlib.sha256(source).hexdigest() != SDF_SHA256:
        raise ValueError("Primary thrombin SDF hash differs.")
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(source), removeHs=False))
    if len(molecules) != 88 or any(mol is None for mol in molecules):
        raise ValueError("Account for all 88 original compounds; no exclusions were made.")
    by_id = {mol.GetProp("_Name"): mol for mol in molecules}
    if len(by_id) != 88 or set(by_id) != set(TRAIN_IDS + TEST_IDS):
        raise ValueError("Primary compound identifiers differ from the frozen order.")
    output = io.StringIO()
    writer = Chem.SDWriter(output)
    for ids, membership, code in ((TRAIN_IDS, "train", "1"), (TEST_IDS, "held_out", "2")):
        for identifier in ids:
            mol = Chem.Mol(by_id[identifier])
            if mol.GetProp("SET") != code:
                raise ValueError("Primary source SET membership differs from the frozen split.")
            mol.SetProp("molecule_id", identifier)
            mol.SetProp("pactivity", mol.GetProp("ACTIVITY"))
            mol.SetProp("source_split", membership)
            writer.write(mol)
    writer.close()
    return output.getvalue().encode()


def primary_thrombin_dataset(payload: bytes) -> FieldDataset:
    sdf = primary_thrombin_sdf(payload)
    raw, molecules = upload_table(sdf, ".sdf")
    result = dataset_from_upload(
        raw,
        molecules,
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="Deposited thrombin log activity",
        source_sha256=hashlib.sha256(sdf).hexdigest(),
        preserve_input_states=True,
        split_col="source_split",
    )
    result.source.update(
        {
            "kind": "fetched_primary_measured_3d_example",
            "example_id": EXAMPLE_ID,
            "target_name": "Thrombin",
            "reference_id": TRAIN_IDS[0],
            "compounds": 88,
            "primary_archive_sha256": ARCHIVE_SHA256,
            "primary_sdf_sha256": SDF_SHA256,
            "download_url": URL,
            "publication": "https://doi.org/10.1021/jm0497141",
            "original_publication": "https://doi.org/10.1021/jm981062r",
            "attribution": "Böhm et al. (1999); original aligned THR data deposited by "
            "Sutherland et al. (2004)",
            "license": "CC-BY-NC-4.0 (publisher supporting-data metadata)",
            "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "assay_comparability": "Historical log endpoint; not independently assay-adjudicated",
            "conversion": "Deposited ACTIVITY values unchanged; no inferred pIC50 conversion",
            "design_comparability_review_required": True,
            "independent_validation": False,
            "coordinate_assumption": "Deposited prealignment, not verified crystal binding poses",
            "preparation": "Original closed-shell states/explicit hydrogens retained; "
            "no neutralisation, tautomerisation or aggregation",
        }
    )
    return result


def archive_from_file(path: Path) -> FieldDataset:
    """Offline entry point for the same reader used by Streamlit/tests."""
    return primary_thrombin_dataset(path.read_bytes())

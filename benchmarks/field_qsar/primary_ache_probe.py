"""Reproduce the primary-source AChE characterisation, without guessed repairs.

This is an offline retrospective experiment, not a fresh/blinded test or a
change to browser limits. Source data retain their separate CC BY-NC4.0 terms.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import httpx
from rdkit import Chem

from sarscope.field_campaign import run_campaign
from sarscope.field_source_quality import require_closed_shell
from sarscope.params import FieldQsarParams

ARCHIVE_HASH = "e69edbb64048a243db5bce8700b198ce9a60b6f8fe5c87f30d863d1f99790ecf"
SDF_HASH = "218ff88d34ab638a949c5b5863116c146c376396ec6b174be929f77954425965"
URL = "https://ndownloader.figshare.com/files/5158327"


def primary_archive(path: Path, download: bool = False) -> bytes:
    if path.exists():
        payload = path.read_bytes()
    elif download:
        response = httpx.get(URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        payload = response.content
        if hashlib.sha256(payload).hexdigest() != ARCHIVE_HASH:
            raise ValueError("Downloaded archive differs from the frozen primary data.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
    else:
        raise FileNotFoundError("Supply the primary ZIP, or explicitly pass --download.")
    if hashlib.sha256(payload).hexdigest() != ARCHIVE_HASH:
        raise ValueError("Primary archive hash differs; stop and re-audit.")
    return payload


def run(archive: Path, out: Path, download: bool = False) -> None:
    with ZipFile(io.BytesIO(primary_archive(archive, download))) as zipped:
        nested = zipped.read("qsar_sets/3dqsar_sd.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(nested), mode="r:gz") as tar:
        stream = tar.extractfile("./3dqsar_sd/ache_3dqsar.sd")
        if stream is None:
            raise ValueError("Primary AChE SDF is missing.")
        raw = stream.read()
    if hashlib.sha256(raw).hexdigest() != SDF_HASH:
        raise ValueError("Primary AChE SDF hash differs.")
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(raw), removeHs=False))
    if len(molecules) != 111 or any(m is None for m in molecules):
        raise ValueError("Account for all 111 source compounds; no silent exclusions.")
    identifiers = [m.GetProp("_Name") for m in molecules]
    require_closed_shell(molecules, identifiers)
    if (
        len(set(identifiers)) != 111
        or len({Chem.MolToSmiles(Chem.RemoveHs(m)) for m in molecules}) != 111
    ):
        raise ValueError("Grouped/source adjudication required for duplicate entries.")
    if any(
        s.specified != Chem.StereoSpecified.Specified
        for m in molecules
        for s in Chem.FindPotentialStereo(Chem.RemoveHs(m))
    ):
        raise ValueError("Source stereochemistry needs review; no states were inferred.")
    if any(m.GetProp("SET") not in ("1", "2") for m in molecules):
        raise ValueError("Unexpected source partition; no censored/other rows discarded.")
    payloads, metadata = {}, {}
    for name, code, count in (("train", "1", 74), ("test", "2", 37)):
        output = io.StringIO()
        writer = Chem.SDWriter(output)
        selected = [m for m in molecules if m.GetProp("SET") == code]
        if len(selected) != count:
            raise ValueError("Source partition size differs.")
        for mol in selected:
            mol.SetDoubleProp("Activity", float(mol.GetProp("ACTIVITY")))
            writer.write(mol)
        writer.close()
        payload = output.getvalue().encode()
        payloads[name] = payload
        metadata[name] = {
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
            "valid": count,
            "invalid": 0,
        }
    settings = FieldQsarParams(
        alignment="prealigned",
        assess_prealigned_core=False,
        preserve_input_hydrogens=True,
        field_subset=("steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor"),
        spacing=1.0,
        field_scaling="block_std",
        component_selection="max_q2",
        permutations=99,
        permutation_timeout=600,
        max_grid_points=80000,
        max_memory_mb=768,
        max_molecules=150,
        seed=42,
    )
    protocol = {
        "schema": 1,
        "purpose": "Additional primary-source family; unchanged THR primary numerical profile",
        "source": {
            "doi": "10.1021/jm0497141",
            "supplement_doi": "10.1021/jm0497141.s002",
            "archive_sha256": ARCHIVE_HASH,
            "source_sdf_sha256": SDF_HASH,
            "license": "CC-BY-NC-4.0, publisher Figshare metadata",
            "download_url": URL,
            "coordinate_policy": "All original coordinates/activities and SET membership retained",
        },
        "selection": "Retrospective, not blinded: previous AChE results were already known. "
        "Current numerical profile fixed before its scores; no test-driven replacement.",
        "resource_retry": "Recorded first run stopped before fitting at a conservative "
        "573 MB estimate. Memory ceiling alone increased from 512 to 768 MB. "
        "111 compounds evaluated offline; browser limits unchanged.",
        "metadata_correction": "The first run used the broad family label carboxylesterase; "
        "this driver labels AChE cholinesterase. No numerical change.",
        "datasets": [
            {
                "name": "AChE",
                "family": "cholinesterase",
                "activity": "Activity",
                "endpoint": "Historical acetylcholinesterase log activity; values unchanged; "
                "assay homogeneity not independently established",
                **metadata,
            }
        ],
        "fields": asdict(settings),
        "methods": [
            {"name": "gaussian_fivefield_primary", "method": "comsia", "filter_mode": "percentile"}
        ],
        "model": {"split": "random", "cv_folds": 3, "regression_cv": "default", "seed": 42},
        "bootstrap_repeats": 1000,
        "interpretation": {
            "scope": "Historical family characterisation, not prospective or universal validation",
            "browser_limit": "Offline memory/compound ceilings differ; no source rows sampled",
        },
    }
    run_campaign(protocol, {"AChE": payloads}, out)
    # Include this adapter in the final integrity record, not just package files.
    driver_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out / "driver_sha256.txt").write_text(driver_hash + "\n")
    manifest = {
        str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "manifest.sha256.json"
    }
    (out / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--download", action="store_true", help="Fetch hash-verified source if absent"
    )
    args = parser.parse_args()
    run(args.original_archive, args.out, args.download)

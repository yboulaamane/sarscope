"""Isolated Meeko preparation and fpocket execution; no imports on the landing page."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import warnings
import zipfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sarscope.docking import DockingError, DockingSettings, _atoms, validate_receptor_pair
from sarscope.receptor import box_from_coordinates

# Official conda-forge distribution, not an unversioned executable download.
FPOCKET_PACKAGE = "https://conda.anaconda.org/conda-forge/linux-64/fpocket-4.2.2-h7b35b64_0.conda"
FPOCKET_PACKAGE_SHA256 = "c3420b1b2665341ca8af88cf4faf5107c1cca215532c3e13fa3424394c0e418f"
FPOCKET_BINARY_SHA256 = "a831cfa0d9ec3a084af34c52548f1ad6fe60e1d1d6c16e26848c60d4015bc806"


def _phase(directory: Path, message: str) -> None:
    (directory / "progress.json").write_text(json.dumps({"phase": message}))


def _install_fpocket() -> tuple[str, dict[str, str]]:
    configured = os.environ.get("SARSCOPE_FPOCKET_BINARY")
    existing = shutil.which(configured or "fpocket")
    if configured and not existing:
        raise DockingError("SARSCOPE_FPOCKET_BINARY does not identify an executable.")
    if existing:
        return existing, {
            "distribution": "User/system fpocket",
            "binary_sha256": hashlib.sha256(Path(existing).read_bytes()).hexdigest(),
        }
    if platform.system() != "Linux" or platform.machine() not in {"x86_64", "amd64"}:
        raise DockingError(
            "Automatic fpocket setup needs Linux x86_64; set SARSCOPE_FPOCKET_BINARY otherwise."
        )
    import httpx
    import zstandard

    cache = (
        Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
        / "sarscope"
        / "fpocket-4.2.2"
    )
    cache.mkdir(parents=True, exist_ok=True)
    binary = cache / "fpocket"
    identity = {
        "distribution": "conda-forge fpocket 4.2.2 h7b35b64_0",
        "package_url": FPOCKET_PACKAGE,
        "package_sha256": FPOCKET_PACKAGE_SHA256,
        "binary_sha256": FPOCKET_BINARY_SHA256,
    }
    if (
        binary.is_file()
        and hashlib.sha256(binary.read_bytes()).hexdigest() == FPOCKET_BINARY_SHA256
    ):
        return str(binary), identity
    content = bytearray()
    with httpx.stream("GET", FPOCKET_PACKAGE, timeout=20, follow_redirects=True) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            content.extend(chunk)
            if len(content) > 4 * 1024 * 1024:
                raise DockingError("Pocket-engine package exceeds its expected download limit.")
    if hashlib.sha256(content).hexdigest() != FPOCKET_PACKAGE_SHA256:
        raise DockingError("Pocket-engine package checksum mismatch; refusing to execute it.")
    executable = None
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        for member in archive.namelist():
            if not member.endswith(".tar.zst"):
                continue
            with zstandard.ZstdDecompressor().stream_reader(
                io.BytesIO(archive.read(member))
            ) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as payload:
                    for item in payload:
                        wanted = item.name == "bin/fpocket" or item.name.startswith(
                            "info/licenses/"
                        )
                        if not wanted or not item.isfile():
                            continue
                        if item.size > 8 * 1024 * 1024:
                            raise DockingError("Unexpected engine archive member size.")
                        stream = payload.extractfile(item)
                        assert stream is not None
                        data = stream.read()
                        if item.name == "bin/fpocket":
                            executable = data
                        else:
                            (cache / ("license_" + Path(item.name).name)).write_bytes(data)
    if executable is None or hashlib.sha256(executable).hexdigest() != FPOCKET_BINARY_SHA256:
        raise DockingError("Pocket-engine binary checksum mismatch.")
    temporary = cache / "fpocket.partial"
    temporary.write_bytes(executable)
    temporary.chmod(0o755)
    temporary.replace(binary)
    return str(binary), identity


def _prepare(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates

    _phase(directory, "Preparing selected protein chains with Meeko residue templates…")
    original = _atoms(request["protein_pdb"], pdbqt=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            polymer = Polymer.from_pdb_string(
                request["protein_pdb"],
                chem_templates=ResidueChemTemplates.create_from_defaults(),
                mk_prep=MoleculePreparation(),
                allow_bad_res=False,
            )
            receptor, flexible = PDBQTWriterLegacy.write_from_polymer(polymer)
            protein = polymer.to_pdb()
        except Exception as exc:
            raise DockingError(
                f"Meeko preparation failed: {exc}. "
                "Missing/unsupported residues are not silently deleted. "
                "Try another structure or use the prepared-upload workflow."
            ) from exc
    if flexible or _atoms(protein, pdbqt=False).keys() != original.keys():
        raise DockingError("Meeko changed the protein heavy-atom set or created flexible residues.")
    atoms = _atoms(protein, pdbqt=False)
    center = next(iter(atoms.values()))
    validate_receptor_pair(protein, receptor, DockingSettings((center[0], center[1], center[2])))
    shifts = [
        sum((v - u) ** 2 for v, u in zip(atoms[k], original[k], strict=True)) ** 0.5 for k in atoms
    ]
    provenance = request["provenance"] | {
        "preparation": "Meeko residue templates; default atom types/charges; no pKa optimisation",
        "meeko_version": version("meeko"),
        "max_heavy_atom_shift_A": max(shifts),
        "preparation_warnings": sorted({str(w.message) for w in caught}),
    }
    return {"protein_pdb": protein, "receptor_pdbqt": receptor, "provenance": provenance}


def parse_fpocket_output(directory: Path) -> list[dict[str, Any]]:
    info = (directory / "protein_info.txt").read_text()
    output = []
    for section in re.split(r"Pocket\s+(\d+)\s*:", info)[1:][::2]:
        index = int(section)
        pattern = rf"Pocket\s+{index}\s*:(.*?)(?=Pocket\s+\d+\s*:|\Z)"
        details = re.search(pattern, info, re.S)
        assert details is not None
        values = {}
        for line in details[1].splitlines():
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            try:
                number = float(value.strip())
                values[label.strip()] = number if math.isfinite(number) else None
            except ValueError:
                continue
        vertices = directory / "pockets" / f"pocket{index}_vert.pqr"
        if not vertices.exists():
            continue
        xyz = [
            [float(line[start : start + 8]) for start in (30, 38, 46)]
            for line in vertices.read_text().splitlines()
            if line.startswith(("ATOM  ", "HETATM"))
        ]
        if not xyz:
            continue
        box = box_from_coordinates(xyz)
        output.append(
            {
                "id": f"fpocket_{index}",
                "kind": "predicted_pocket",
                "rank": index,
                "score": values.get("Score"),
                "druggability_score": values.get("Druggability Score"),
                "volume_A3": values.get("Volume"),
                "alpha_spheres": len(xyz),
                "coordinates": xyz,
                **box,
            }
        )
    return output[:20]


def _pockets(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    _atoms(request["protein_pdb"], pdbqt=False)
    _phase(directory, "Checking/downloading the pinned fpocket engine (first use only)…")
    binary, identity = _install_fpocket()
    protein = directory / "protein.pdb"
    protein.write_text(request["protein_pdb"])
    _phase(directory, "Detecting candidate pockets with fpocket on one CPU…")
    result = subprocess.run(
        [binary, "-f", str(protein)], cwd=directory, capture_output=True, text=True, check=False
    )
    output = directory / "protein_out"
    if not result.returncode and "no pockets found" in result.stdout.lower():
        return {
            "pockets": [],
            "engine": identity,
            "protein_sha256": hashlib.sha256(request["protein_pdb"].encode()).hexdigest(),
            "method": "fpocket defaults; alpha-sphere bounds + 8 Å total padding; no box clipping",
            "warnings": [result.stderr.strip()] if result.stderr.strip() else [],
        }
    if result.returncode or not (output / "protein_info.txt").exists():
        raise DockingError(
            "fpocket returned no usable output. Inspect this structure or define a manual box."
        )
    return {
        "pockets": parse_fpocket_output(output),
        "engine": identity,
        "protein_sha256": hashlib.sha256(request["protein_pdb"].encode()).hexdigest(),
        "method": "fpocket defaults; alpha-sphere bounds + 8 Å total padding; no box clipping",
    }


def main() -> None:
    directory = Path(sys.argv[1])
    try:
        request = json.loads((directory / "request.json").read_text())
        if request["action"] == "prepare":
            result = _prepare(request, directory)
        elif request["action"] == "pockets":
            result = _pockets(request, directory)
        else:
            raise DockingError("Unknown receptor action.")
        (directory / "result.json").write_text(json.dumps(result, allow_nan=False))
    except Exception as exc:
        (directory / "result.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        )
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

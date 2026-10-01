"""Bounded, opt-in docking of exactly two molecules. No engine imports at startup.

The worker and Vina subprocess share a process group so timeouts/interrupted
Streamlit reruns terminate the entire job. A host-wide file lock admits one job.
Uploaded structures stay on the host; this module calls no external services.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory, gettempdir
from typing import Any

MAX_INPUT_BYTES = 3 * 1024 * 1024
MAX_RECEPTOR_ATOMS = 15_000
PROTEIN_RESIDUES = frozenset(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL "
    "HID HIE HIP HSD HSE HSP ASH GLH CYM CYX LYN ARN ACE NME".split()
)


class DockingError(ValueError):
    """A preparation, execution or chemistry validation failure."""


@dataclass(frozen=True)
class DockingSettings:
    center: tuple[float, float, float]
    size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    exhaustiveness: int = 8
    seed: int = 42
    n_poses: int = 3
    timeout_seconds: int = 180

    def __post_init__(self) -> None:
        if any(
            type(v) is not int
            for v in (self.exhaustiveness, self.seed, self.n_poses, self.timeout_seconds)
        ):
            raise DockingError("Exhaustiveness, seed, pose count and timeout must be integers.")
        if len(self.center) != 3 or not all(
            math.isfinite(v) and abs(v) <= 10_000 for v in self.center
        ):
            raise DockingError("Box center must contain three finite coordinates in Å.")
        if len(self.size) != 3 or not all(math.isfinite(v) and 6 <= v <= 25 for v in self.size):
            raise DockingError("Each box dimension must be between 6 and 25 Å.")
        if not 1 <= self.exhaustiveness <= 16 or not 1 <= self.n_poses <= 3:
            raise DockingError("Use exhaustiveness 1–16 and 1–3 poses for this bounded workflow.")
        if not 1 <= self.seed <= 2_147_483_647 or not 1 <= self.timeout_seconds <= 300:
            raise DockingError(
                "Use a positive seed and a total-job timeout of at most 300 seconds."
            )


def docking_unavailable_reason() -> str | None:
    """Check availability cheaply, without importing optional scientific packages."""
    if os.name != "posix":
        return "The bounded docking worker currently requires Linux or macOS."
    if not vina_binary():
        return (
            "AutoDock Vina executable is missing. "
            "Install autodock-vina or set SARSCOPE_VINA_BINARY."
        )
    missing = [
        name
        for name in ("meeko", "prolif", "gemmi", "filelock")
        if importlib.util.find_spec(name) is None
    ]
    if missing:
        return (
            f"Optional docking dependencies missing: {', '.join(missing)}. "
            "Install sarscope[docking]."
        )
    return None


def vina_binary() -> str | None:
    configured = os.environ.get("SARSCOPE_VINA_BINARY")
    return shutil.which(configured or "vina")


def _atoms(text: str, *, pdbqt: bool) -> dict[tuple[str, ...], tuple[float, ...]]:
    if not text or len(text.encode("utf-8")) > MAX_INPUT_BYTES:
        raise DockingError("Each receptor file must be nonempty and no larger than 3 MiB.")
    atoms: dict[tuple[str, ...], tuple[float, ...]] = {}
    serials: set[int] = set()
    for line in text.splitlines():
        record = line[:6].strip()
        if record in {"MODEL", "ROOT", "BRANCH", "TORSDOF"}:
            raise DockingError(
                "Supply a single rigid receptor, not an ensemble or flexible ligand."
            )
        if record not in {"ATOM", "HETATM"}:
            continue
        try:
            serial = int(line[6:11])
            coords = tuple(float(line[start : start + 8]) for start in (30, 38, 46))
            atom_name, residue, chain = line[12:16].strip(), line[17:20].strip(), line[21:22]
            number = str(int(line[22:26]))
            insertion = line[26:27].strip()
        except (ValueError, IndexError) as exc:
            raise DockingError(
                "Cannot parse receptor atom records; use standard PDB/PDBQT files."
            ) from exc
        if not all(math.isfinite(v) and abs(v) <= 10_000 for v in coords):
            raise DockingError("Receptor coordinates must be finite and expressed in Å.")
        if line[16:17].strip():
            raise DockingError("Resolve alternate atom locations before uploading the receptor.")
        if residue not in PROTEIN_RESIDUES:
            raise DockingError(
                f"Unsupported receptor residue {residue!r}. This version is protein-only; "
                "prepare an appropriate receptor without ligand, waters or cofactors."
            )
        if (
            not re.fullmatch(r"[A-Za-z0-9'*]{1,4}", atom_name)
            or not re.fullmatch(r"[A-Za-z0-9 ]", chain)
            or (insertion and not re.fullmatch(r"[A-Za-z0-9]", insertion))
        ):
            raise DockingError("Receptor atom/residue identifiers are malformed.")
        if serial in serials:
            raise DockingError("Receptor atom serial numbers must be unique.")
        serials.add(serial)
        if len(serials) > MAX_RECEPTOR_ATOMS:
            raise DockingError("Receptor exceeds the 15,000-atom browser limit.")
        if pdbqt:
            cells = line[54:].split()
            if len(line) < 78 or len(cells) < 2:
                raise DockingError("PDBQT receptor is missing atom types or partial charges.")
            try:
                charge = float(line[70:76])
            except ValueError as exc:
                raise DockingError("PDBQT charges are invalid.") from exc
            if not math.isfinite(charge):
                raise DockingError("PDBQT charges must be finite.")
            hydrogen = cells[-1] in {"H", "HD", "HS"}
        else:
            element = line[76:78].strip().upper()
            hydrogen = element in {"H", "D"} or (
                not element and atom_name.lstrip("0123456789").startswith("H")
            )
        if hydrogen:
            continue
        key = (chain.strip(), number, insertion, residue, atom_name)
        if key in atoms:
            raise DockingError("Duplicate heavy-atom identifiers in the receptor.")
        atoms[key] = coords
    if not atoms:
        raise DockingError("No protein heavy atoms found in the receptor.")
    return atoms


def validate_receptor_pair(pdb: str, pdbqt: str, settings: DockingSettings) -> str:
    """Reject mismatched coordinates/atom sets and return an atom-only safe PDB."""
    protein = _atoms(pdb, pdbqt=False)
    receptor = _atoms(pdbqt, pdbqt=True)
    if protein.keys() != receptor.keys():
        raise DockingError(
            "Protein PDB and receptor PDBQT must have identical heavy-atom identifiers."
        )
    if any(math.dist(protein[key], receptor[key]) > 0.05 for key in protein):
        raise DockingError("Protein PDB and receptor PDBQT coordinates disagree (>0.05 Å).")
    if not any(
        all(
            abs(v - c) <= s / 2 for v, c, s in zip(xyz, settings.center, settings.size, strict=True)
        )
        for xyz in receptor.values()
    ):
        raise DockingError("The docking box contains no receptor heavy atoms. Check its center.")
    # Strip user-supplied headers/annotations; only validated atom records reach HTML viewers.
    return (
        "\n".join(line for line in pdb.splitlines() if line.startswith(("ATOM  ", "HETATM")))
        + "\nEND\n"
    )


def interaction_changes(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare residue/type contacts, not atom indices of two different molecules."""

    def contacts(rows: list[dict[str, Any]]) -> dict[tuple[str, str], float | None]:
        result: dict[tuple[str, str], float | None] = {}
        for row in rows:
            key = (row["protein_residue"], row["interaction"])
            distance = row.get("distance_angstrom")
            previous = result.get(key)
            if key not in result or (
                distance is not None and (previous is None or distance < previous)
            ):
                result[key] = distance
        return result

    left, right = contacts(a), contacts(b)
    return [
        {
            "protein_residue": key[0],
            "interaction": key[1],
            "change_A_to_B": "retained"
            if key in left and key in right
            else "lost"
            if key in left
            else "gained",
            "distance_A_Angstrom": left.get(key),
            "distance_B_Angstrom": right.get(key),
        }
        for key in sorted(left.keys() | right.keys())
    ]


def docking_request_key(request: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(request, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _stop_worker(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_pair_docking(
    ligands: list[dict[str, Any]],
    protein_pdb: str,
    receptor_pdbqt: str,
    settings: DockingSettings,
    *,
    receptor_label: str,
    on_progress: Callable[[str], None] | None = None,
    receptor_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One isolated job. Uploaded structures are never submitted to a remote service."""
    if len(ligands) != 2:
        raise DockingError("Select exactly two different molecules from the cliff pair.")
    from sarscope.ligand_states import validate_ligand_state

    clean_ligands = []
    for ligand in ligands:
        if not all(
            isinstance(ligand.get(k), str) and ligand[k].strip() for k in ("molecule_id", "smiles")
        ):
            raise DockingError("Each ligand needs a molecule ID and curated SMILES.")
        activity = ligand.get("pactivity")
        if not isinstance(activity, (int, float)) or not math.isfinite(activity):
            raise DockingError("Each ligand needs a finite experimental pActivity.")
        state = validate_ligand_state(
            ligand.get("source_smiles", ligand["smiles"]), ligand["smiles"]
        )
        clean_ligands.append(
            {k: ligand[k] for k in ("molecule_id", "smiles", "pactivity")}
            | {"source_smiles": state["source_smiles"], "stereochemistry": state}
        )
    if clean_ligands[0]["molecule_id"] == clean_ligands[1]["molecule_id"]:
        raise DockingError("Select exactly two different molecules from the cliff pair.")
    if not isinstance(receptor_label, str) or not receptor_label.strip():
        raise DockingError("Record a receptor label or PDB ID for provenance.")
    safe_pdb = validate_receptor_pair(protein_pdb, receptor_pdbqt, settings)
    if receptor_provenance:
        from sarscope.receptor import check_site_components

        check_site_components(receptor_provenance, settings)
    reason = docking_unavailable_reason()
    if reason:
        raise DockingError(reason)

    request = {
        "ligands": clean_ligands,
        "protein_pdb": safe_pdb,
        "receptor_pdbqt": receptor_pdbqt,
        "settings": asdict(settings),
        "receptor_label": receptor_label,
        "vina_binary": vina_binary(),
        "receptor_provenance": receptor_provenance,
    }
    result = run_structure_job(
        request,
        module="sarscope.docking_worker",
        timeout=settings.timeout_seconds,
        label="Pair docking",
        on_progress=on_progress,
    )
    result["protein_pdb"] = safe_pdb
    result["receptor_pdbqt"] = receptor_pdbqt
    result["interaction_changes"] = interaction_changes(
        result["ligands"][0]["interactions"], result["ligands"][1]["interactions"]
    )
    return result


def run_structure_job(
    request: dict[str, Any],
    *,
    module: str,
    timeout: int = 120,
    label: str = "Structure job",
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Shared host lock and process-group deadline for preparation, pockets and docking."""
    if module not in {"sarscope.docking_worker", "sarscope.receptor_worker"}:
        raise DockingError("Unknown structure worker.")
    if os.name != "posix" or type(timeout) is not int or not 1 <= timeout <= 300:
        raise DockingError("Structure jobs require POSIX and an integer timeout of 1–300 seconds.")
    from filelock import FileLock, Timeout

    lock = FileLock(str(Path(gettempdir()) / "sarscope-pair-docking.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout as exc:
        raise DockingError(
            "Another pair or structure job is running on this host. "
            "Please try again when it finishes."
        ) from exc
    try:
        with TemporaryDirectory(prefix="sarscope-pair-") as tmp:
            directory = Path(tmp)
            (directory / "request.json").write_text(json.dumps(request, allow_nan=False))
            env = dict(os.environ)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            ):
                env[name] = "1"
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
            with (directory / "worker.log").open("w") as log:
                process = subprocess.Popen(
                    [sys.executable, "-m", module, str(directory)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
                start, previous = time.monotonic(), ""
                try:
                    while True:
                        try:
                            process.wait(timeout=0.5)
                            break
                        except subprocess.TimeoutExpired:
                            if time.monotonic() - start >= timeout:
                                raise DockingError(
                                    f"{label} exceeded its {timeout}-second total timeout."
                                ) from None
                        status = directory / "progress.json"
                        if on_progress and status.exists():
                            try:
                                phase = json.loads(status.read_text())["phase"]
                            except (OSError, ValueError, KeyError):
                                continue
                            if phase != previous:
                                on_progress(phase)
                                previous = phase
                finally:
                    # Also reap descendants if the worker itself exited unexpectedly.
                    _stop_worker(process)
            output = directory / "result.json"
            if not output.exists():
                raise DockingError(
                    f"{label} worker stopped without a result; "
                    "the host may have exhausted resources."
                )
            result = json.loads(output.read_text())
            if process.returncode or "error" in result:
                raise DockingError(result.get("error", f"{label} worker failed."))
            return result
    finally:
        lock.release()


def docking_result_zip(result: dict[str, Any]) -> bytes:
    """Downloadable standalone artifact; not silently added to the main SAR report."""
    import pandas as pd

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(result["manifest"], indent=2))
        archive.writestr("protein.pdb", result["protein_pdb"])
        archive.writestr("receptor.pdbqt", result["receptor_pdbqt"])
        provenance = result["manifest"].get("receptor_preparation") or {}
        if provenance.get("reference_pdb"):
            archive.writestr("reference_ligand.pdb", provenance["reference_pdb"])
        archive.writestr(
            "interaction_changes.csv",
            pd.DataFrame(result["interaction_changes"]).to_csv(index=False),
        )
        for slot, ligand in zip(("A", "B"), result["ligands"], strict=True):
            archive.writestr(f"ligand_{slot}_prepared.pdbqt", ligand["prepared_pdbqt"])
            for extension, field in (("sdf", "sdf"), ("pdbqt", "pdbqt"), ("html", "diagram_html")):
                archive.writestr(f"ligand_{slot}.{extension}", ligand[field])
            archive.writestr(
                f"ligand_{slot}_contacts.csv",
                pd.DataFrame(ligand["interactions"]).to_csv(index=False),
            )
        archive.writestr(
            "result.json",
            json.dumps(
                {k: v for k, v in result.items() if k not in {"protein_pdb", "receptor_pdbqt"}},
                indent=2,
            ),
        )
    return buffer.getvalue()

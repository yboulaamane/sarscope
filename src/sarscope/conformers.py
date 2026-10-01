"""Bounded ETKDG retries with diagnostics and coordinate-derived stereo validation."""

from __future__ import annotations

from typing import Any

from sarscope.docking import DockingError
from sarscope.ligand_states import validate_ligand_state


def validate_conformer_stereo(molecule: Any, smiles: str) -> None:
    """Check the actual 3D geometry, not merely the chiral tags copied from SMILES."""
    import numpy as np
    from rdkit import Chem

    if not molecule.GetNumConformers() or not molecule.GetConformer().Is3D():
        raise DockingError("Ligand preparation did not produce a 3D conformer.")
    if not np.isfinite(molecule.GetConformer().GetPositions()).all():
        raise DockingError("Ligand conformer contains invalid coordinates.")
    geometry = Chem.Mol(molecule)
    Chem.AssignStereochemistryFrom3D(geometry, confId=0, replaceExistingTags=True)
    try:
        validate_ligand_state(
            smiles, Chem.MolToSmiles(Chem.RemoveHs(geometry), isomericSmiles=True)
        )
    except DockingError as exc:
        raise DockingError(
            "Generated 3D geometry does not preserve the selected stereochemistry."
        ) from exc


def embed_ligand_conformer(smiles: str, seed: int) -> tuple[Any, dict[str, Any]]:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDistGeom

    validate_ligand_state(smiles, smiles)
    if type(seed) is not int or not 1 <= seed <= 2_147_483_647:
        raise DockingError("Use a positive 32-bit integer embedding seed.")
    original = Chem.AddHs(Chem.MolFromSmiles(smiles))
    traces: list[dict[str, Any]] = []
    strategies = (
        ("ETKDGv3", False, False, seed, 200),
        ("ETKDGv3 random coordinates", True, False, seed, 100),
        (
            "ETKDGv3 random coordinates + small-ring torsions",
            True,
            True,
            (seed + 104729) % 2147483647 or 1,
            100,
        ),
    )
    names = {int(value): name for name, value in rdDistGeom.EmbedFailureCauses.names.items()}
    for method, random_coords, small_rings, trial_seed, iterations in strategies:
        molecule = Chem.Mol(original)
        params = AllChem.ETKDGv3()
        params.randomSeed, params.numThreads = trial_seed, 1
        params.maxIterations = iterations
        params.enforceChirality = True
        params.useRandomCoords, params.useSmallRingTorsions = random_coords, small_rings
        params.trackFailures = True
        if hasattr(params, "timeout"):
            params.timeout = 10  # The enclosing worker also has a hard process-group deadline.
        result = AllChem.EmbedMolecule(molecule, params)
        counts = {
            names.get(i, f"CAUSE_{i}"): int(count)
            for i, count in enumerate(params.GetFailureCounts())
            if count
        }
        trace: dict[str, Any] = {
            "method": method,
            "seed": trial_seed,
            "max_iterations": iterations,
            "timeout_seconds": getattr(params, "timeout", None),
            "return_code": int(result),
            "failure_counts": counts,
            "enforce_chirality": True,
        }
        traces.append(trace)
        if result == 0:
            try:
                validate_conformer_stereo(molecule, smiles)
            except DockingError as exc:
                trace["geometry_error"] = str(exc)
                continue
            return molecule, {"method": method, "attempts": traces, "stereo_verified_from_3d": True}
    stereo_failure = any(
        any("CHIRAL" in cause or "STEREO" in cause for cause in t["failure_counts"])
        or t.get("geometry_error")
        for t in traces
    )
    detail = "; ".join(f"{t['method']}: {t['failure_counts']}" for t in traces)
    advice = (
        "Stereo constraints failed. Bridged rings or conflicting R/S/E/Z assignments may be "
        "incompatible with this embedding protocol. If this is an exploratory state, review "
        "another generated alternative; otherwise verify the source structure. "
        "No assigned stereochemistry was changed or disabled. "
        if stereo_failure
        else "Review the selected structure or use an external, validated preparation protocol. "
    )
    raise DockingError(
        f"Could not generate a 3D conformer after 3 bounded strategies. {advice}{detail}"
    )

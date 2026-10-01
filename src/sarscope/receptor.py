"""Structure inspection, bounded Meeko preparation and explicit docking-site provenance."""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from importlib.metadata import version
from typing import Any

from sarscope.docking import (
    MAX_RECEPTOR_ATOMS,
    PROTEIN_RESIDUES,
    DockingError,
    DockingSettings,
    run_structure_job,
)
from sarscope.sources.pdb import MAX_STRUCTURE_BYTES

ESSENTIAL_COMPONENTS = frozenset(
    "ZN FE FE2 MG MN CA CU NI CO CD HG HEM HEC FAD FMN NAD NAP NDP PLP COA SAM SAH".split()
)


def box_from_coordinates(xyz: list[list[float]], *, padding: float = 8.0) -> dict[str, Any]:
    import numpy as np

    points = np.asarray(xyz, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points) or not np.isfinite(points).all():
        raise DockingError("Site coordinates must be finite 3D points.")
    if not math.isfinite(padding) or not 0 <= padding <= 12:
        raise DockingError("Use total box padding between 0 and 12 Å.")
    lower, upper = points.min(axis=0), points.max(axis=0)
    center = list(map(float, (lower + upper) / 2))
    size = list(map(float, np.maximum(upper - lower + padding, 6)))
    return {
        "center": center,
        "size": size,
        "padding_A": padding,
        "fits_browser_limit": max(size) <= 25,
    }


def _chosen_atoms(residue: Any) -> tuple[list[Any], str]:
    occupancy: dict[str, float] = defaultdict(float)
    for atom in residue:
        if atom.element.is_hydrogen:
            continue
        if atom.altloc not in {"\x00", " "}:
            occupancy[atom.altloc] += float(atom.occ)
    chosen = min(occupancy, key=lambda code: (-occupancy[code], code)) if occupancy else ""
    atoms = [a for a in residue if not a.element.is_hydrogen and a.altloc in {"\x00", " ", chosen}]
    if len({a.name for a in atoms}) != len(atoms):
        raise DockingError("Duplicate atoms remain after alternate-location selection.")
    return atoms, chosen


def inspect_structure(cif: str, chains: list[str]) -> dict[str, Any]:
    """Model 1, selected chains, highest summed-occupancy altloc; never delete bad residues."""
    import gemmi
    import numpy as np
    from scipy.spatial import cKDTree

    if not cif or len(cif.encode()) > MAX_STRUCTURE_BYTES or not chains:
        raise DockingError("Choose protein chains from a coordinate file of at most 20 MiB.")
    try:
        structure = gemmi.make_structure_from_block(gemmi.cif.read_string(cif).sole_block())
        structure.setup_entities()
    except (RuntimeError, ValueError) as exc:
        raise DockingError(f"Cannot read this PDB mmCIF structure: {exc}") from exc
    if not len(structure):
        raise DockingError("No coordinate model found.")
    model = structure[0]
    if any(chain not in {c.name for c in model} for chain in chains):
        raise DockingError("A selected chain is absent from coordinate model 1.")
    if len(chains) > 62 or len(set(chains)) != len(chains):
        raise DockingError("Select 1–62 distinct protein chains.")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    mapping = {chain: alphabet[i] for i, chain in enumerate(chains)}
    clean = gemmi.Structure()
    clean.add_model(gemmi.Model(1))
    protein_xyz, alternatives, components = [], [], []
    for chain in model:
        selected = chain.name in chains
        output = gemmi.Chain(mapping[chain.name]) if selected else None
        for residue in chain:
            atoms, altloc = _chosen_atoms(residue)
            if not atoms:
                continue
            coords = [[float(a.pos.x), float(a.pos.y), float(a.pos.z)] for a in atoms]
            if not np.isfinite(coords).all():
                raise DockingError("Structure contains nonfinite atom coordinates.")
            polymer = residue.entity_type == gemmi.EntityType.Polymer
            protein = polymer or (
                residue.entity_type == gemmi.EntityType.Unknown and residue.name in PROTEIN_RESIDUES
            )
            if protein and selected:
                if residue.name not in PROTEIN_RESIDUES:
                    raise DockingError(
                        f"Selected chain contains unsupported protein residue {residue.name}. "
                        "Choose another structure or prepare it explicitly outside this workflow."
                    )
                record = gemmi.Residue()
                record.name, record.seqid, record.het_flag = residue.name, residue.seqid, "A"
                for a in atoms:
                    copy = a.clone()
                    copy.altloc = "\x00"
                    record.add_atom(copy)
                assert output is not None
                output.add_residue(record)
                protein_xyz.extend(coords)
                if altloc:
                    alternatives.append(f"{chain.name}:{residue.seqid}:{residue.name}={altloc}")
            elif not protein and residue.name not in {"HOH", "DOD", "WAT"}:
                label = f"{chain.name}:{residue.seqid}:{residue.name}"
                reference = gemmi.Structure()
                reference.add_model(gemmi.Model(1))
                reference_chain = gemmi.Chain("L")
                record = gemmi.Residue()
                record.name, record.seqid, record.het_flag = residue.name, gemmi.SeqId(1, " "), "H"
                for a in atoms:
                    copy = a.clone()
                    copy.altloc = "\x00"
                    record.add_atom(copy)
                reference_chain.add_residue(record)
                reference[0].add_chain(reference_chain)
                components.append(
                    {
                        "id": label,
                        "name": residue.name,
                        "heavy_atoms": len(atoms),
                        "carbon_atoms": sum(a.element.name == "C" for a in atoms),
                        "coordinates": coords,
                        "pdb": reference.make_pdb_string(),
                    }
                )
        if output is not None and len(output):
            clean[0].add_chain(output)
    if not protein_xyz or len(protein_xyz) > MAX_RECEPTOR_ATOMS:
        raise DockingError("Selected protein must have 1–15,000 heavy atoms.")
    # Polymer–nonpolymer covalent links are not handled by ordinary rigid protein docking.
    for connection in structure.connections:
        if connection.type != gemmi.ConnectionType.Covale:
            continue
        p1, p2 = connection.partner1, connection.partner2
        if (
            p1.chain_name in chains
            and p1.res_id.name in PROTEIN_RESIDUES
            and p2.res_id.name not in PROTEIN_RESIDUES
        ) or (
            p2.chain_name in chains
            and p2.res_id.name in PROTEIN_RESIDUES
            and p1.res_id.name not in PROTEIN_RESIDUES
        ):
            raise DockingError(
                "Covalently attached nonprotein components require external preparation."
            )
    tree = cKDTree(protein_xyz)
    references, essential, removed = [], [], []
    for component in components:
        nearest = float(tree.query(component["coordinates"])[0].min())
        component["nearest_protein_distance_A"] = nearest
        removed.append(
            {k: component[k] for k in ("id", "name", "heavy_atoms", "nearest_protein_distance_A")}
        )
        if component["name"] in ESSENTIAL_COMPONENTS:
            essential.append({k: component[k] for k in ("id", "name", "coordinates")})
        if nearest <= 6 and component["heavy_atoms"] >= 6 and component["carbon_atoms"]:
            references.append(
                component
                | {"kind": "bound_ligand", **box_from_coordinates(component["coordinates"])}
            )
    return {
        "protein_pdb": clean.make_pdb_string(),
        "references": references,
        "provenance": {
            "gemmi_version": version("gemmi"),
            "source_cif_sha256": hashlib.sha256(cif.encode()).hexdigest(),
            "selected_protein_sha256": hashlib.sha256(clean.make_pdb_string().encode()).hexdigest(),
            "model": 1,
            "model_count": len(structure),
            "selected_auth_chains": chains,
            "chain_mapping": mapping,
            "alternate_locations": alternatives,
            "removed_nonwater_components": removed,
            "excluded_essential_components": essential,
            "water_policy": "waters removed",
            "hydrogen_policy": "source hydrogens replaced by Meeko templates",
            "scope": "Asymmetric-unit selected chains, not validated biological assembly.",
        },
    }


def check_site_components(provenance: dict[str, Any], settings: DockingSettings) -> None:
    for component in provenance.get("excluded_essential_components", []):
        if any(
            all(
                abs(v - c) <= s / 2 + 3
                for v, c, s in zip(point, settings.center, settings.size, strict=True)
            )
            for point in component["coordinates"]
        ):
            raise DockingError(
                f"Excluded metal/cofactor {component['id']} is near this box. "
                "Protein-only docking is unsuitable here; use a validated external protocol."
            )


def prepare_receptor(
    protein_pdb: str, provenance: dict[str, Any], *, on_progress: Any = None
) -> dict[str, Any]:
    return run_structure_job(
        {"action": "prepare", "protein_pdb": protein_pdb, "provenance": provenance},
        module="sarscope.receptor_worker",
        timeout=120,
        label="Receptor preparation",
        on_progress=on_progress,
    )


def predict_pockets(protein_pdb: str, *, on_progress: Any = None) -> dict[str, Any]:
    return run_structure_job(
        {"action": "pockets", "protein_pdb": protein_pdb},
        module="sarscope.receptor_worker",
        timeout=120,
        label="Pocket prediction",
        on_progress=on_progress,
    )

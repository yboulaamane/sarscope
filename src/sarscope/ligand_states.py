"""Bounded ligand preflight and explicit, docking-only stereochemistry choices."""

from __future__ import annotations

from typing import Any

from sarscope.docking import DockingError

MAX_STEREOISOMERS = 8


def _molecule(smiles: str) -> Any:
    from rdkit import Chem
    from rdkit.Chem import Lipinski

    if not isinstance(smiles, str) or not smiles.strip() or len(smiles) > 20_000:
        raise DockingError("Use a valid ligand SMILES of at most 20,000 characters.")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or len(Chem.GetMolFrags(molecule)) != 1:
        raise DockingError("Ligands must be valid, single-component curated structures.")
    if not 2 <= molecule.GetNumHeavyAtoms() <= 80 or Lipinski.NumRotatableBonds(molecule) > 15:
        raise DockingError(
            "Browser docking supports 2–80 heavy atoms and at most 15 rotatable bonds."
        )
    if any(a.GetAtomicNum() not in {6, 7, 8, 9, 15, 16, 17, 35, 53} for a in molecule.GetAtoms()):
        raise DockingError("Browser docking supports organic C/N/O/F/P/S/Cl/Br/I ligands only.")
    if any(len(ring) >= 9 for ring in molecule.GetRingInfo().AtomRings()):
        raise DockingError("Macrocycles need an external, validated preparation protocol.")
    if molecule.GetStereoGroups():
        raise DockingError(
            "Enhanced/relative stereochemistry needs external review; "
            "this selector supports individual absolute stereoisomers only."
        )
    return molecule


def inspect_ligand(smiles: str) -> dict[str, Any]:
    from rdkit import Chem

    molecule = _molecule(smiles)
    features = [
        {"type": str(feature.type), "atom_or_bond_index": int(feature.centeredOn)}
        for feature in Chem.FindPotentialStereo(molecule)
        if feature.specified != Chem.StereoSpecified.Specified
    ]
    return {
        "canonical_smiles": Chem.MolToSmiles(molecule, isomericSmiles=True),
        "unassigned_features": features,
    }


def enumerate_ligand_states(smiles: str) -> list[str]:
    """No random subset or conformer work; preserve specified centres and double bonds."""
    from rdkit import Chem
    from rdkit.Chem.EnumerateStereoisomers import (
        EnumerateStereoisomers,
        GetStereoisomerCount,
        StereoEnumerationOptions,
    )

    molecule = _molecule(smiles)
    options = StereoEnumerationOptions(
        onlyUnassigned=True, unique=True, maxIsomers=MAX_STEREOISOMERS + 1, tryEmbedding=False
    )
    count = GetStereoisomerCount(molecule, options=options)
    if count > MAX_STEREOISOMERS:
        raise DockingError(
            f"Up to {count} stereoisomers are possible, exceeding the {MAX_STEREOISOMERS}-state "
            "browser limit. Provide one fully specified, reviewed SMILES below."
        )
    states = sorted(
        {
            Chem.MolToSmiles(m, isomericSmiles=True)
            for m in EnumerateStereoisomers(molecule, options)
        }
    )
    if not states or len(states) > MAX_STEREOISOMERS:
        raise DockingError("No bounded set of stereoisomers available; supply a reviewed SMILES.")
    for state in states:
        validate_ligand_state(smiles, state)
    return states


def validate_ligand_state(source_smiles: str, selected_smiles: str) -> dict[str, Any]:
    """Require identical connectivity/charges and retain every supplied stereo assignment."""
    from rdkit import Chem

    source, selected = _molecule(source_smiles), _molecule(selected_smiles)
    if inspect_ligand(selected_smiles)["unassigned_features"]:
        raise DockingError(
            "Resolve unspecified ligand stereochemistry (including E/Z) before docking. "
            "Use the ligand-state review controls to select an explicit stereoisomer."
        )
    source_graph, selected_graph = Chem.Mol(source), Chem.Mol(selected)
    Chem.RemoveStereochemistry(source_graph)
    Chem.RemoveStereochemistry(selected_graph)
    if Chem.MolToSmiles(source_graph, isomericSmiles=True) != Chem.MolToSmiles(
        selected_graph, isomericSmiles=True
    ) or not selected.HasSubstructMatch(source, useChirality=True):
        raise DockingError(
            "The docking state must retain the original connectivity, charges, isotopes "
            "and every already-specified stereocentre/E/Z bond."
        )
    features = inspect_ligand(source_smiles)["unassigned_features"]
    return {
        "source_smiles": source_smiles,
        "selected_smiles": Chem.MolToSmiles(selected, isomericSmiles=True),
        "status": "exploratory_selection" if features else "as_supplied",
        "unassigned_source_features": features,
        "experimental_configuration_verified": False,
        "note": (
            "Selected stereoisomer is a docking-only hypothesis; source experimental activity "
            "is not established for this resolved stereoisomer."
            if features
            else "Input stereochemistry retained; not independently verified against the assay."
        ),
    }

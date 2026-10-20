"""Independent MMFF94 molecular-interaction fields from published equations.

Halgren (1996), J Comput Chem 17, 520-552,
doi:10.1002/(SICI)1096-987X(199604)17:5/6<520::AID-JCC2>3.0.CO;2-W.
Buffered 14-7 VDW uses RDKit's public MMFF mixed-pair parameter API and a
type-1 alkyl-carbon probe. MMFF point-probe electrostatics use its published
0.05 A distance buffer, not a fitted soft-core or an intermolecular 1-4 scale.
This is not a full MMFF force-field engine.
No Open3DQSAR/OpenBabel algorithm source was consulted or ported.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem

MMFF_COULOMB = 332.0716
MMFF_ELECTROSTATIC_BUFFER = 0.05
PARAMETER_SOURCE = "RDKit public MMFF94 mixed-pair API; Halgren 1996 buffered 14-7"


def carbon_probe_parameters(mol: Any) -> tuple[NDArray[np.float64], NDArray[np.float64], list[int]]:
    """Get mixed atom/CR parameters without maintaining copied force-field tables.

    The disconnected methane fragment is used for *parameter lookup only*;
    its carbon must be MMFF type 1. None of its atoms enter the field sum.
    Ring perception must be reinitialised after CombineMols.
    """
    probe = Chem.AddHs(Chem.MolFromSmiles("C"))
    combined = Chem.CombineMols(Chem.Mol(mol), probe)
    Chem.SanitizeMol(combined)
    props = AllChem.MMFFGetMoleculeProperties(combined, mmffVariant="MMFF94")
    index = mol.GetNumAtoms()
    if props is None or props.GetMMFFAtomType(index) != 1:
        raise ValueError("MMFF94 atom typing or alkyl-carbon probe parameters unavailable.")
    radii, depths, atom_types = [], [], []
    for i in range(index):
        values = props.GetMMFFVdWParams(i, index)
        if values is None or props.GetMMFFAtomType(i) == 0:
            raise ValueError(f"MMFF94 mixed-pair parameters unavailable for atom {i}.")
        # Returned tuple: unscaled R*, epsilon, H-bond-scaled R*, epsilon.
        # The CR probe is neither a donor nor acceptor; use the public scaled pair.
        radii.append(values[2])
        depths.append(values[3])
        atom_types.append(int(props.GetMMFFAtomType(i)))
    radius = np.asarray(radii, dtype=float)
    epsilon = np.asarray(depths, dtype=float)
    if (
        not np.isfinite(radius).all()
        or not np.isfinite(epsilon).all()
        or ((radius <= 0).any() or (epsilon <= 0).any())
    ):
        raise ValueError("MMFF94 pair parameters must be finite and positive.")
    return radius, epsilon, atom_types


def buffered_14_7(
    distance: NDArray[np.float64], radius: NDArray[np.float64], epsilon: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Per-atom probe energy in kcal/mol; finite even at zero separation."""
    ratio = distance / radius
    return epsilon * (1.07 / (ratio + 0.07)) ** 7 * (1.12 / (ratio**7 + 0.12) - 2)


def energy_chunk(
    distance2: NDArray[np.float64],
    radius: NDArray[np.float64],
    epsilon: NDArray[np.float64],
    charges: NDArray[np.float64],
    *,
    dielectric: float,
    dielectric_model: str,
    steric_cutoff: float,
    electrostatic_cutoff: float,
) -> dict[str, NDArray[np.float64]]:
    """Sum all explicit atoms, then bound the two energies independently.

    VDW uses the actual distance: buffering already makes its origin finite.
    The published 0.05 A buffer regularises the Coulomb origin.
    There is no cross-field NaN mask or silent imputation in this profile.
    """
    distance = np.sqrt(distance2)
    steric = buffered_14_7(distance, radius, epsilon).sum(axis=1)
    if dielectric_model != "constant":
        raise ValueError("This MMFF94 MIF profile supports constant dielectric only.")
    denominator = distance + MMFF_ELECTROSTATIC_BUFFER
    electrostatic = MMFF_COULOMB * (charges / denominator).sum(axis=1) / dielectric
    return {
        "steric": np.clip(steric, -steric_cutoff, steric_cutoff),
        "electrostatic": np.clip(electrostatic, -electrostatic_cutoff, electrostatic_cutoff),
    }

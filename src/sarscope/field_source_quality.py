"""Source-graph checks, independent of model scores and alignment quality."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd
from rdkit import Chem


def audit_field_structures(molecules: Sequence[Any], identifiers: Sequence[str]) -> pd.DataFrame:
    """Account for encoded radicals without diagnosing or repairing their cause.

    Formal ions are allowed: a charged closed-shell molecule is not a radical.
    An encoded radical is not evidence that the measured compound really was
    a radical. Historical source conversion defects require primary-data review.
    """
    if len(molecules) != len(identifiers):
        raise ValueError("Source structures and compound identifiers must have matching lengths.")
    rows = []
    for identifier, molecule in zip(identifiers, molecules, strict=True):
        if molecule is None:
            raise ValueError(f"{identifier}: source structure could not be parsed.")
        atoms = [a for a in molecule.GetAtoms() if a.GetNumRadicalElectrons()]
        rows.append(
            {
                "molecule_id": str(identifier),
                "radical_electrons": sum(a.GetNumRadicalElectrons() for a in atoms),
                "radical_atom_indices_zero_based": ";".join(str(a.GetIdx()) for a in atoms),
                "formal_charge": Chem.GetFormalCharge(molecule),
                "source_graph_review_required": bool(atoms),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "molecule_id",
            "radical_electrons",
            "radical_atom_indices_zero_based",
            "formal_charge",
            "source_graph_review_required",
        ],
    )


def require_closed_shell(molecules: Sequence[Any], identifiers: Sequence[str]) -> pd.DataFrame:
    """Native 3D-QSAR supports closed-shell inputs, not unreviewed radical graphs."""
    audit = audit_field_structures(molecules, identifiers)
    flagged = audit[audit.source_graph_review_required]
    if not flagged.empty:
        examples = ", ".join(flagged.molecule_id.iloc[:3])
        raise ValueError(
            f"Source-structure review required: {len(flagged)} compound(s) encode radical "
            f"electrons ({examples}). Native 3D-QSAR requires closed-shell structures. "
            "Verify bond orders, valence and charges against the original source; "
            "no automatic repair or compound exclusion was performed."
        )
    return audit

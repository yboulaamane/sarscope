"""Single-cut matched molecular pairs (MMPs).

Unlike R-group analysis, MMPs do not require a preselected shared scaffold.
Every eligible single bond is cut; compounds sharing the larger (context)
fragment form pairs whose smaller fragments describe the transformation.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdMMPA

MMP_COLUMNS: tuple[str, ...] = (
    "id_a",
    "id_b",
    "context",
    "fragment_a",
    "fragment_b",
    "pactivity_a",
    "pactivity_b",
    "delta_pactivity_b_minus_a",
)


def _heavy_atoms(smiles: str) -> int:
    molecule = Chem.MolFromSmiles(smiles)
    return molecule.GetNumHeavyAtoms() if molecule is not None else 0


def _cuts(smiles: str, max_variable_heavy_atoms: int) -> list[tuple[str, str]]:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return []
    pairs: set[tuple[str, str]] = set()
    for _, fragments in rdMMPA.FragmentMol(molecule, maxCuts=1, resultsAsMols=False):
        pieces = fragments.split(".")
        if len(pieces) != 2:
            continue
        ranked = sorted(pieces, key=lambda part: (_heavy_atoms(part), part))
        variable, context = ranked[0], ranked[1]
        if _heavy_atoms(variable) <= max_variable_heavy_atoms:
            pairs.add((context, variable))
    return sorted(pairs)


def matched_molecular_pairs(
    table: pd.DataFrame,
    *,
    max_variable_heavy_atoms: int = 10,
    max_pairs: int = 100_000,
) -> pd.DataFrame:
    """Enumerate transformations between molecules sharing a single-cut context."""
    contexts: dict[str, list[tuple[str, str, float]]] = {}
    for row in table[["molecule_id", "smiles", "pactivity"]].itertuples(index=False):
        for context, variable in _cuts(str(row.smiles), max_variable_heavy_atoms):
            contexts.setdefault(context, []).append(
                (str(row.molecule_id), variable, float(row.pactivity))
            )

    rows: list[dict[str, Any]] = []
    for context in sorted(contexts):
        members = sorted(set(contexts[context]))
        for left, right in combinations(members, 2):
            if left[0] == right[0] or left[1] == right[1]:
                continue
            rows.append(
                {
                    "id_a": left[0],
                    "id_b": right[0],
                    "context": context,
                    "fragment_a": left[1],
                    "fragment_b": right[1],
                    "pactivity_a": left[2],
                    "pactivity_b": right[2],
                    "delta_pactivity_b_minus_a": right[2] - left[2],
                }
            )
            if len(rows) >= max_pairs:
                break
        if len(rows) >= max_pairs:
            break
    result = pd.DataFrame(rows, columns=list(MMP_COLUMNS))
    if result.empty:
        return result
    return result.sort_values(
        "delta_pactivity_b_minus_a", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)

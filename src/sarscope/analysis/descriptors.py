"""The six physicochemical properties of the reference paper's Table 2.

    MW              Descriptors.MolWt            (average, not exact, mass)
    logP            Crippen.MolLogP
    TPSA            Descriptors.TPSA             (default: N and O only, Ertl)
    RB              Descriptors.NumRotatableBonds
    NumHDonors      Descriptors.NumHDonors
    NumHAcceptors   Descriptors.NumHAcceptors

**This deliberately does not reuse Sorbent's descriptors.** Sorbent's ``hba`` is
``Lipinski.NOCount`` (count of N + O), chosen because it matched vendor
catalogue values on 97.5% of a 200k library. The paper names RDKit's
``NumHAcceptors`` explicitly, and on diverse PubChem compounds the two disagree
74% of the time. Reproducing the paper means using the paper's definition.

TPSA stays at RDKit's default, which reproduces Ertl's reference values on 100%
of RDKit's own NCI test set; ``includeSandP=True`` matched 0.4%. (Measured
while building Sorbent.)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors

if TYPE_CHECKING:
    from rdkit.Chem import Mol

#: Column names, in the paper's order. Every analysis downstream uses these.
PAPER_DESCRIPTORS: tuple[str, ...] = (
    "MW",
    "logP",
    "TPSA",
    "RB",
    "NumHDonors",
    "NumHAcceptors",
)


def compute_descriptors(mol: Mol) -> dict[str, float]:
    """All of PAPER_DESCRIPTORS for one molecule, as floats. Keys exactly those."""
    return {
        "MW": float(Descriptors.MolWt(mol)),
        "logP": float(Crippen.MolLogP(mol)),
        "TPSA": float(Descriptors.TPSA(mol)),
        "RB": float(Descriptors.NumRotatableBonds(mol)),
        "NumHDonors": float(Descriptors.NumHDonors(mol)),
        "NumHAcceptors": float(Descriptors.NumHAcceptors(mol)),
    }


def add_descriptors(table: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``table`` with one column per PAPER_DESCRIPTORS.

    Parses ``table["smiles"]`` (already standardised by curation, so a parse
    failure here is a bug, not bad input: raise ValueError naming the molecule).
    Does not mutate ``table``.
    """
    rows = []
    for molecule_id, smiles in zip(
        table.get("molecule_id", table.index), table["smiles"], strict=True
    ):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"cannot parse curated SMILES for {molecule_id}: {smiles!r}")
        rows.append(compute_descriptors(mol))
    values = pd.DataFrame(rows, index=table.index, columns=list(PAPER_DESCRIPTORS))
    return pd.concat([table, values], axis=1)

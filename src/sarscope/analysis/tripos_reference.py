"""Public Tripos-5.2 VDW parameter subset; never read proprietary/GPL source.

Clark, Cramer & Van Opdenbosch (1989), doi:10.1002/jcc.540100804.
Numeric R/K pairs: rDock manual, Appendix, Table 21 (public documentation):
https://rdock.github.io/documentation/html_docs/reference-guide/appendix.html

This is parameter selection for explicit-H organic molecules, NOT the SYBYL
atom typer. C/N/O rows in the published table share R/K within each element;
polar H and oxidised S need separate values. Unknown elements/environments
are refused. No default/undefined atom parameter or UFF fallback is allowed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

PARAMETER_SOURCE = (
    "Tripos 5.2 R/K subset, Clark et al. 1989; "
    "public rDock documentation Table 21; not vendor atom-typing parity"
)

# R in Å, K in kcal/mol. Facts from the cited table, not program source.
ELEMENT_RK = {
    "C": (1.70, 0.107),
    "N": (1.55, 0.095),
    "O": (1.52, 0.116),
    "P": (1.80, 0.314),
    "F": (1.47, 0.109),
    "Cl": (1.75, 0.314),
    "Br": (1.85, 0.434),
    "I": (1.98, 0.623),
}


def vdw_parameters(mol: Any) -> tuple[NDArray[np.float64], NDArray[np.float64], list[str]]:
    """Return explicit-H R/K values and the actual parameter class for every atom.

    Attached explicit hydrogen is required, so united-atom C.Hn rows cannot
    accidentally be combined with separate H contributions. Polar H on N/O/S
    uses H.P; C-bound H uses H. Other H environments require adjudication.
    Sulfoxide/sulfone S uses S.o/S.o2 when directly double-bonded to O.
    These public environment rules are recorded as SARscope rules, not as a
    reverse-engineered claim about SYBYL's complete atom-typing behaviour.
    """
    radii, depths, classes = [], [], []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if atom.GetNumImplicitHs() or atom.GetNumExplicitHs():
            raise ValueError("Tripos-5.2 reference fields require separate explicit H atoms.")
        if symbol == "H":
            neighbours = list(atom.GetNeighbors())
            if len(neighbours) != 1:
                raise ValueError("Unsupported hydrogen environment for Tripos-5.2 parameters.")
            attached = neighbours[0].GetSymbol()
            if attached == "C":
                label, pair = "H", (1.50, 0.042)
            elif attached in ("N", "O", "S"):
                label, pair = "H.P", (1.20, 0.042)
            else:
                raise ValueError(f"No documented H parameter rule for H attached to {attached}.")
        elif symbol == "S":
            oxygen_double_bonds = sum(
                bond.GetBondTypeAsDouble() == 2 and bond.GetOtherAtom(atom).GetAtomicNum() == 8
                for bond in atom.GetBonds()
            )
            if oxygen_double_bonds:
                label, pair = "S.o2" if oxygen_double_bonds >= 2 else "S.o", (1.70, 0.314)
            else:
                label, pair = "S.2/S.3 (same R/K)", (1.80, 0.314)
        elif symbol in ELEMENT_RK:
            label, pair = symbol + " (explicit-H R/K)", ELEMENT_RK[symbol]
        else:
            raise ValueError(f"No public Tripos-5.2 parameter class implemented for {symbol}.")
        radii.append(pair[0])
        depths.append(pair[1])
        classes.append(label)
    return np.asarray(radii), np.asarray(depths), classes

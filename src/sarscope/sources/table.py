"""Your own activity data from a CSV or TSV, for anything that is not in ChEMBL.

Output is the same measurement-level frame the ChEMBL filter produces, so the
rest of the pipeline cannot tell the sources apart. Repeated rows for one
molecule are allowed and are aggregated during curation, exactly like repeated
ChEMBL measurements.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from sarscope.curate import MEASUREMENT_COLUMNS
from sarscope.units import to_pactivity


def read_activity_table(
    path: Path,
    *,
    id_col: str = "molecule_id",
    smiles_col: str = "smiles",
    value_col: str = "pactivity",
    unit: str = "p",
) -> pd.DataFrame:
    """Read a table and return it in MEASUREMENT_COLUMNS form.

    ``unit="p"`` means ``value_col`` is already on the -log10(M) scale (pIC50,
    pKi). Any unit in ``units.MOLAR_OFFSETS`` ("nM", "uM", ...) is converted.
    Rows with a missing ID, SMILES or value are dropped; a non-positive
    concentration is an error, because it is always a data-entry mistake and
    silently dropping it would hide that.
    """
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    raw = pd.read_csv(path, sep=sep, dtype={id_col: str, smiles_col: str})

    missing = [c for c in (id_col, smiles_col, value_col) if c not in raw.columns]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; found {list(raw.columns)}")

    frame = raw[[id_col, smiles_col, value_col]].dropna()
    values = pd.to_numeric(frame[value_col], errors="raise").astype(float)
    if unit == "p":
        pactivity = values
    else:
        pactivity = values.map(lambda v: to_pactivity(v, unit))

    return pd.DataFrame(
        {
            "molecule_id": frame[id_col].astype(str).str.strip(),
            "smiles": frame[smiles_col].astype(str).str.strip(),
            "pactivity": pactivity,
            "record_id": [f"{path.name}:{i + 2}" for i in frame.index],  # 1-based + header
        }
    ).reset_index(drop=True)[list(MEASUREMENT_COLUMNS)]

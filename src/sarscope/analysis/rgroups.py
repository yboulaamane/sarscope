"""R-group decomposition with per-position activity statistics.

This is the input to the one step of the reference workflow a tool cannot
finish. The paper's Table 5 reads "tert-butyl on R3 is highly beneficial;
substitution on R5 gives inactive compounds" - a chemist's reading of a
substituent table. SARscope builds that table and its statistics; the
interpretation stays with the chemist.

For each scaffold with enough members, RDKit's ``RGroupDecomposition`` splits
every molecule into a shared core plus substituents at numbered positions
(R1, R2, ...). Grouping potency by (position, substituent) gives, for each
substituent, how many molecules carry it and how their median potency compares
with the rest of the series at that position.

**The delta is within-position, not against the whole series.** A substituent
at R1 is compared with the molecules that differ at R1, so the comparison is
between alternatives at one point of substitution.

**Hydrogen is a substituent.** RDKit returns ``[H][*:1]`` for an unsubstituted
position, and that is exactly the reference a chemist wants ("adding fluorine
here gains a log unit" is a comparison against H). It is kept and labelled "H".

**A delta is a description of this dataset, not a prediction.** Two molecules
sharing a substituent may differ everywhere else, so a large delta on few
molecules means little. Every row therefore carries ``n`` and the
Mann-Whitney p-value for that substituent against the rest of its position,
and rows are ordered by effect size only within a position.

RDKit's decomposition is symmetry-aware: on a para-substituted ring the two
equivalent positions may be numbered either way round. Positions are therefore
reported as RDKit labels them for the given core, not as a chemist would number
the parent ring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdRGroupDecomposition
from scipy.stats import mannwhitneyu

#: Columns of ScaffoldSar.substituents, in order.
SUBSTITUENT_COLUMNS: tuple[str, ...] = (
    "position",  # "R1", "R2", ...
    "substituent",  # SMILES with the attachment point, or "H"
    "n",  # molecules carrying it at this position
    "median_pactivity",
    "median_other",  # median of the molecules differing at this position
    "delta",  # median_pactivity - median_other
    "p_value",  # Mann-Whitney U vs the rest of the position; NaN if undefined
    "min_pactivity",
    "max_pactivity",
)


@dataclass
class ScaffoldSar:
    """One scaffold's decomposition."""

    scaffold: str  # the Murcko scaffold this came from
    core: str  # the labelled core RDKit produced, with attachment points
    n_molecules: int  # molecules decomposed
    n_unmatched: int  # members the core could not be matched to
    #: One row per molecule: molecule_id, pactivity, activity_class, then one
    #: column per position holding that molecule's substituent.
    members: pd.DataFrame
    #: SUBSTITUENT_COLUMNS, sorted by position then delta descending.
    substituents: pd.DataFrame

    @property
    def positions(self) -> list[str]:
        return [c for c in self.members.columns if c.startswith("R")]


def decompose_scaffold(
    table: pd.DataFrame,
    scaffold: str,
    *,
    min_substituent_count: int = 2,
) -> ScaffoldSar | None:
    """Decompose the molecules of ``table`` sharing ``scaffold``.

    ``table`` needs molecule_id, smiles, pactivity, activity_class and murcko
    columns (the curated table after ``add_scaffolds``). Returns None when the
    scaffold cannot be parsed or no molecule decomposes against it.

    Substituents carried by fewer than ``min_substituent_count`` molecules at a
    position are dropped from ``substituents`` - a single-molecule substituent
    has a delta but no evidence - while staying in ``members``. Positions where
    every molecule carries the same substituent are dropped entirely: they are
    part of the core in practice and carry no SAR.
    """
    core_mol = Chem.MolFromSmiles(scaffold)
    if core_mol is None:
        return None
    members = table[table["murcko"] == scaffold]
    if members.empty:
        return None

    mols = []
    kept = []
    for row in members.itertuples():
        mol = Chem.MolFromSmiles(row.smiles)
        if mol is not None:
            mols.append(mol)
            kept.append(row)
    if not mols:
        return None

    # RDKit logs a warning per molecule it cannot place on the core; the
    # unmatched count below reports the same thing once.
    RDLogger.DisableLog("rdApp.*")
    rows, unmatched = rdRGroupDecomposition.RGroupDecompose(
        [core_mol], mols, asSmiles=True, asRows=True
    )
    if not rows:
        return None

    unmatched_set = set(unmatched)
    matched = [r for i, r in enumerate(kept) if i not in unmatched_set]
    if len(matched) != len(rows):  # pragma: no cover - RDKit contract guard
        raise RuntimeError(f"decomposition returned {len(rows)} rows for {len(matched)} molecules")

    positions = sorted(
        {k for row in rows for k in row if k.startswith("R")},
        key=lambda name: int(name[1:]),
    )
    frame = pd.DataFrame(
        {
            "molecule_id": [r.molecule_id for r in matched],
            "pactivity": [float(r.pactivity) for r in matched],
            "activity_class": [r.activity_class for r in matched],
            **{position: [_label(row.get(position, "")) for row in rows] for position in positions},
        }
    )
    varying = [p for p in positions if frame[p].nunique() > 1]
    frame = frame[["molecule_id", "pactivity", "activity_class", *varying]]

    stats = _substituent_stats(frame, varying, min_substituent_count)
    return ScaffoldSar(
        scaffold=scaffold,
        core=rows[0].get("Core", ""),
        n_molecules=len(frame),
        n_unmatched=len(unmatched_set),
        members=frame.reset_index(drop=True),
        substituents=stats,
    )


def _label(smiles: str) -> str:
    """RDKit's substituent SMILES, with the hydrogen placeholder named "H"."""
    if not smiles:
        return "H"
    # An unsubstituted position comes back as "[H][*:1]" (any attachment index).
    stripped = smiles.replace("[H]", "")
    if stripped.startswith("[*:") and stripped.endswith("]"):
        return "H"
    return smiles


def _substituent_stats(frame: pd.DataFrame, positions: list[str], min_count: int) -> pd.DataFrame:
    rows = []
    for position in positions:
        for substituent, block in frame.groupby(position):
            if len(block) < min_count:
                continue
            others = frame.loc[frame[position] != substituent, "pactivity"]
            mine = block["pactivity"]
            if len(others) and mine.nunique() + others.nunique() > 1:
                p_value = float(mannwhitneyu(mine, others, alternative="two-sided").pvalue)
            else:
                p_value = float("nan")
            median_other = float(others.median()) if len(others) else float("nan")
            rows.append(
                {
                    "position": position,
                    "substituent": str(substituent),
                    "n": len(block),
                    "median_pactivity": float(mine.median()),
                    "median_other": median_other,
                    "delta": float(mine.median() - median_other) if len(others) else np.nan,
                    "p_value": p_value,
                    "min_pactivity": float(mine.min()),
                    "max_pactivity": float(mine.max()),
                }
            )
    stats = pd.DataFrame(rows, columns=list(SUBSTITUENT_COLUMNS))
    stats = stats.sort_values(["position", "delta", "substituent"], ascending=[True, False, True])
    return stats.reset_index(drop=True)


def decompose_top_scaffolds(
    table: pd.DataFrame,
    enrichment: pd.DataFrame,
    *,
    max_scaffolds: int = 10,
    min_members: int = 8,
    min_substituent_count: int = 2,
) -> list[ScaffoldSar]:
    """Decompose the most populated scaffolds, largest series first.

    Populated, not most enriched: a series needs members before its
    substituents can be compared, and an all-active scaffold of three
    molecules has the highest possible enrichment and nothing to say about
    SAR. Scaffolds with fewer than ``min_members`` molecules are skipped,
    as are any that fail to decompose.
    """
    counts = enrichment[enrichment["n"] >= min_members]
    ordered = counts.sort_values(["n", "ef_lower", "scaffold"], ascending=[False, False, True])
    out: list[ScaffoldSar] = []
    for scaffold in ordered["scaffold"]:
        if len(out) >= max_scaffolds:
            break
        sar = decompose_scaffold(table, str(scaffold), min_substituent_count=min_substituent_count)
        if sar is not None:
            out.append(sar)
    return out

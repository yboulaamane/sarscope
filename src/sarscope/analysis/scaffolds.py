"""Murcko scaffold diversity (Table 4) and scaffold enrichment factors.

Scaffolds come from Sorbent: ``murcko_scaffold`` (ring systems + linkers, atom
types kept) and ``generic_scaffold`` (all atoms carbon, all bonds single - the
"cyclic skeleton"). Both return None for an acyclic molecule.

**Acyclic molecules** count towards N but have no scaffold, so they contribute
to no scaffold or skeleton count. The table reports how many there were.

**Skeleton counts will not match the paper.** Table 4 reports 47 cyclic
skeletons for 1,953 Murcko scaffolds; RDKit's generic scaffold collapses far
less than that. The paper used DataWarrior, whose skeleton definition is not
RDKit's. The Murcko columns are comparable, the skeleton columns are not.

**Enrichment factor.** EF = (a / n) / (A / N): the Group 1 fraction within a
scaffold over the Group 1 fraction of the whole dataset, where the whole
dataset includes acyclic molecules. This definition is confirmed by the
paper's own numbers: its maximum EF, for scaffolds where every member is Group
1, is 1.719, and N / A = 3952 / 2298 = 1.7198, with 2298 = 1218 potent + 1080
active.

The problem with EF alone: a singleton Group 1 scaffold also scores the
maximum, exactly as high as a 30-member all-active series. So each scaffold also
gets ``ef_lower``: the Wilson score lower bound on a / n, divided by A / N. It
is what separates a real enriched series from one lucky compound, and it is
the default sort key.
"""

from __future__ import annotations

import pandas as pd

#: Columns of diversity_table, in order.
DIVERSITY_COLUMNS: tuple[str, ...] = (
    "N",
    "acyclic",
    "Ns",
    "Nss",
    "Ncsk",
    "Ns/N",
    "Nss/Ns",
    "Ncsk/N",
    "Ncsk/Ns",
)

#: Columns of enrichment_table, in order.
ENRICHMENT_COLUMNS: tuple[str, ...] = (
    "scaffold",
    "n",
    "n_group1",
    "frac_group1",
    "ef",
    "ef_lower",
    "median_pactivity",
    "max_pactivity",
)


def add_scaffolds(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of ``table`` plus ``murcko`` and ``skeleton`` columns (None if acyclic)."""
    raise NotImplementedError


def diversity_table(
    table: pd.DataFrame,
    class_col: str = "activity_class",
    classes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Table 4. Index: "complete" then each class; columns DIVERSITY_COLUMNS.

    Needs ``murcko`` and ``skeleton`` columns (see add_scaffolds). ``classes``
    fixes the row order; by default, classes in order of first appearance.

    Ns   distinct Murcko scaffolds within the row's molecules
    Nss  scaffolds with exactly one molecule *within that row* (a scaffold can
         be a singleton in one class and populated in another, which is why
         the class rows sum to more than "complete")
    Ncsk distinct skeletons
    Ratios with a zero denominator are NaN.
    """
    raise NotImplementedError


def enrichment_table(
    table: pd.DataFrame,
    scaffold_col: str = "murcko",
    group_col: str = "group",
    min_size: int = 1,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Scaffold enrichment, one row per scaffold with at least ``min_size`` members.

    Columns ENRICHMENT_COLUMNS. Sorted by ef_lower descending, then n
    descending, then scaffold, so the order is fully deterministic. Raises
    ValueError if the dataset has no Group 1 molecules (EF is undefined).
    """
    raise NotImplementedError

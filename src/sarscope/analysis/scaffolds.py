"""Murcko scaffold diversity and scaffold enrichment factors.

Scaffolds come from Sorbent: ``murcko_scaffold`` (ring systems + linkers, atom
types kept) and ``generic_scaffold`` (all atoms carbon, all bonds single - the
"cyclic skeleton"). Both return None for an acyclic molecule.

**Acyclic molecules** count towards N but have no scaffold, so they contribute
to no scaffold or skeleton count. The table reports how many there were.

**Skeleton counts are tool-specific.** "Cyclic skeleton" has no single
definition: RDKit's generic scaffold (every atom carbon, every bond single)
collapses far less aggressively than DataWarrior's, so skeleton counts from two
tools are not comparable. Murcko scaffold counts are.

**Enrichment factor.** EF = (a / n) / (A / N): the Group 1 fraction within a
scaffold over the Group 1 fraction of the whole dataset, where the whole dataset
includes acyclic molecules. A scaffold whose members are all Group 1 scores
N / A, the highest value the measure can take.

The problem with EF alone: a singleton Group 1 scaffold also scores the
maximum, exactly as high as a 30-member all-active series. So each scaffold also
gets ``ef_lower``: the Wilson score lower bound on a / n, divided by A / N. It
is what separates a real enriched series from one lucky compound, and it is
the default sort key.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import norm
from sorbent.chem.scaffolds import generic_scaffold, murcko_scaffold

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
    murcko: list[str | None] = []
    skeleton: list[str | None] = []
    for smiles in table["smiles"]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"cannot parse curated SMILES {smiles!r}")
        murcko.append(murcko_scaffold(mol))
        skeleton.append(generic_scaffold(mol))
    return table.assign(
        murcko=pd.Series(murcko, index=table.index, dtype=object),
        skeleton=pd.Series(skeleton, index=table.index, dtype=object),
    )


def _ratio(a: int, b: int) -> float:
    return a / b if b else math.nan


def _diversity_row(rows: pd.DataFrame) -> dict[str, float]:
    scaffolds = rows["murcko"].dropna()
    counts = scaffolds.value_counts()
    n, ns = len(rows), len(counts)
    nss = int((counts == 1).sum())
    ncsk = rows["skeleton"].dropna().nunique()
    return {
        "N": n,
        "acyclic": int(rows["murcko"].isna().sum()),
        "Ns": ns,
        "Nss": nss,
        "Ncsk": ncsk,
        "Ns/N": _ratio(ns, n),
        "Nss/Ns": _ratio(nss, ns),
        "Ncsk/N": _ratio(ncsk, n),
        "Ncsk/Ns": _ratio(ncsk, ns),
    }


def diversity_table(
    table: pd.DataFrame,
    class_col: str = "activity_class",
    classes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Scaffold diversity. Index: "complete" then each class; columns DIVERSITY_COLUMNS.

    Needs ``murcko`` and ``skeleton`` columns (see add_scaffolds). ``classes``
    fixes the row order; by default, classes in order of first appearance.

    Ns   distinct Murcko scaffolds within the row's molecules
    Nss  scaffolds with exactly one molecule *within that row* (a scaffold can
         be a singleton in one class and populated in another, which is why
         the class rows sum to more than "complete")
    Ncsk distinct skeletons
    Ratios with a zero denominator are NaN.
    """
    if classes is None:
        classes = tuple(pd.unique(table[class_col]))
    rows = {"complete": _diversity_row(table)}
    for name in classes:
        rows[name] = _diversity_row(table[table[class_col] == name])
    out = pd.DataFrame.from_dict(rows, orient="index")[list(DIVERSITY_COLUMNS)]
    int_cols = ["N", "acyclic", "Ns", "Nss", "Ncsk"]
    return out.astype({c: int for c in int_cols})


def _wilson_lower(successes: np.ndarray, n: np.ndarray, confidence: float) -> np.ndarray:
    z = norm.ppf(1 - (1 - confidence) / 2)
    p = successes / n
    centre = p + z**2 / (2 * n)
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (centre - margin) / (1 + z**2 / n)


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
    is_g1 = table[group_col] == 1
    base = is_g1.mean()
    if not base > 0:
        raise ValueError("no Group 1 molecules, so enrichment factors are undefined")

    has = table[table[scaffold_col].notna()].assign(_g1=is_g1)
    agg = has.groupby(scaffold_col).agg(
        n=("_g1", "size"),
        n_group1=("_g1", "sum"),
        median_pactivity=("pactivity", "median"),
        max_pactivity=("pactivity", "max"),
    )
    agg = agg[agg["n"] >= min_size]
    n = agg["n"].to_numpy(dtype=float)
    a = agg["n_group1"].to_numpy(dtype=float)
    agg["frac_group1"] = a / n
    agg["ef"] = agg["frac_group1"] / base
    agg["ef_lower"] = _wilson_lower(a, n, confidence) / base
    out = agg.reset_index().rename(columns={scaffold_col: "scaffold"})
    out = out.astype({"n": int, "n_group1": int})
    out = out.sort_values(["ef_lower", "n", "scaffold"], ascending=[False, False, True])
    return out[list(ENRICHMENT_COLUMNS)].reset_index(drop=True)

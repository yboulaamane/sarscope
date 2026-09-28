"""Structure-activity similarity (SAS) maps, SALI, activity cliffs and generators.

Every unordered pair (i, j) gets x = Tanimoto similarity and y = |pact_i -
pact_j|. With thresholds t (similarity) and a (activity), each pair falls in
one of four regions:

    cliff         x >  t  and  y >  a    similar structure, very different potency
    smooth        x >  t  and  y <= a    similar structure, similar potency
    scaffold_hop  x <= t  and  y <= a    different structure, similar potency
    nondescript   x <= t  and  y >  a    different structure, different potency

Regions are defined by meaning, not by "upper left" etc., because the paper's
prose swaps the two left-hand quadrants relative to its own axes (it plots
activity *difference* on y, then describes the lower-left as low *activity
similarity*).

**Identical fingerprints.** SALI = y / (1 - x) is undefined at x = 1. The paper
excluded such pairs ("stereoisomers are not included"). With chirality off in
the fingerprints, x = 1 is exactly where stereoisomers land, and a pair of
enantiomers two log units apart is arguably the most interesting cliff in the
dataset. So these pairs are excluded from the map and SALI, as in the paper,
but returned separately as ``identical_pairs`` instead of silently vanishing.

**Scale.** 5,000 molecules is 12.5 million pairs. Do not materialise an n x n
matrix or an index array of all pairs: walk row i against rows j > i with
Sorbent's ``bulk_tanimoto``, tally region counts, and keep only cliff and
identical pairs. Target: 5,000 molecules in well under a minute.

**Generators.** A molecule's cliff count is the number of cliff pairs it is in.
The threshold is mean + k * SD of the counts over molecules with at least one
cliff (SD with ddof=1, pandas' default). The paper confirms the population:
882 cliff pairs and a mean of 2.00 means 1,764 memberships over 882 molecules
- molecules *in* cliffs, not the whole dataset (whose mean would be 0.45).
Threshold 2.00 + 2 * 1.60 = 5.2, as reported.

**Consensus.** The paper uses "consensus" at both levels: pairs that are cliffs
under every fingerprint (section 2.4) and generators common to every
fingerprint (section 3.3, which gives its final list of sixteen). Both are
returned.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from sarscope.params import LandscapeParams

REGIONS: tuple[str, ...] = ("cliff", "smooth", "scaffold_hop", "nondescript")

#: Columns of SasResult.cliffs and SasResult.identical_pairs.
PAIR_COLUMNS: tuple[str, ...] = ("id_a", "id_b", "similarity", "delta", "sali")


@dataclass
class SasResult:
    fingerprint: str
    #: Pair counts per region, over all pairs except identical ones. Keys REGIONS.
    region_counts: dict[str, int]
    #: Cliff pairs, PAIR_COLUMNS, sorted by sali descending. id_a < id_b in
    #: (len, str) order so a pair has one spelling.
    cliffs: pd.DataFrame
    #: Pairs with similarity exactly 1.0. ``sali`` is NaN.
    identical_pairs: pd.DataFrame


def sas_map(
    fingerprints: Sequence[Any],
    pactivity: Sequence[float],
    ids: Sequence[str],
    params: LandscapeParams,
    fingerprint_name: str = "",
) -> SasResult:
    """Classify every pair. The three sequences must be the same length."""
    raise NotImplementedError


def cliff_generators(cliffs: pd.DataFrame, n_sd: float = 2.0) -> pd.DataFrame:
    """Per-molecule cliff counts.

    Columns: molecule_id, n_cliffs, is_generator. Only molecules in at least
    one cliff appear. Sorted by n_cliffs descending, then molecule_id. The
    threshold used is stored in ``result.attrs["threshold"]``. With fewer than
    two molecules the SD is undefined: no generators, threshold NaN.
    """
    raise NotImplementedError


def consensus(
    results: Mapping[str, SasResult], n_sd: float = 2.0
) -> tuple[pd.DataFrame, list[str]]:
    """(pairs that are cliffs under every fingerprint, generators common to all).

    Generators are computed per fingerprint with ``cliff_generators`` and then
    intersected; the list is sorted. An empty mapping is a ValueError.
    """
    raise NotImplementedError

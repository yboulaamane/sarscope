"""Structure-activity similarity (SAS) maps, SALI, activity cliffs and generators.

Every unordered pair (i, j) gets x = Tanimoto similarity and y = |pact_i -
pact_j|. With thresholds t (similarity) and a (activity), each pair falls in
one of four regions:

    cliff         x >  t  and  y >  a    similar structure, very different potency
    smooth        x >  t  and  y <= a    similar structure, similar potency
    scaffold_hop  x <= t  and  y <= a    different structure, similar potency
    nondescript   x <= t  and  y >  a    different structure, different potency

Regions are named by meaning rather than by quadrant position ("upper left"
and so on), because which corner a region occupies depends on whether the axis
plots similarity or distance, and descriptions in the literature disagree.

**Identical fingerprints.** SALI = y / (1 - x) is undefined at x = 1, so such
pairs cannot appear on the map. They are not discarded, though: with chirality
off in the fingerprints, x = 1 is exactly where stereoisomers land, and a pair
of enantiomers two log units apart is arguably the most interesting cliff in
the dataset. They come back as ``identical_pairs`` rather than vanishing.

**Scale.** 5,000 molecules is 12.5 million pairs. Do not materialise an n x n
matrix or an index array of all pairs: walk row i against rows j > i with
Sorbent's ``bulk_tanimoto``, tally region counts, and keep only cliff and
identical pairs. Target: 5,000 molecules in well under a minute.

**Generators.** A molecule's cliff count is the number of cliff pairs it is in.
The threshold is mean + k * SD of those counts, taken over molecules with at
least one cliff rather than over the whole dataset - otherwise the mean is
dominated by molecules in no cliffs at all and the threshold collapses.

**Consensus.** Which pairs are cliffs depends on the fingerprint, so both
levels of agreement are returned: pairs that are cliffs under every
fingerprint, and generators that are generators under every fingerprint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sorbent.chem.fingerprints import bulk_tanimoto

from sarscope.curate import id_order
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
    n = len(fingerprints)
    if not len(pactivity) == len(ids) == n:
        raise ValueError(
            f"lengths differ: {n} fingerprints, {len(pactivity)} potencies, {len(ids)} ids"
        )
    t, a = params.similarity_threshold, params.activity_threshold
    potency = np.asarray(pactivity, dtype=float)
    counts = dict.fromkeys(REGIONS, 0)
    cliffs: list[tuple[str, str, float, float, float]] = []
    identical: list[tuple[str, str, float, float, float]] = []

    def spell(i: int, j: int) -> tuple[str, str]:
        x, y = str(ids[i]), str(ids[j])
        return (x, y) if id_order(x) <= id_order(y) else (y, x)

    for i in range(n - 1):
        sims = np.asarray(bulk_tanimoto(fingerprints[i], list(fingerprints[i + 1 :])))
        deltas = np.abs(potency[i + 1 :] - potency[i])
        same = sims >= 1.0
        similar = (sims > t) & ~same
        dissimilar = (sims <= t) & ~same
        big = deltas > a
        counts["cliff"] += int((similar & big).sum())
        counts["smooth"] += int((similar & ~big).sum())
        counts["scaffold_hop"] += int((dissimilar & ~big).sum())
        counts["nondescript"] += int((dissimilar & big).sum())
        for k in np.flatnonzero(similar & big):
            sim, delta = float(sims[k]), float(deltas[k])
            cliffs.append((*spell(i, i + 1 + int(k)), sim, delta, delta / (1.0 - sim)))
        for k in np.flatnonzero(same):
            identical.append((*spell(i, i + 1 + int(k)), 1.0, float(deltas[k]), float("nan")))

    cliff_frame = pd.DataFrame(cliffs, columns=list(PAIR_COLUMNS))
    cliff_frame = cliff_frame.sort_values(
        ["sali", "id_a", "id_b"], ascending=[False, True, True]
    ).reset_index(drop=True)
    return SasResult(
        fingerprint=fingerprint_name,
        region_counts=counts,
        cliffs=cliff_frame,
        identical_pairs=pd.DataFrame(identical, columns=list(PAIR_COLUMNS)),
    )


def cliff_generators(cliffs: pd.DataFrame, n_sd: float = 2.0) -> pd.DataFrame:
    """Per-molecule cliff counts.

    Columns: molecule_id, n_cliffs, is_generator. Only molecules in at least
    one cliff appear. Sorted by n_cliffs descending, then molecule_id. The
    threshold used is stored in ``result.attrs["threshold"]``. With fewer than
    two molecules the SD is undefined: no generators, threshold NaN.
    """
    counts = pd.concat([cliffs["id_a"], cliffs["id_b"]]).value_counts()
    threshold = counts.mean() + n_sd * counts.std() if len(counts) >= 2 else float("nan")
    out = pd.DataFrame(
        {
            "molecule_id": counts.index.astype(str),
            "n_cliffs": counts.to_numpy(dtype=int),
        }
    )
    out["is_generator"] = out["n_cliffs"] > threshold  # NaN compares False
    out["_order"] = out["molecule_id"].map(id_order)
    out = out.sort_values(["n_cliffs", "_order"], ascending=[False, True]).drop(columns="_order")
    out = out.reset_index(drop=True)
    out.attrs["threshold"] = float(threshold)
    return out


def consensus(
    results: Mapping[str, SasResult], n_sd: float = 2.0
) -> tuple[pd.DataFrame, list[str]]:
    """(pairs that are cliffs under every fingerprint, generators common to all).

    Generators are computed per fingerprint with ``cliff_generators`` and then
    intersected; the list is sorted. An empty mapping is a ValueError.
    Pair columns: id_a, id_b, delta, then ``similarity_<fingerprint>`` for each.
    """
    if not results:
        raise ValueError("consensus needs at least one SAS map")

    def key(frame: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [frozenset(p) for p in zip(frame["id_a"], frame["id_b"], strict=True)],
            index=frame.index,
        )

    shared = set.intersection(*(set(key(r.cliffs)) for r in results.values()))
    merged: pd.DataFrame | None = None
    for name, result in results.items():
        frame = result.cliffs[key(result.cliffs).isin(shared)]
        frame = frame.assign(_pair=key(frame))
        part = frame[["_pair", "id_a", "id_b", "delta", "similarity"]].rename(
            columns={"similarity": f"similarity_{name}"}
        )
        if merged is None:
            merged = part
        else:
            merged = merged.merge(part[["_pair", f"similarity_{name}"]], on="_pair")
    assert merged is not None
    pairs = merged.drop(columns="_pair").sort_values(["id_a", "id_b"]).reset_index(drop=True)

    generator_sets = []
    for result in results.values():
        gens = cliff_generators(result.cliffs, n_sd)
        generator_sets.append(set(gens.loc[gens["is_generator"], "molecule_id"]))
    generators = sorted(set.intersection(*generator_sets), key=id_order)
    return pairs, generators

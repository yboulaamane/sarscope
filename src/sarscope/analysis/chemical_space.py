"""Opt-in structural chemical-space visualization, separate from QSAR features."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sarscope.analysis.features import fingerprint_matrix


@dataclass
class StructuralSpace:
    points: pd.DataFrame
    settings: dict[str, str | int | float]


def ecfp4_umap(
    table: pd.DataFrame,
    *,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    max_molecules: int = 2000,
    seed: int = 42,
) -> StructuralSpace:
    """Embed binary Morgan radius-2/2048-bit fingerprints with Jaccard distance.

    Sampling is uniform, deterministic and independent of activity. No labels
    enter the embedding; the projection is exploratory, not a model or domain
    test. Import UMAP only when this step is explicitly requested.
    """
    if len(table) < 3:
        raise ValueError("UMAP needs at least three curated molecules.")
    if n_neighbors < 2 or not 0 <= min_dist <= 1 or max_molecules < 3:
        raise ValueError("Require neighbors >= 2, min_dist in [0, 1], and sample cap >= 3.")
    try:
        from umap import UMAP
    except ImportError as exc:
        raise ImportError("Install umap-learn or the 'sarscope[space]' extra for UMAP.") from exc

    positions = np.arange(len(table))
    if len(table) > max_molecules:
        positions = np.sort(
            np.random.default_rng(seed).choice(len(table), size=max_molecules, replace=False)
        )
    selected = table.iloc[positions]
    matrix = fingerprint_matrix(selected["smiles"].astype(str).tolist(), "ecfp4").astype(bool)
    neighbors = min(n_neighbors, len(selected) - 1)
    coordinates = UMAP(
        n_components=2,
        metric="jaccard",
        n_neighbors=neighbors,
        min_dist=min_dist,
        random_state=seed,
        n_jobs=1,
        init="random",  # also works for three-molecule datasets
    ).fit_transform(matrix)
    points = selected[["molecule_id", "smiles", "pactivity", "activity_class"]].copy()
    points["UMAP1"] = coordinates[:, 0]
    points["UMAP2"] = coordinates[:, 1]
    points["embedded"] = np.isfinite(coordinates).all(axis=1)
    if not points["embedded"].any():
        raise ValueError("UMAP found no connected fingerprint neighborhoods to plot.")
    return StructuralSpace(
        points,
        {
            "fingerprint": "ECFP4 (Morgan radius 2; chirality off)",
            "bits": 2048,
            "metric": "jaccard",
            "n_neighbors": neighbors,
            "min_dist": min_dist,
            "seed": seed,
            "sampling": "uniform without replacement when above cap",
            "max_molecules": max_molecules,
            "total_molecules": len(table),
            "sampled_molecules": len(points),
            "plotted_molecules": int(points["embedded"].sum()),
            "unembedded_molecules": int((~points["embedded"]).sum()),
        },
    )

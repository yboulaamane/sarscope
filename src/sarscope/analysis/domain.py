"""Applicability domain: the PCA bounding box.

Fit PCA on the *training* feature matrix (after the feature filter, which is
fitted on the training set too), take the per-component [min, max] of the
training scores, and call a query compound in-domain if every one of its
component scores falls inside that box, bounds inclusive.

This is the weakest of the common AD methods - a box in two dimensions is
generous, and a molecule can sit inside it while being far from every training
compound. It is here because it is what most published QSAR work reports, so
it makes results comparable; the report says plainly what it does and does not
establish. Treat a high coverage number as a floor, not a guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.decomposition import PCA


@dataclass
class DomainResult:
    in_domain: NDArray[np.bool_]  # one per query row
    train_scores: NDArray[np.float64]  # (n_train, n_components), for the plot
    query_scores: NDArray[np.float64]  # (n_query, n_components)

    @property
    def coverage(self) -> float:
        """Fraction of query compounds inside the box; NaN for no queries."""
        return float(self.in_domain.mean()) if self.in_domain.size else float("nan")


def pca_bounding_box(
    X_train: NDArray[Any], X_query: NDArray[Any], n_components: int = 2
) -> DomainResult:
    """Every training compound is in its own box, by construction; test that."""
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)
    k = min(n_components, X_train.shape[1], X_train.shape[0])
    pca = PCA(n_components=k).fit(X_train)
    train_scores = pca.transform(X_train)
    if X_query.shape[0] == 0:
        query_scores = np.empty((0, k))
    else:
        query_scores = pca.transform(X_query)
    # A hair of tolerance: float round-off must not push a training row out of its own box.
    lo = train_scores.min(axis=0) - 1e-9
    hi = train_scores.max(axis=0) + 1e-9
    inside = np.all((query_scores >= lo) & (query_scores <= hi), axis=1)
    return DomainResult(inside, train_scores, query_scores)

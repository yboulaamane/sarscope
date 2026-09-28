"""Applicability domain: the PCA bounding box, as in the paper's Fig. 10.

Fit PCA on the *training* feature matrix (after the feature filter, which is
fitted on the training set too), take the per-component [min, max] of the
training scores, and call a query compound in-domain if every one of its
component scores falls inside that box, bounds inclusive.

This is the weakest common AD method - a box in two dimensions is generous,
and a molecule can sit inside it while being far from every training compound.
It is implemented because the paper uses it; the report states what it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


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
    raise NotImplementedError

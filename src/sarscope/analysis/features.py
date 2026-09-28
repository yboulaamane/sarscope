"""Fingerprints, and the feature filter that has to live inside cross-validation.

Fingerprints

    ecfp4    Morgan radius 2, ``FeatureParams.ecfp_bits`` bits, via Sorbent's
             ``compute_fingerprint`` (chirality off, so enantiomers are
             identical - see landscape.py for why that matters).
    maccs    ``rdkit.Chem.MACCSkeys.GenMACCSKeys``. RDKit returns 167 bits with
             bit 0 unused (checked: never set); drop it to get the standard 166.
    pubchem  881 bits via scikit-fingerprints (``pip install sarscope[pubchem]``).
             The paper used PaDEL, which needs Java. Whether scikit-
             fingerprints matches PaDEL bit-for-bit is NOT verified; do that on
             a sample before claiming the paper's PubChem results reproduce.
             Raise ImportError naming the extra when it is missing.

The feature filter

The paper drops features with variance < 0.1, then one of each pair correlated
above 0.95. On binary bits, variance is p(1 - p), so a 0.1 threshold keeps only
bits set in roughly 11-89% of molecules - an aggressive cut, which is why 881
PubChem bits fall to about a hundred.

It is a scikit-learn transformer rather than a function for one reason: which
features survive depends on the data, so choosing them on the full dataset and
then cross-validating leaks the held-out folds into the model. As a
transformer it is re-fitted inside every fold by model.py.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin

from sarscope.params import FingerprintName


def bit_vectors(
    smiles: Sequence[str], name: FingerprintName, *, ecfp_bits: int = 2048
) -> list[Any]:
    """RDKit ExplicitBitVects, for Tanimoto similarity in landscape.py.

    For "pubchem", build ExplicitBitVects from the numpy rows so that callers
    get one type regardless of fingerprint. Invalid SMILES is a ValueError:
    the input is curated, so it indicates a bug upstream.
    """
    raise NotImplementedError


def fingerprint_matrix(
    smiles: Sequence[str], name: FingerprintName, *, ecfp_bits: int = 2048
) -> NDArray[np.uint8]:
    """Shape (n_molecules, n_bits), values 0/1. Widths: ecfp_bits, 166, 881."""
    raise NotImplementedError


class VarianceCorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop low-variance features, then highly correlated ones.

    Variance: keep features with variance strictly greater than the threshold
    (``sklearn.feature_selection.VarianceThreshold`` semantics).

    Correlation: among the survivors, scan columns in order; drop a column if
    its absolute Pearson correlation with any *kept* earlier column exceeds the
    threshold. Greedy and order-dependent, but deterministic.

    After ``fit``: ``support_`` is a boolean mask over the input columns, and
    ``n_after_variance_`` / ``n_after_correlation_`` record the counts for the
    report. Transforming a matrix of a different width is a ValueError.
    """

    def __init__(
        self, variance_threshold: float = 0.1, correlation_threshold: float = 0.95
    ) -> None:
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold

    def fit(self, X: NDArray[Any], y: Any = None) -> VarianceCorrelationFilter:
        raise NotImplementedError

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        raise NotImplementedError

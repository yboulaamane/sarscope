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
from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys
from sklearn.base import BaseEstimator, TransformerMixin
from sorbent.chem.fingerprints import compute_fingerprint

from sarscope.params import FingerprintName

PUBCHEM_BITS = 881


def _mols(smiles: Sequence[str]) -> list[Any]:
    mols = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"cannot parse SMILES {smi!r}")
        mols.append(mol)
    return mols


def _pubchem_matrix(smiles: Sequence[str]) -> NDArray[np.uint8]:
    try:
        from skfp.fingerprints import PubChemFingerprint
    except ImportError as exc:
        raise ImportError(
            "PubChem fingerprints need scikit-fingerprints: pip install 'sarscope[pubchem]'"
        ) from exc
    _mols(smiles)  # same ValueError contract as the other fingerprints
    X = PubChemFingerprint().transform(list(smiles))
    return (np.asarray(X) > 0).astype(np.uint8)


def _to_numpy(fps: Sequence[Any]) -> NDArray[np.uint8]:
    if not fps:
        return np.zeros((0, 0), dtype=np.uint8)
    out = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.uint8)
    for row, fp in zip(out, fps, strict=True):
        DataStructs.ConvertToNumpyArray(fp, row)
    return out


def _to_bitvect(row: NDArray[Any]) -> Any:
    vect = DataStructs.ExplicitBitVect(len(row))
    for bit in np.flatnonzero(row):
        vect.SetBit(int(bit))
    return vect


def bit_vectors(
    smiles: Sequence[str], name: FingerprintName, *, ecfp_bits: int = 2048
) -> list[Any]:
    """RDKit ExplicitBitVects, for Tanimoto similarity in landscape.py.

    For "pubchem", build ExplicitBitVects from the numpy rows so that callers
    get one type regardless of fingerprint. Invalid SMILES is a ValueError:
    the input is curated, so it indicates a bug upstream.
    """
    if name == "pubchem":
        return [_to_bitvect(row) for row in _pubchem_matrix(smiles)]
    mols = _mols(smiles)
    if name == "ecfp4":
        return [compute_fingerprint(m, radius=2, n_bits=ecfp_bits) for m in mols]
    if name == "maccs":
        # Drop RDKit's unused bit 0 so widths and Tanimoto match the 166-key standard.
        maccs = _to_numpy([MACCSkeys.GenMACCSKeys(m) for m in mols])[:, 1:]
        return [_to_bitvect(row) for row in maccs]
    raise ValueError(f"unknown fingerprint {name!r}")


def fingerprint_matrix(
    smiles: Sequence[str], name: FingerprintName, *, ecfp_bits: int = 2048
) -> NDArray[np.uint8]:
    """Shape (n_molecules, n_bits), values 0/1. Widths: ecfp_bits, 166, 881."""
    if name == "pubchem":
        return _pubchem_matrix(smiles)
    return _to_numpy(bit_vectors(smiles, name, ecfp_bits=ecfp_bits))


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
        X = np.asarray(X, dtype=float)
        variance_ok = X.var(axis=0) > self.variance_threshold
        survivors = np.flatnonzero(variance_ok)

        kept: list[int] = []
        if survivors.size:
            corr = np.abs(np.corrcoef(X[:, survivors], rowvar=False).reshape(survivors.size, -1))
            for i in range(survivors.size):
                if not kept or corr[i, kept].max() <= self.correlation_threshold:
                    kept.append(i)

        support = np.zeros(X.shape[1], dtype=bool)
        support[survivors[kept]] = True
        self.support_ = support
        self.n_features_in_ = X.shape[1]
        self.n_after_variance_ = int(survivors.size)
        self.n_after_correlation_ = len(kept)
        return self

    def transform(self, X: NDArray[Any]) -> NDArray[Any]:
        X = np.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"fitted on {self.n_features_in_} features, got {X.shape[1]}")
        return X[:, self.support_]

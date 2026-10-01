"""Specification for sarscope.analysis.features."""

import importlib.util

import numpy as np
import pytest
from rdkit import DataStructs

from sarscope.analysis.features import (
    VarianceCorrelationFilter,
    bit_vectors,
    fingerprint_matrix,
)

pytestmark = pytest.mark.science

SMILES = ["CCO", "c1ccccc1O", "CC(=O)Oc1ccccc1C(=O)O", "CN1CCC[C@H]1c1cccnc1"]


@pytest.mark.parametrize(("name", "width"), [("ecfp4", 1024), ("maccs", 166)])
def test_matrix_shape_and_values(name, width):
    X = fingerprint_matrix(SMILES, name, ecfp_bits=1024)
    assert X.shape == (len(SMILES), width)
    assert X.dtype == np.uint8
    assert set(np.unique(X)) <= {0, 1}


def test_matrix_and_bit_vectors_agree():
    X = fingerprint_matrix(SMILES, "maccs")
    fps = bit_vectors(SMILES, "maccs")
    assert [fp.GetNumOnBits() for fp in fps] == X.sum(axis=1).tolist()


def test_enantiomers_are_identical_under_ecfp4():
    a, b = bit_vectors(["CN1CCC[C@H]1c1cccnc1", "CN1CCC[C@@H]1c1cccnc1"], "ecfp4")
    assert DataStructs.TanimotoSimilarity(a, b) == 1.0


def test_invalid_smiles_is_an_error():
    with pytest.raises(ValueError):
        fingerprint_matrix(["CCO", "C1CC"], "ecfp4")


@pytest.mark.skipif(
    importlib.util.find_spec("skfp") is not None, reason="scikit-fingerprints installed"
)
def test_pubchem_without_the_extra_says_how_to_get_it():
    with pytest.raises(ImportError, match=r"sarscope\[pubchem\]"):
        fingerprint_matrix(SMILES, "pubchem")


def test_filter_drops_low_variance_then_correlated_columns():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 2, (200, 3))
    X = np.column_stack(
        [
            base[:, 0],
            np.zeros(200),  # variance 0
            base[:, 1],
            base[:, 0],  # duplicate of column 0 -> correlated, later -> dropped
            (rng.random(200) < 0.05).astype(int),  # rare bit, variance < 0.1
            base[:, 2],
        ]
    )
    f = VarianceCorrelationFilter(0.1, 0.95).fit(X)
    assert f.support_.tolist() == [True, False, True, False, False, True]
    assert (f.n_after_variance_, f.n_after_correlation_) == (4, 3)
    assert f.transform(X).shape == (200, 3)


def test_filter_does_not_refit_on_transform():
    rng = np.random.default_rng(1)
    train = rng.integers(0, 2, (100, 5))
    f = VarianceCorrelationFilter().fit(train)
    other = np.zeros((10, 5))
    assert f.transform(other).shape == (10, int(f.support_.sum()))


def test_filter_rejects_a_different_width():
    f = VarianceCorrelationFilter().fit(np.random.default_rng(2).integers(0, 2, (50, 4)))
    with pytest.raises(ValueError):
        f.transform(np.zeros((5, 3)))


def test_filter_is_a_proper_sklearn_estimator():
    from sklearn.base import clone

    f = clone(VarianceCorrelationFilter(0.2, 0.9))
    assert f.get_params() == {
        "variance_threshold": 0.2,
        "correlation_threshold": 0.9,
        "continuous": False,
    }

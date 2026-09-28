"""Specification for sarscope.analysis.domain."""

import numpy as np
import pytest

from sarscope.analysis.domain import pca_bounding_box

pytestmark = pytest.mark.science


def test_training_compounds_are_inside_their_own_box():
    X = np.random.default_rng(0).integers(0, 2, (80, 30)).astype(float)
    result = pca_bounding_box(X, X)
    assert result.in_domain.all()
    assert result.coverage == 1.0


def test_a_far_compound_is_outside():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (100, 10))
    far = np.full((1, 10), 50.0)
    result = pca_bounding_box(X, np.vstack([X[:5], far]))
    assert result.in_domain.tolist() == [True] * 5 + [False]
    assert result.coverage == pytest.approx(5 / 6)


def test_shapes():
    rng = np.random.default_rng(2)
    result = pca_bounding_box(rng.normal(size=(40, 8)), rng.normal(size=(7, 8)), n_components=3)
    assert result.train_scores.shape == (40, 3)
    assert result.query_scores.shape == (7, 3)


def test_empty_query_has_nan_coverage():
    result = pca_bounding_box(np.random.default_rng(3).normal(size=(20, 4)), np.empty((0, 4)))
    assert np.isnan(result.coverage)

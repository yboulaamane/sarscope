import builtins
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sarscope.analysis.chemical_space import ecfp4_umap


def molecules(n=12):
    return pd.DataFrame(
        {
            "molecule_id": [f"m{i}" for i in range(n)],
            "smiles": ["C" * (i + 1) for i in range(n)],
            "pactivity": np.linspace(5, 9, n),
            "activity_class": ["active"] * n,
        },
        index=np.arange(n) * 2 + 100,
    )


def test_umap_passes_binary_ecfp4_jaccard_and_samples_without_labels(monkeypatch):
    seen = []

    class FakeUMAP:
        def __init__(self, **settings):
            self.settings = settings

        def fit_transform(self, matrix):
            seen.append((self.settings, matrix.copy()))
            return np.arange(len(matrix) * 2).reshape(-1, 2)

    monkeypatch.setitem(sys.modules, "umap", SimpleNamespace(UMAP=FakeUMAP))
    table = molecules()
    result = ecfp4_umap(table, max_molecules=5, n_neighbors=15)
    other = ecfp4_umap(table.assign(pactivity=0.0, activity_class="inactive"), max_molecules=5)
    assert result.points["molecule_id"].tolist() == other.points["molecule_id"].tolist()
    np.testing.assert_array_equal(seen[0][1], seen[1][1])
    assert seen[0][1].shape == (5, 2048)
    assert seen[0][1].dtype == np.bool_
    assert seen[0][0]["metric"] == "jaccard"
    assert seen[0][0]["n_neighbors"] == 4
    assert seen[0][0]["n_jobs"] == 1
    assert result.settings["total_molecules"] == 12
    assert result.settings["plotted_molecules"] == 5
    pd.testing.assert_series_equal(
        result.points["smiles"], table.loc[result.points.index, "smiles"]
    )


@pytest.mark.parametrize("kwargs", [{"n_neighbors": 1}, {"min_dist": 2}, {"max_molecules": 2}])
def test_invalid_umap_settings(kwargs):
    with pytest.raises(ValueError):
        ecfp4_umap(molecules(), **kwargs)


def test_too_few_molecules():
    with pytest.raises(ValueError, match="three"):
        ecfp4_umap(molecules(2))


def test_missing_umap_is_actionable(monkeypatch):
    original = builtins.__import__

    def missing(name, *args, **kwargs):
        if name == "umap":
            raise ImportError("missing")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="umap-learn"):
        ecfp4_umap(molecules())


def test_real_umap_finite_coordinates_and_small_input():
    pytest.importorskip("umap")
    for n in (3, 12):
        result = ecfp4_umap(molecules(n))
        assert result.points.shape == (n, 7)
        finite = np.isfinite(result.points[["UMAP1", "UMAP2"]].to_numpy()).all(axis=1)
        np.testing.assert_array_equal(result.points["embedded"], finite)
        assert result.settings["unembedded_molecules"] == int((~finite).sum())
        assert result.settings["plotted_molecules"] == int(finite.sum())
        assert finite.any()

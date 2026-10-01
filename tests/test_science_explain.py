import numpy as np
import pandas as pd
import pytest

from sarscope.analysis.explain import (
    EXPLANATION_DESCRIPTORS,
    add_explanation_descriptors,
    explain_descriptor_model,
)

pytestmark = pytest.mark.science


SMILES = [
    "Cc1ccccc1",
    "CCc1ccccc1",
    "COc1ccccc1",
    "Nc1ccccc1",
    "Clc1ccccc1",
    "Fc1ccccc1",
    "N#Cc1ccccc1",
    "O=C(O)c1ccccc1",
    "Cc1ccncc1",
    "CCc1ccncc1",
    "COc1ccncc1",
    "Nc1ccncc1",
]


def table():
    return pd.DataFrame(
        {
            "molecule_id": [f"m{i}" for i in range(len(SMILES))],
            "smiles": SMILES,
            "pactivity": np.linspace(5.0, 8.0, len(SMILES)),
        }
    )


def test_named_rdkit_descriptors_are_finite():
    result = add_explanation_descriptors(table())
    assert set(EXPLANATION_DESCRIPTORS) <= set(result)
    assert np.isfinite(result[list(EXPLANATION_DESCRIPTORS)].to_numpy()).all()


@pytest.mark.parametrize("algorithm", ["random_forest", "mlp"])
def test_descriptor_explanation_uses_held_out_rows(algorithm):
    result = explain_descriptor_model(
        table(),
        np.arange(9),
        np.arange(9, 12),
        algorithm=algorithm,
        permutation_repeats=2,
    )
    assert result.metrics.iloc[0]["n_test"] == 3
    assert len(result.predictions) == 3
    assert set(result.permutation["descriptor"]) == set(EXPLANATION_DESCRIPTORS)
    if algorithm == "random_forest":
        assert not result.intrinsic.empty
    else:
        assert result.intrinsic.empty


def test_shap_is_optional_and_reports_how_to_enable_it(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_shap(name, *args, **kwargs):
        if name == "shap":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_shap)
    result = explain_descriptor_model(
        table(),
        np.arange(9),
        np.arange(9, 12),
        compute_shap=True,
        permutation_repeats=2,
    )
    assert "sarscope[explain]" in result.shap_status


def test_selected_descriptor_explanation_uses_requested_names():
    names = ("MolWt", "TPSA", "NumAromaticRings", "qed")
    result = explain_descriptor_model(
        table(),
        np.arange(9),
        np.arange(9, 12),
        descriptor_names=names,
        permutation_repeats=2,
    )
    assert set(result.permutation["descriptor"]) == set(names)
    assert set(result.intrinsic["descriptor"]) == set(names)
    assert result.predictions["molecule_id"].tolist() == ["m9", "m10", "m11"]


def test_real_treeshap_additivity_on_held_out_predictions():
    pytest.importorskip("shap")
    result = explain_descriptor_model(
        table(),
        np.arange(9),
        np.arange(9, 12),
        compute_shap=True,
        shap_max_samples=2,
        permutation_repeats=2,
    )
    assert len(result.shap_local) == 2
    assert result.shap_local["molecule_id"].tolist() == ["m9", "m10"]
    reconstructed = (
        result.shap_local[list(EXPLANATION_DESCRIPTORS)].sum(axis=1).to_numpy()
        + result.shap_base_value
    )
    np.testing.assert_allclose(
        reconstructed,
        result.predictions["predicted_pactivity"].to_numpy()[:2],
        atol=1e-6,
    )

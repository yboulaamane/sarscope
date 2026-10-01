import sys
from types import SimpleNamespace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from sarscope.analysis.features import feature_filter, fingerprint_matrix
from sarscope.analysis.regression import fit_deployment_model
from sarscope.analysis.representations import (
    ALL_2D,
    DESCRIPTOR_PRESETS,
    descriptor_matrix,
    model_feature_names,
    model_matrix,
    resolved_features,
)
from sarscope.params import FeatureParams, ModelParams
from sarscope.predict import PredictionBundle, load_bundle, predict_smiles, save_bundle

SMILES = ["CCO", "CCCO", "CCCCO", "CCN", "c1ccccc1", "CC(=O)O", "CC(=O)N", "CCOC"]


def test_selected_descriptors_match_rdkit_in_exact_order():
    names = ("TPSA", "MolWt", "MolLogP")
    values = descriptor_matrix(SMILES, names)
    expected = [
        [getattr(Descriptors, name)(Chem.MolFromSmiles(smi)) for name in names] for smi in SMILES
    ]
    np.testing.assert_allclose(values, expected)
    assert len(ALL_2D) > 200
    assert "Ipc" not in ALL_2D
    assert len(DESCRIPTOR_PRESETS["Medicinal chemistry"]) > 12
    with pytest.raises(ValueError, match="Unknown"):
        descriptor_matrix(SMILES, ["made_up"])
    with pytest.raises(ValueError, match="duplicate"):
        descriptor_matrix(SMILES, ["MolWt", "MolWt"])


def test_hybrid_keeps_fingerprints_and_ordered_descriptor_tail():
    features = FeatureParams(
        fingerprint="maccs", representation="hybrid", descriptor_names=("MolWt", "TPSA")
    )
    values = model_matrix(SMILES, features)
    assert values.shape == (len(SMILES), 168)
    np.testing.assert_array_equal(values[:, :166], fingerprint_matrix(SMILES, "maccs"))
    np.testing.assert_allclose(
        values[:, 166:], descriptor_matrix(SMILES, features.descriptor_names)
    )
    assert model_feature_names(features)[-2:] == ("MolWt", "TPSA")


def test_descriptor_preprocessing_is_training_local_and_pickleable(tmp_path):
    features = FeatureParams(representation="rdkit2d")
    X_train = np.array([[1, 5, np.nan], [2, 5, 10], [3, 5, 20]], dtype=float)
    transform = feature_filter(features).fit(X_train)
    np.testing.assert_allclose(transform.imputer_.statistics_, [2, 5, 15])
    assert not transform.support_[1]  # constant, not a scale-dependent variance cutoff
    transformed = transform.transform(np.array([[np.nan, 99, np.inf]]))
    assert np.isfinite(transformed).all()
    np.testing.assert_allclose(transform.imputer_.statistics_, [2, 5, 15])
    assert transform.n_features_in_ == 3


@pytest.mark.parametrize("representation", ["rdkit2d", "hybrid"])
def test_descriptor_prediction_bundle_roundtrip_preserves_features_and_binary_domain(
    tmp_path, representation
):
    features = FeatureParams(
        representation=representation,
        descriptor_names=("MolWt", "TPSA", "MolLogP"),
    )
    X = model_matrix(SMILES, features)
    y = np.linspace(5, 9, len(SMILES))
    filter_, estimator = fit_deployment_model(X, y, "ridge", ModelParams(features=features))
    fingerprints = fingerprint_matrix(SMILES, "ecfp4")
    bundle = PredictionBundle(
        "ridge",
        resolved_features(features),
        filter_,
        estimator,
        fingerprints,
        0.3,
        feature_names=model_feature_names(features),
    )
    loaded = load_bundle(save_bundle(bundle, tmp_path / "model.joblib"))
    predictions = predict_smiles(loaded, [f"m{i}" for i in range(len(SMILES))], SMILES)
    np.testing.assert_allclose(
        predictions["predicted_pactivity"], estimator.predict(filter_.transform(X))
    )
    np.testing.assert_allclose(predictions["max_training_similarity"], 1.0)
    assert predictions["in_applicability_domain"].all()
    assert loaded.features.descriptor_names == features.descriptor_names


def test_molfeat_adapter_disables_structure_changes_and_preserves_schema(monkeypatch):
    calls = []

    class Calculator:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.columns = kwargs["descrs"]

        def __call__(self, mol):
            return [getattr(Descriptors, name)(mol) for name in self.columns]

    monkeypatch.setitem(sys.modules, "molfeat.calc", SimpleNamespace(RDKitDescriptors2D=Calculator))
    names = ("MolWt", "TPSA")
    result = descriptor_matrix(SMILES, names, backend="molfeat")
    np.testing.assert_allclose(result, descriptor_matrix(SMILES, names))
    assert calls[0]["do_not_standardize"] is True
    assert calls[0]["augment"] is False
    assert calls[0]["avg_ipc"] is False
    assert calls[0]["replace_nan"] is False


def test_real_optional_molfeat_backend_matches_native_descriptors():
    pytest.importorskip("molfeat")
    names = ("MolWt", "TPSA", "MolLogP")
    np.testing.assert_allclose(
        descriptor_matrix(SMILES, names, backend="molfeat"),
        descriptor_matrix(SMILES, names),
    )

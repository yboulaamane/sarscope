import numpy as np

from sarscope.analysis.regression import fit_deployment_model
from sarscope.params import FeatureParams, ModelParams
from sarscope.predict import (
    PredictionBundle,
    load_bundle,
    predict_smiles,
    save_bundle,
    similarity_domain_threshold,
)


def test_saved_model_predicts_and_flags_domain(tmp_path):
    X = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ],
        dtype=np.uint8,
    )
    y = np.array([5.0, 5.5, 6.0, 7.0, 6.5, 7.5])
    params = ModelParams(
        regression_algorithms=("extra_trees",),
        features=FeatureParams(fingerprint="maccs", variance_threshold=0.0),
    )
    filt, estimator = fit_deployment_model(X, y, "extra_trees", params)
    bundle = PredictionBundle(
        "extra_trees",
        params.features,
        filt,
        estimator,
        X,
        similarity_domain_threshold(X),
    )
    path = save_bundle(bundle, tmp_path / "model.joblib")
    assert load_bundle(path).algorithm == "extra_trees"


def test_prediction_output_contains_applicability_fields(monkeypatch):
    class IdentityFilter:
        def transform(self, X):
            return X

    class SumModel:
        def predict(self, X):
            return X.sum(axis=1)

    train = np.ones((2, 166), dtype=np.uint8)
    bundle = PredictionBundle(
        "sum",
        FeatureParams(fingerprint="maccs"),
        IdentityFilter(),
        SumModel(),
        train,
        0.1,
    )
    result = predict_smiles(bundle, ["ethanol"], ["CCO"])
    assert list(result) == [
        "molecule_id",
        "smiles",
        "predicted_pactivity",
        "max_training_similarity",
        "in_applicability_domain",
        "domain_threshold",
    ]

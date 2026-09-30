import numpy as np
import pandas as pd
import pytest

from sarscope.analysis.features import fingerprint_matrix
from sarscope.params import FeatureParams
from sarscope.predict import PredictionBundle
from sarscope.prioritise import diverse_shortlist, predict_candidates


class IdentityFilter:
    def transform(self, values):
        return values


class SimpleModel:
    def predict(self, values):
        return 5.0 + values[:, :10].sum(axis=1) * 0.1


@pytest.fixture
def bundle():
    smiles = ["CCO", "CCN"]
    features = FeatureParams(fingerprint="maccs")
    return PredictionBundle(
        "simple",
        features,
        IdentityFilter(),
        SimpleModel(),
        fingerprint_matrix(smiles, "maccs"),
        0.0,
        train_ids=["CHEMBL1", "CHEMBL2"],
        train_smiles=smiles,
        train_pactivity=[6.0, 7.0],
    )


def test_upload_keeps_invalid_rows_and_shows_measured_analogue(bundle):
    source = pd.DataFrame(
        {
            "molecule_id": ["test_1", "bad", "test_3", "repeat"],
            "smiles": ["CCO", "not_smiles", "Cc1ccccc1", "CCO"],
        }
    )
    result = predict_candidates(bundle, source)
    assert len(result) == 4
    assert result.loc[0, "canonical_smiles"] == "CCO"
    assert result.loc[0, "nearest_training_id"] == "CHEMBL1"
    assert result.loc[0, "nearest_training_pactivity"] == pytest.approx(6.0)
    assert result.loc[0, "already_measured"] == np.True_
    assert result.loc[1, "status"] != "predicted"
    assert pd.isna(result.loc[1, "predicted_pactivity"])
    assert result.loc[2, "MW"] > 0

    shortlist = diverse_shortlist(result, bundle, n=5, max_mw=900, max_tpsa=250)
    assert len(shortlist) == 2
    assert shortlist["canonical_smiles"].is_unique
    assert shortlist["rank"].tolist() == [1, 2]
    assert shortlist.loc[1, "max_shortlist_similarity"] <= 1


def test_shortlist_respects_properties_and_domain(bundle):
    result = predict_candidates(bundle, pd.DataFrame({"smiles": ["CCO", "CCN"]}))
    assert diverse_shortlist(result, bundle, max_mw=10).empty
    result.loc[0, "in_applicability_domain"] = False
    shortlist = diverse_shortlist(result, bundle, in_domain_only=True)
    assert shortlist["canonical_smiles"].tolist() == ["CCN"]


def test_upload_validation(bundle):
    with pytest.raises(ValueError, match="smiles"):
        predict_candidates(bundle, pd.DataFrame({"structure": ["CCO"]}))
    with pytest.raises(ValueError, match="500"):
        predict_candidates(bundle, pd.DataFrame({"smiles": ["CCO"] * 501}))

import json

from sarscope.params import ClassScheme, LandscapeParams, ModelParams, RunParams


def test_class_scheme_reproduces_the_paper():
    scheme = ClassScheme()
    assert scheme.labels == ("potent", "active", "intermediate", "inactive")
    assert dict(scheme.bounds) == {"potent": 8.0, "active": 7.0, "intermediate": 6.0}
    assert scheme.group1 == ("potent", "active")


def test_landscape_defaults_reproduce_the_paper():
    p = LandscapeParams()
    assert (p.similarity_threshold, p.activity_threshold, p.generator_sd) == (0.9, 2.0, 2.0)


def test_model_defaults_reproduce_the_paper_where_it_is_explicit():
    p = ModelParams()
    assert (p.test_fraction, p.cv_folds, p.seed) == (0.2, 10, 42)
    assert (p.features.variance_threshold, p.features.correlation_threshold) == (0.1, 0.95)


def test_model_defaults_depart_from_the_paper_where_it_leaks():
    p = ModelParams()
    assert p.split == "scaffold"
    assert p.leakage_audit is True


def test_run_params_serialise_to_json():
    data = json.loads(json.dumps(RunParams().to_dict()))
    assert set(data) == {"curation", "classes", "landscape", "model"}
    assert data["curation"]["relations"] == ["="]
    assert data["model"]["features"]["fingerprint"] == "ecfp4"

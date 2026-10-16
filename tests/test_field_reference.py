"""Offline checks for the independent comparator harness, not predictive validation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/field_qsar"
spec = importlib.util.spec_from_file_location("field_reference_probe", ROOT / "reference_probe.py")
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)
parity_spec = importlib.util.spec_from_file_location(
    "field_parity_probe", ROOT / "field_parity_probe.py"
)
parity = importlib.util.module_from_spec(parity_spec)
parity_spec.loader.exec_module(parity)
model_spec = importlib.util.spec_from_file_location(
    "model_parity_probe", ROOT / "model_parity_probe.py"
)
model_probe = importlib.util.module_from_spec(model_spec)
model_spec.loader.exec_module(model_probe)


def protocol():
    return json.loads((ROOT / "reference-selection.json").read_text())


def test_benchmark_q2_is_pooled_press_over_training_sst():
    assert model_probe.pooled_q2([1, 2, 3], [1, 2, 3]) == 1
    assert model_probe.pooled_q2([1, 2, 3], [2, 2, 2]) == 0
    assert model_probe.pooled_q2([1, 2, 3], [3, 2, 1]) == -3


def test_benchmark_q2_refuses_broadcasting_missing_predictions_and_constant_targets():
    for actual, predicted in (
        ([1, 2, 3], [[1], [2], [3]]),
        ([1, 2], [1, np.nan]),
        ([1, 1], [1, 1]),
    ):
        with pytest.raises(ValueError):
            model_probe.pooled_q2(actual, predicted)


def test_model_parity_does_not_hide_raw_failures_or_promote_posthoc_diagnostics():
    folder = ROOT / "model-parity-20261016-v3"
    if not (folder / "assessment.json").exists():
        pytest.skip("Local model comparison not distributed in this checkout.")
    result = json.loads((folder / "assessment.json").read_text())
    manifest = json.loads((folder / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assert len(result["comparisons"]) == 4
    for case in result["comparisons"]:
        expected = case["reference_variant"] == "reference_unit_converted"
        assert case["comparison_passes_declared_tolerances"] == expected
    assert not result["exact_historical_reproduction"]
    assert not result["independent_predictive_validation"]
    assert "added after" in result["posthoc_diagnostic_note"]
    assert (
        result["probe_sha256"]
        == hashlib.sha256((ROOT / "model_parity_probe.py").read_bytes()).hexdigest()
    )


def test_fixed_split_historical_reproduction_matches_paper_without_forcing_component_count():
    folder = ROOT / "fixed-split-api-20261016-v3"
    if not (folder / "assessment.json").exists():
        pytest.skip("Local historical API reproduction not distributed in this checkout.")
    result = json.loads((folder / "assessment.json").read_text())
    manifest = json.loads((folder / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    for source in ("reference", "sarscope"):
        row = result["results"][source]
        assert row["paper_q2_within_tolerance"]
        assert row["paper_train_r2_within_tolerance"]
        state = json.loads((folder / source / "api_state.json").read_text())
        assert len(state["q2_scores"]) == 12
        assert np.argmax(state["q2_scores"]) + 1 == state["optimal_components"] == 3
        test = pd.read_csv(folder / source / "PLS_Analysis/Test_Predictions_and_Residuals.csv")
        train = pd.read_csv(folder / source / "PLS_Analysis/Training_Predictions_and_Residuals.csv")
        assert (len(train), len(test)) == (21, 10)
        y, pred = test["Actual Values"].to_numpy(), test["Predicted Values"].to_numpy()
        expected_r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
        assert row["r2_test"] == pytest.approx(expected_r2, abs=1e-12)
    assert max(result["cross_source_prediction_max_absolute_differences"].values()) < 1e-8
    assert result["same_historical_api_not_independent_pls_implementation"]
    assert not result["production_preprocessing_changed"]
    assert not result["independent_predictive_validation"]
    assert (
        result["probe_sha256"]
        == hashlib.sha256((ROOT / "fixed_split_api_probe.py").read_bytes()).hexdigest()
    )


def test_field_parity_distinguishes_raw_match_from_posthoc_unit_conversion():
    values = np.linspace(-1, 2, 30).reshape(3, 10)
    actual = {name: values.copy() for name in parity.FIELDS}
    reference = {name: values * 7 for name in parity.FIELDS}
    result = parity.compare_arrays(reference, actual)
    for row in result.values():
        assert not row["matches_at_declared_tolerance"]
        assert row["spatial_shape_agrees_after_posthoc_scaling"]
        assert row["posthoc_reference_over_sarscope_factor"] == pytest.approx(7)
    assert all(
        row["matches_at_declared_tolerance"]
        for row in parity.compare_arrays(actual, actual).values()
    )


def test_field_parity_detects_spatial_disagreement_even_after_scaling():
    actual = {name: np.arange(1, 31, dtype=float).reshape(3, 10) for name in parity.FIELDS}
    reference = {name: value**2 for name, value in actual.items()}
    result = parity.compare_arrays(reference, actual)
    assert not any(row["spatial_shape_agrees_after_posthoc_scaling"] for row in result.values())
    reference["steric"] = np.ones((2, 10))
    with pytest.raises(ValueError, match="Unmatched field columns"):
        parity.compare_arrays(reference, actual)


def test_frozen_field_parity_remains_a_diagnostic_not_full_validation():
    folder = ROOT / "field-parity-20261016-v3"
    if not (folder / "assessment.json").exists():
        pytest.skip("Frozen local field-parity evidence not distributed in this checkout.")
    assessment = json.loads((folder / "assessment.json").read_text())
    manifest = json.loads((folder / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assert not assessment["full_method_reproduction"]
    assert not assessment["predictive_validation"]
    assert assessment["fields"]["charges"]["max_absolute_error"] == 0
    for name in ("steric", "electrostatic", "hydrophobic"):
        row = assessment["fields"][name]
        assert row["values_compared"] == 31 * 5152
        assert not row["matches_at_declared_tolerance"]
        assert row["spatial_shape_agrees_after_posthoc_scaling"]
    for name in ("hbond_donor", "hbond_acceptor"):
        assert not assessment["fields"][name]["spatial_shape_agrees_after_posthoc_scaling"]
    assert (
        assessment["probe_sha256"]
        == hashlib.sha256((ROOT / "field_parity_probe.py").read_bytes()).hexdigest()
    )


def test_external_reference_is_frozen_and_not_claimed_as_full_reproduction():
    settings = protocol()
    assert len(settings["commit"]) == 40
    assert settings["commit"] in settings["archive_url"]
    assert len(settings["archive_sha256"]) == 64
    assert settings["dataset"]["train"]["valid_records"] == 21
    assert settings["dataset"]["test"]["valid_records"] == 10
    assert settings["dataset"]["train"]["invalid_records"] == 1
    assert settings["cli_settings"]["cli_fields"] == "SEH"
    assert settings["paper_targets"]["q2"] == 0.609
    assert settings["paper_targets"]["train_r2"] == 0.917


def test_reference_hash_tampering_is_refused():
    expected = hashlib.sha256(b"frozen").hexdigest()
    probe.verify(b"frozen", expected)
    with pytest.raises(ValueError, match="hash changed"):
        probe.verify(b"changed", expected)


def test_cli_uses_executable_help_syntax_not_inaccurate_readme():
    args = probe.command(Path("/tmp/venv/bin/python"), Path("/tmp/ref"), protocol()["cli_settings"])
    assert args[args.index("--fields") + 1] == "SEH"
    assert "--disable_visualization" in args
    assert args[args.index("--num_components") + 1] == "3"
    assert args[args.index("--sdf_activity") + 1] == "Activity"


def test_graph_identity_hash_is_independent_of_explicit_hydrogen_representation():
    mol = Chem.MolFromSmiles("C[C@H](O)F")
    assert probe.graph_hash(mol) == probe.graph_hash(Chem.AddHs(mol))
    assert probe.graph_hash(mol) != probe.graph_hash(Chem.MolFromSmiles("C[C@@H](O)F"))


def mock_outputs(folder):
    metrics_folder = folder / "run/PLS_Analysis"
    metrics_folder.mkdir(parents=True)
    pd.DataFrame([{"q2": 0.607851, "r2_train": 0.896074, "r2_test": 0.964991}]).to_csv(
        metrics_folder / "PLS_Metrics.csv", index=False
    )
    pd.DataFrame({"Actual Values": np.arange(16)}).to_csv(
        metrics_folder / "Training_Predictions_and_Residuals.csv", index=False
    )
    pd.DataFrame({"Actual Values": np.arange(5)}).to_csv(
        metrics_folder / "Test_Predictions_and_Residuals.csv", index=False
    )
    records = [{"name": f"m{i}", "activity": float(i), "graph_sha256": str(i)} for i in range(10)]
    pd.DataFrame(
        {"Predicted Activity": np.arange(10) + 3, "graph_sha256": [str(i) for i in range(10)]}
    ).to_csv(folder / "predictions.csv", index=False)
    # Use real-looking hash strings so pandas does not infer integers.
    for row in records:
        row["graph_sha256"] = hashlib.sha256(row["name"].encode()).hexdigest()
    exported = pd.read_csv(folder / "predictions.csv")
    exported["graph_sha256"] = [row["graph_sha256"] for row in records]
    exported.to_csv(folder / "predictions.csv", index=False)
    return {"test": {"records": records}}


def test_external_scores_recalculated_and_never_confused_with_internal_test(tmp_path):
    audit = mock_outputs(tmp_path)
    result = probe.assess_outputs(tmp_path, audit, protocol())
    assert result["reported_training_count"] == 16
    assert result["reported_internal_test_count"] == 5
    assert result["supplied_external_count"] == 10
    assert result["reported_internal_test_r2"] > 0.9
    assert result["supplied_external_r2_recomputed"] < 0
    assert result["supplied_external_rmse_recomputed"] == 3
    assert result["q2_within_predeclared_tolerance"]
    assert not result["training_r2_within_predeclared_tolerance"]
    assert not result["sarscope_field_equivalence_established"]
    assert not result["comfa_reproduction_established"]


def test_reference_predictions_identity_mismatch_blocks_comparison(tmp_path):
    audit = mock_outputs(tmp_path)
    audit["test"]["records"].reverse()
    with pytest.raises(ValueError, match="identities/order"):
        probe.assess_outputs(tmp_path, audit, protocol())


def test_nonempty_reference_evidence_must_not_be_overwritten(tmp_path):
    (tmp_path / "prior.txt").write_text("preserve me")
    with pytest.raises(ValueError, match="new empty"):
        probe.run_probe(Path("/nonexistent/python"), tmp_path)
    assert (tmp_path / "prior.txt").read_text() == "preserve me"


def test_retained_reference_manifest_and_metrics_are_consistent():
    folder = ROOT / "reference-20261016-v4"
    if not folder.exists():
        pytest.skip("Frozen local reference run is not distributed in this checkout.")
    manifest = json.loads((folder / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assessment = json.loads((folder / "assessment.json").read_text())
    assert assessment["q2_within_predeclared_tolerance"]
    assert not assessment["sarscope_field_equivalence_established"]
    for path in folder.rglob("*.csv"):
        assert "SMILES" not in pd.read_csv(path).columns
    run = json.loads((folder / "run.json").read_text())
    assert (
        run["probe_sha256"]
        == hashlib.sha256((ROOT / "reference_probe.py").read_bytes()).hexdigest()
    )
    assert (
        run["selection_sha256"]
        == hashlib.sha256((folder / "reference-selection.json").read_bytes()).hexdigest()
    )

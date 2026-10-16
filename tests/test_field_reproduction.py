"""Independent own-code benchmark tests; synthetic runs are not efficacy evidence."""

import builtins
import hashlib
import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from threadpoolctl import threadpool_limits

from sarscope.analysis.align3d import align_series
from sarscope.analysis.fields import GridSpec
from sarscope.analysis.regression import _fit_field_pls
from sarscope.field_qsar import run_field_qsar
from sarscope.field_reproduction import historical_standardise, reproduce
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[1] / "benchmarks/field_qsar"


def test_historical_block_scaling_is_own_algebra_and_explicitly_transductive():
    raw = np.arange(30, dtype=float).reshape(5, 6) ** 2
    before = raw.copy()
    result, state = historical_standardise(raw, 3, 2)
    for field in range(2):
        section = slice(3 * field, 3 * (field + 1))
        block = raw[:, section]
        expected = block / block.std()
        expected[:3] -= expected[:3].mean(0)
        expected[3:] -= expected[3:].mean(0)
        np.testing.assert_allclose(result[:, section], expected)
        assert state["scale"][field] == block.std()
    np.testing.assert_array_equal(raw, before)
    changed = raw.copy()
    changed[3:] *= 3
    transformed, _ = historical_standardise(changed, 3, 2)
    assert not np.allclose(result[:3], transformed[:3])  # historical, NOT native validation
    for bad in (raw[:, :5], np.full((5, 6), np.nan)):
        with pytest.raises(ValueError):
            historical_standardise(bad, 3, 2)


def molecule():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def test_prealigned_input_hydrogen_coordinates_are_preserved_when_requested():
    mol = molecule()
    xyz = mol.GetConformer().GetPositions().copy()
    xyz[-1] += np.array([0.1, -0.1, 0.2])
    mol.GetConformer().SetAtomPosition(mol.GetNumAtoms() - 1, xyz[-1])
    params = FieldQsarParams(preserve_input_hydrogens=True, assess_prealigned_core=False)
    result = align_series([mol], ["reviewed"], params)
    assert result.molecules[0].GetNumAtoms() == mol.GetNumAtoms()
    np.testing.assert_array_equal(result.molecules[0].GetConformer().GetPositions(), xyz)
    assert np.isnan(result.common_core_coverage)
    assert result.report.warning.astype(bool).all()
    with pytest.raises(ValueError, match="pre-aligned"):
        replace(params, alignment="rigid_mcs")


def test_rounded_grid_bounds_have_documented_half_open_point_count():
    mol = molecule()
    xyz = mol.GetConformer().GetPositions()
    params = FieldQsarParams(spacing=1, margin=4, grid_convention="rounded_extent")
    grid = GridSpec.around([mol], params)
    rounded = np.rint(xyz)
    np.testing.assert_array_equal(grid.origin, rounded.min(0) - 4)
    np.testing.assert_array_equal(grid.shape, rounded.max(0) - rounded.min(0) + 8)
    default = GridSpec.around([mol], replace(params, grid_convention="enclosing"))
    assert default.n_points >= grid.n_points


def test_max_q2_search_does_not_stop_at_an_intermediate_plateau():
    rng = np.random.default_rng(73)
    X = rng.normal(size=(20, 12))
    y = X[:, 0] + X[:, 5] * 0.8 + rng.normal(scale=0.2, size=20)
    settings = FieldQsarParams(
        component_selection="max_q2",
        component_cap=3,
        keep_fraction=1,
        pls_max_iter=100,
        pls_tolerance=1e-4,
        # This exercises PLS component search on a synthetic matrix, not field
        # blocks; one declared field keeps per-field screening out of the way.
        field_subset=("steric",),
    )
    params = ModelParams(split="random", regression_cv="loo", field_qsar=settings)
    labels = np.full(20, "series")
    with threadpool_limits(limits=1):
        _, model = _fit_field_pls(X, y, params, labels, labels, None)
    assert model.selection_table_.components.tolist() == [1, 2, 3]
    best = model.selection_table_.loc[model.selection_table_.selection_q2.idxmax(), "components"]
    assert model.selected_components_ == best
    assert model.max_iter == 100 and model.tol == 1e-4


def synthetic_sources(test_offset=0):
    payloads, metadata = {}, {}
    original = molecule()
    for split, start, count in (("train", 0, 20), ("test", 20, 6)):
        stream = io.StringIO()
        writer = Chem.SDWriter(stream)
        for i in range(start, start + count):
            mol = Chem.Mol(original)
            mol.SetProp("name", f"synthetic-{i}")
            mol.SetDoubleProp("Activity", 5 + i / 20 + (test_offset if split == "test" else 0))
            xyz = mol.GetConformer().GetPositions() + [i * 0.04, np.sin(i) * 0.05, 0]
            for index, point in enumerate(xyz):
                mol.GetConformer().SetAtomPosition(index, point)
            writer.write(mol)
        writer.close()
        payloads[split] = stream.getvalue().encode()
        metadata[split] = {
            "sha256": hashlib.sha256(payloads[split]).hexdigest(),
            "valid": count,
            "invalid": 0,
        }
    protocol = {
        "data": {**metadata, "activity": "Activity"},
        "fields": {
            "method": "comsia",
            "field_subset": ["steric", "electrostatic", "hydrophobic"],
            "preserve_input_hydrogens": True,
            "assess_prealigned_core": False,
            "spacing": 2,
            "grid_convention": "rounded_extent",
            "keep_fraction": 1,
            "max_features": 60000,
            "component_cap": 2,
            "component_sample_ratio": None,
            "component_selection": "max_q2",
            "field_scaling": "raw",
            "permutations": 0,
        },
        "targets": {"q2": 0, "train_r2": 0, "components": 1},
        "absolute_tolerances": {"q2": 1, "train_r2": 1},
    }
    return protocol, payloads


def test_own_pipeline_runs_without_reference_software_and_does_not_fit_test_labels(
    tmp_path, monkeypatch
):
    original_import = builtins.__import__

    def no_reference(name, *args, **kwargs):
        if name.startswith(("src.PLS", "src.Molecular", "pycomsia")):
            raise AssertionError("Independent reproduction imported reference software")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_reference)
    protocol, payloads = synthetic_sources()
    first = reproduce(protocol, payloads, tmp_path / "first")
    altered_protocol, altered_payloads = synthetic_sources(test_offset=100)
    second = reproduce(altered_protocol, altered_payloads, tmp_path / "second")
    assert not first["external_reference_program_used"]
    assert first["q2"] == pytest.approx(second["q2"], abs=1e-12)
    a = pd.read_csv(tmp_path / "first/predictions.csv")
    b = pd.read_csv(tmp_path / "second/predictions.csv")
    np.testing.assert_allclose(a.prediction, b.prediction, atol=1e-12)
    assert second["historical_oracle_diagnostic"]["test_label_mean_shift"] == pytest.approx(
        first["historical_oracle_diagnostic"]["test_label_mean_shift"] + 100
    )
    assert not first["historical_oracle_diagnostic"]["is_model_prediction"]


def test_native_workflow_refuses_historical_uncapped_mode():
    from types import SimpleNamespace

    with pytest.raises(ValueError, match="benchmark-only"):
        run_field_qsar(
            SimpleNamespace(params=FieldQsarParams(component_sample_ratio=None)), ModelParams()
        )


def test_independent_frozen_result_is_own_training_reproduction_not_oracle_validation():
    folder = ROOT / "independent-20261016-v3"
    if not (folder / "assessment.json").exists():
        pytest.skip("Local independent reproduction artifacts are absent.")
    result = json.loads((folder / "assessment.json").read_text())
    for name, expected in json.loads((folder / "manifest.sha256.json").read_text()).items():
        assert hashlib.sha256((folder / name).read_bytes()).hexdigest() == expected
    assert not result["external_reference_program_used"]
    assert result["meets_declared_paper_tolerances"]
    assert result["selected_components"] == 3
    assert result["test_r2"] < 0
    assert not result["historical_oracle_diagnostic"]["is_model_prediction"]
    assert not result["independent_predictive_validation"]
    table = pd.read_csv(folder / "component_search.csv")
    assert table.components.tolist() == list(range(1, 13))
    from sarscope import field_reproduction

    assert (
        result["source_sha256"]["field_reproduction.py"]
        == hashlib.sha256(Path(field_reproduction.__file__).read_bytes()).hexdigest()
    )

"""Public parameter/manual component checks, NOT numerical SYBYL parity."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from test_field_numerics import methanol
from test_field_qsar import molecule
from test_field_qsar import synthetic_result as _synthetic_result

from sarscope.analysis.field_filter import FieldColumnFilter
from sarscope.analysis.fields import CHARGE_MODELS, GridSpec, compute_fields
from sarscope.analysis.tripos_reference import vdw_parameters
from sarscope.field_qsar import run_field_qsar
from sarscope.params import FieldQsarParams, ModelParams


def reference_params(method="comfa", **kwargs):
    return FieldQsarParams(
        method=method,
        field_definition="tripos52_se",
        field_subset=("steric", "electrostatic"),
        **kwargs,
    )


def test_literal_tripos_table_values_and_polar_h_class_are_not_rdkit_uff():
    radii, depth, classes = vdw_parameters(methanol())
    np.testing.assert_array_equal(radii, [1.70, 1.52, 1.50, 1.50, 1.50, 1.20])
    np.testing.assert_array_equal(depth, [0.107, 0.116, 0.042, 0.042, 0.042, 0.042])
    assert classes[2:] == ["H", "H", "H", "H.P"]
    for smiles in ("CS(=O)C", "CS(=O)(=O)C"):
        mol = molecule(smiles)
        radius, _, assigned = vdw_parameters(mol)
        sulfur = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 16)
        assert radius[sulfur] == 1.7
        assert assigned[sulfur] in ("S.o", "S.o2")
    with pytest.raises(ValueError, match="for B"):
        vdw_parameters(molecule("CB(O)O"))


@pytest.mark.parametrize("method", ["comfa", "comsia"])
def test_reference_se_kernels_match_independent_explicit_parameters(method, monkeypatch):
    mol = methanol()
    charges = np.array([0.2, -0.2, 0, 0, 0, 0])
    monkeypatch.setitem(CHARGE_MODELS, "reference_test", lambda _: charges)
    grid = GridSpec((-4.0, -4.0, -4.0), (4, 4, 4), spacing=2)
    settings = reference_params(method, charge_model="reference_test")
    actual = compute_fields([mol], grid, settings)
    xyz = mol.GetConformer().GetPositions()
    r2 = ((grid.points[:, None] - xyz[None]) ** 2).sum(2)
    radii = np.array([1.70, 1.52, 1.50, 1.50, 1.50, 1.20])
    if method == "comsia":
        kernel = np.exp(-0.3 * r2)
        expected = np.stack([kernel @ radii**3, kernel @ charges])
    else:
        distance = np.sqrt(np.maximum(r2, 0.3**2))
        depth = np.array([0.107, 0.116, 0.042, 0.042, 0.042, 0.042])
        ratio = (radii + 1.7) / distance
        steric = (np.sqrt(depth * 0.107) * (ratio**12 - 2 * ratio**6)).sum(1)
        electro = 332.0636 * (charges / distance).sum(1)
        electro[steric >= 30] = np.nan
        expected = np.stack([np.minimum(steric, 30), electro])
    np.testing.assert_allclose(
        actual.matrix.reshape(2, -1), expected, rtol=1e-11, atol=1e-11, equal_nan=True
    )
    assert "no vendor" in actual.convention
    assert "parameter_classes" in actual.charges[0]


def test_distance_dielectric_and_signed_electrostatic_plateau(monkeypatch):
    mol = methanol()
    charges = np.array([1.0, -1.0, 0, 0, 0, 0])
    monkeypatch.setitem(CHARGE_MODELS, "reference_test", lambda _: charges)
    grid = GridSpec((-4.0, -4.0, -4.0), (5, 5, 5), spacing=2)
    params = reference_params(
        charge_model="reference_test",
        dielectric_model="distance",
        electrostatic_cutoff=30,
        steric_cutoff=1e30,
    )
    field = compute_fields([mol], grid, params).matrix.reshape(2, -1)[1]
    r2 = ((grid.points[:, None] - mol.GetConformer().GetPositions()[None]) ** 2).sum(2)
    expected = np.clip(332.0636 * (charges / np.maximum(r2, 0.3**2)).sum(1), -30, 30)
    np.testing.assert_allclose(field, expected)
    assert field.min() == -30 and field.max() == 30


def test_missing_policy_uses_only_fitting_column_means_and_counts_replacements():
    X = np.array([[1, np.nan, np.nan, 4], [3, 2, np.nan, 4], [5, 8, np.nan, 4]])
    params = FieldQsarParams(method="comfa", missing_field_policy="training_mean", keep_fraction=1)
    filt = FieldColumnFilter(params).fit(X)
    np.testing.assert_array_equal(filt.support_, [True, True, False, False])
    np.testing.assert_allclose(filt.column_mean_, [3, 5, np.nan, 4], equal_nan=True)
    test = np.array(
        [[np.nan, 1000, np.nan, 4], [50, np.nan, np.nan, 4], [np.nan, np.nan, np.nan, 4]]
    )
    before = filt.column_mean_.copy()
    np.testing.assert_allclose(filt.transform(test), [[3, 1000], [50, 5], [3, 5]])
    np.testing.assert_array_equal(filt.missing_counts(test), [1, 1, 2])
    np.testing.assert_allclose(filt.column_mean_, before, equal_nan=True)
    assert filt.n_all_missing_ == 1
    assert np.isnan(X[0, 1]) and np.isnan(test[0, 0])
    with pytest.raises(ValueError, match="Infinite"):
        filt.transform(np.full((1, 4), np.inf))
    strict = FieldColumnFilter(replace(params, missing_field_policy="refuse")).fit(X)
    with pytest.raises(ValueError, match="not scorable"):
        strict.transform(test)


def test_reference_profile_does_not_silently_substitute_hydrophobic_or_hbond_fields():
    for fields in ((), ("hydrophobic",), ("hbond_donor",)):
        with pytest.raises(ValueError, match="not substituted"):
            FieldQsarParams(field_definition="tripos52_se", field_subset=fields)
    with pytest.raises(ValueError, match="CoMFA only"):
        reference_params("comsia", missing_field_policy="training_mean")
    with pytest.raises(ValueError, match="cutoff"):
        reference_params(electrostatic_cutoff=-30)


def test_native_pipeline_can_score_reference_comfa_with_reported_missingness():
    original = _synthetic_result.__wrapped__()
    params = reference_params(
        missing_field_policy="training_mean",
        electrostatic_cutoff=30,
        field_scaling="block_std",
        component_cap=1,
        max_features=100,
        permutations=0,
        spacing=3,
    )
    result = run_field_qsar(
        replace(original.prepared, params=params),
        ModelParams(split="random", cv_folds=3),
        split_indices=(
            original.regression.train_index.tolist(),
            original.regression.test_index.tolist(),
        ),
    )
    assert np.isfinite(result.predictions.prediction).all()
    assert "mean_filled_fraction" in result.predictions
    assert result.predictions.mean_filled_fraction.between(0, 1).all()
    assert "fitting-column means" in result.provenance["feature_mask_policy"]
    assert "parity not verified" in result.provenance["vendor_compatibility"]
    assert "not validated" in result.status


@pytest.mark.parametrize("method", ["comfa", "comsia"])
def test_public_se_zero_sign_chunking_and_input_immutability(method, monkeypatch):
    mol = methanol()
    before = mol.ToBinary()
    q = np.array([1.0, -1.0, 0, 0, 0, 0])
    monkeypatch.setitem(CHARGE_MODELS, "validation_control", lambda _: q)
    grid = GridSpec((3.0, 2.0, 4.0), (5, 4, 3), spacing=1)
    params = reference_params(method, charge_model="validation_control", chunk_points=7)
    a = compute_fields([mol], grid, params).matrix.reshape(2, -1)
    whole = compute_fields([mol], grid, replace(params, chunk_points=grid.n_points))
    np.testing.assert_allclose(a, whole.matrix.reshape(2, -1), atol=1e-12, rtol=0)
    monkeypatch.setitem(CHARGE_MODELS, "validation_control", lambda _: -q)
    b = compute_fields([mol], grid, params).matrix.reshape(2, -1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], -b[1], atol=1e-12, rtol=0)
    assert np.corrcoef(a[1], b[1])[0, 1] < -0.999
    monkeypatch.setitem(CHARGE_MODELS, "validation_control", lambda _: q * 0)
    zero = compute_fields([mol], grid, params).matrix.reshape(2, -1)
    np.testing.assert_array_equal(zero[1], np.zeros(grid.n_points))
    np.testing.assert_array_equal(a[0], zero[0])
    assert mol.ToBinary() == before


@pytest.mark.parametrize("method", ["comfa", "comsia"])
@pytest.mark.parametrize("charge", ["gasteiger", "mmff94"])
def test_public_se_rigid_covariance_and_atom_permutation(method, charge):
    mol = methanol()
    grid = GridSpec((-2.0, -2.0, -2.0), (5, 5, 5), spacing=1)
    params = reference_params(method, charge_model=charge)
    original = compute_fields([mol], grid, params).matrix.reshape(2, 5, 5, 5)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([4.0, 6.0, -2.0])
    moved = Chem.Mol(mol)
    for index, point in enumerate(mol.GetConformer().GetPositions()):
        moved.GetConformer().SetAtomPosition(index, rotation @ point + shift)
    changed = compute_fields(
        [moved], GridSpec(tuple(np.array(grid.origin) + shift), grid.shape, 1), params
    ).matrix.reshape(2, 5, 5, 5)
    np.testing.assert_allclose(
        changed,
        np.rot90(original, k=1, axes=(1, 2)),
        atol=1e-11,
        rtol=1e-11,
        equal_nan=True,
    )
    permuted = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    np.testing.assert_allclose(
        compute_fields([permuted], grid, params).matrix.reshape(2, 5, 5, 5),
        original,
        atol=1e-11,
        rtol=1e-11,
        equal_nan=True,
    )


def test_public_gaussian_spatial_decay_matches_closed_form_not_just_a_pretty_contour():
    mol = methanol()
    grid = GridSpec((3.0, 0.0, 0.0), (4, 2, 2), spacing=2)
    actual = compute_fields([mol], grid, reference_params("comsia")).matrix[0, : grid.n_points]
    # Literal public parameter values, not a call to the production atom selector.
    radii = np.array([1.70, 1.52, 1.50, 1.50, 1.50, 1.20])
    squared = ((grid.points[:, None] - mol.GetConformer().GetPositions()[None]) ** 2).sum(2)
    np.testing.assert_allclose(actual, np.exp(-0.3 * squared) @ radii**3, atol=1e-12)
    along_x = actual.reshape(grid.shape)[:, 0, 0]
    assert (np.diff(along_x) < 0).all()
    assert along_x[-1] / along_x[0] < 1e-4


@pytest.mark.parametrize("method", ["comfa", "comsia"])
def test_public_pipeline_fit_is_independent_of_heldout_labels_and_geometry(method):
    original = _synthetic_result.__wrapped__()
    params = reference_params(
        method,
        spacing=3,
        component_cap=1,
        max_features=100,
        field_scaling="block_std",
        permutations=0,
        missing_field_policy="training_mean" if method == "comfa" else "refuse",
    )
    prepared = replace(original.prepared, params=params)
    split = (original.regression.train_index.tolist(), original.regression.test_index.tolist())
    model = ModelParams(split="random", cv_folds=3)
    first = run_field_qsar(prepared, model, split_indices=split)
    altered_table = prepared.dataset.table.copy()
    altered_table.loc[split[1], "pactivity"] += 1000
    mols = [Chem.Mol(m) for m in prepared.alignment.molecules]
    for index in split[1]:
        for atom, point in enumerate(mols[index].GetConformer().GetPositions() + 1000):
            mols[index].GetConformer().SetAtomPosition(atom, point)
    altered = replace(
        prepared,
        dataset=replace(prepared.dataset, table=altered_table),
        alignment=replace(prepared.alignment, molecules=mols),
    )
    second = run_field_qsar(altered, model, split_indices=split)
    assert first.fields.grid == second.fields.grid
    a, b = first.regression, second.regression
    np.testing.assert_array_equal(a.fitted_filter.support_, b.fitted_filter.support_)
    np.testing.assert_allclose(
        a.fitted_filter.column_mean_, b.fitted_filter.column_mean_, equal_nan=True
    )
    np.testing.assert_array_equal(a.fitted_filter.column_scale_, b.fitted_filter.column_scale_)
    np.testing.assert_array_equal(a.fitted_model.coef_, b.fitted_model.coef_)
    np.testing.assert_array_equal(a.cv_predictions, b.cv_predictions)
    assert second.predictions.loc[split[1], "outside_training_grid"].all()
    assert "not validated" in second.status

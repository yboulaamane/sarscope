"""Independent numerical oracles and synthetic controls, NOT biological validation.

Scalar equations deliberately do not call production field/property helpers.
The small parameter snapshot tests SARscope's documented RDKit/UFF/Crippen
convention, not proprietary Tripos or directional hydrogen-bond equivalence.
"""

import math
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from threadpoolctl import threadpool_limits

from sarscope.analysis.field_filter import FieldColumnFilter
from sarscope.analysis.fields import CHARGE_MODELS, GridSpec, compute_fields
from sarscope.analysis.regression import _fit_field_pls, regression_cv_predictions
from sarscope.field_qsar import _pooled_q2
from sarscope.params import FieldQsarParams, ModelParams


def methanol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CO"))
    conformer = Chem.Conformer(mol.GetNumAtoms())
    conformer.Set3D(True)
    for index, point in enumerate(
        [
            (-0.6, 0, 0),
            (0.8, 0, 0),
            (-0.9, 0.8, 0.4),
            (-0.9, -0.8, 0.4),
            (-0.9, 0, -0.9),
            (1.3, 0.4, 0.5),
        ]
    ):
        conformer.SetAtomPosition(index, point)
    mol.AddConformer(conformer)
    return mol


@pytest.mark.parametrize("method", ["comfa", "comsia"])
def test_every_field_matches_independent_scalar_equations_and_parameter_snapshot(
    method, monkeypatch
):
    mol = methanol()
    q = np.array([0.2, -0.2, 0, 0, 0, 0])
    monkeypatch.setitem(CHARGE_MODELS, "numerical_oracle", lambda _: q)
    settings = FieldQsarParams(method=method, charge_model="numerical_oracle", chunk_points=7)
    grid = GridSpec((-2.0, -2.0, -2.0), (4, 4, 4), spacing=1.5)
    actual = compute_fields([mol], grid, settings).matrix[0].reshape(-1, grid.n_points)
    radii = [1.7, 1.55, 1.2, 1.2, 1.2, 1.2]
    depths = [0.105, 0.06, 0.044, 0.044, 0.044, 0.044]
    logp = [-0.2035, -0.2893, 0.123, 0.123, 0.123, -0.2677]
    xyz = mol.GetConformer().GetPositions()
    expected = []
    for point in grid.points:
        r2 = [
            sum((float(a) - float(b)) ** 2 for a, b in zip(point, atom, strict=True))
            for atom in xyz
        ]
        if method == "comsia":
            kernel = [math.exp(-0.3 * squared) for squared in r2]
            weights = [
                list(map(lambda r: r**3, radii)),
                q,
                logp,
                [0, 1, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
            ]
            expected.append(
                [math.fsum(a * b for a, b in zip(kernel, w, strict=True)) for w in weights]
            )
        else:
            distance = [math.sqrt(max(squared, 0.3**2)) for squared in r2]
            lj = math.fsum(
                math.sqrt(e * 0.107) * (((r + 1.7) / d) ** 12 - 2 * ((r + 1.7) / d) ** 6)
                for r, e, d in zip(radii, depths, distance, strict=True)
            )
            electro = 332.0636 * math.fsum(c / d for c, d in zip(q, distance, strict=True))
            expected.append([min(lj, 30.0), math.nan if lj >= 30 else electro])
    np.testing.assert_allclose(actual, np.array(expected).T, rtol=1e-11, atol=1e-11, equal_nan=True)


@pytest.mark.parametrize("method", ["comfa", "comsia"])
@pytest.mark.parametrize("charge_model", ["gasteiger", "mmff94"])
def test_fields_are_covariant_to_common_translation_and_right_angle_rotation(method, charge_model):
    mol = methanol()
    grid = GridSpec((-2.0, -2.0, -2.0), (5, 5, 5), spacing=1)
    settings = FieldQsarParams(method=method, charge_model=charge_model)
    original = compute_fields([mol], grid, settings).matrix[0].reshape(-1, 5, 5, 5)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    shift = np.array([4.0, 6.0, -2.0])
    transformed = Chem.Mol(mol)
    for index, point in enumerate(mol.GetConformer().GetPositions()):
        transformed.GetConformer().SetAtomPosition(index, rotation @ point + shift)
    moved = (
        compute_fields(
            [transformed],
            GridSpec(tuple(np.array(grid.origin) + shift), grid.shape, spacing=1),
            settings,
        )
        .matrix[0]
        .reshape(-1, 5, 5, 5)
    )
    expected = np.rot90(original, k=1, axes=(1, 2))
    # Swapping spatial axes is not a change in chemistry or field sign.
    np.testing.assert_allclose(moved, expected, rtol=1e-11, atol=1e-11, equal_nan=True)


def test_one_component_pls_matches_closed_form_and_raw_scale_contours():
    rng = np.random.default_rng(62)
    X = rng.normal(size=(40, 10)) * np.arange(1, 11)
    y = 6 + X[:, 0] * 0.2 - X[:, 3] * 0.3 + rng.normal(size=40) * 0.05
    settings = FieldQsarParams(
        field_subset=("steric",), keep_fraction=1, fixed_components=1, field_scaling="block_std"
    )
    with threadpool_limits(limits=1):
        filt, model = _fit_field_pls(X, y, ModelParams(field_qsar=settings), None, None, None)
    Z = filt.transform(X)
    centred, response = Z - Z.mean(0), y - y.mean()
    weight = centred.T @ response
    weight /= np.linalg.norm(weight)
    score = centred @ weight
    slope = float(score @ response / (score @ score))
    coefficients = weight * slope
    np.testing.assert_allclose(model.coef_.reshape(-1), coefficients, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(model.predict(Z).reshape(-1), y.mean() + score * slope, atol=1e-10)


def test_full_nested_selection_detects_synthetic_signal_and_rejects_permuted_controls():
    rng = np.random.default_rng(45)
    X = rng.normal(size=(48, 10))
    y = 6 + 2 * X[:, 0] - X[:, 1] + rng.normal(scale=0.05, size=48)
    settings = FieldQsarParams(
        field_subset=("steric",), keep_fraction=1, component_cap=2, component_selection="max_q2"
    )
    params = ModelParams(field_qsar=settings, split="random", cv_folds=3)
    labels = np.full(len(y), "synthetic")
    with threadpool_limits(limits=1):
        predictions, _, _ = regression_cv_predictions("field_pls", X, y, labels, labels, params)
        observed = _pooled_q2(y, predictions)
        null = []
        for _ in range(19):
            scrambled = rng.permutation(y)
            pred, _, _ = regression_cv_predictions(
                "field_pls", X, scrambled, labels, labels, params
            )
            null.append(_pooled_q2(scrambled, pred))
    assert observed > 0.9
    assert np.mean(null) < 0
    assert max(null) < observed


@pytest.mark.parametrize("scaling", ["raw", "block_std", "column_std"])
def test_training_scaler_and_mask_do_not_change_when_transforming_extreme_test_rows(scaling):
    rng = np.random.default_rng(64)
    X = rng.normal(size=(20, 12))
    settings = FieldQsarParams(field_scaling=scaling, field_subset=("steric", "electrostatic"))
    filt = FieldColumnFilter(settings).fit(X)
    before = [filt.support_.copy(), filt.training_sd_.copy(), filt.column_scale_.copy()]
    filt.transform(rng.normal(size=(6, 12)) * 1e6 + 1e8)
    for a, b in zip(before, [filt.support_, filt.training_sd_, filt.column_scale_], strict=True):
        np.testing.assert_array_equal(a, b)


def test_comfa_dielectric_changes_only_electrostatics():
    mol = methanol()
    grid = GridSpec((4.0, 4.0, 4.0), (2, 2, 2))
    settings = FieldQsarParams(method="comfa")
    a = compute_fields([mol], grid, settings).matrix.reshape(2, -1)
    b = compute_fields([mol], grid, replace(settings, dielectric=4)).matrix.reshape(2, -1)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_allclose(a[1], b[1] * 4, atol=1e-12)

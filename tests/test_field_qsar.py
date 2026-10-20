"""Synthetic specification checks, NOT a literature benchmark or efficacy evidence."""

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

from sarscope.analysis.align3d import align_series, best_mapping
from sarscope.analysis.contours import contribution_maps, open_dx
from sarscope.analysis.field_filter import FieldColumnFilter
from sarscope.analysis.fields import CHARGE_MODELS, GridSpec, _properties, compute_fields
from sarscope.analysis.regression import evaluate_regression
from sarscope.field_qsar import (
    BENCHMARK_STATUS,
    FieldDataset,
    assay_contexts,
    calibration_diagnostics,
    dataset_from_assay,
    dataset_from_upload,
    prepare_field_series,
    run_field_qsar,
    upload_table,
)
from sarscope.field_qsar_report import aligned_sdf, field_qsar_archive
from sarscope.params import FieldQsarParams, ModelParams


def molecule(smiles="CO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


@pytest.mark.parametrize("method", ["comfa", "comsia"])
def test_charge_dependence_zero_sign_and_independent_steric(method, monkeypatch):
    mol = molecule()
    params = FieldQsarParams(method=method, charge_model="test")
    grid = GridSpec((3.0, 3.0, 3.0), (3, 3, 3))
    charges = np.zeros(mol.GetNumAtoms())
    charges[0], charges[1] = 1, -1
    monkeypatch.setitem(CHARGE_MODELS, "test", lambda m: charges)
    fields = compute_fields([mol], grid, params).matrix[0].reshape(-1, grid.n_points)
    monkeypatch.setitem(CHARGE_MODELS, "test", lambda m: -charges)
    inverted = compute_fields([mol], grid, params).matrix[0].reshape(-1, grid.n_points)
    np.testing.assert_allclose(fields[0], inverted[0], atol=0, rtol=0)
    assert np.corrcoef(fields[1], inverted[1])[0, 1] < -0.999
    monkeypatch.setitem(CHARGE_MODELS, "test", lambda m: charges * 0)
    zero = compute_fields([mol], grid, params).matrix[0].reshape(-1, grid.n_points)
    assert (zero[1] == 0).all()
    if method == "comfa":
        distances = np.linalg.norm(
            grid.points[:, None, :] - mol.GetConformer().GetPositions(), axis=2
        )
        np.testing.assert_allclose(fields[1], 332.0636 * (charges / distances).sum(axis=1))


def test_undefined_comfa_electrostatics_remain_nan():
    mol = molecule()
    grid = GridSpec(tuple(mol.GetConformer().GetPositions()[0]), (2, 2, 2))
    result = compute_fields([mol], grid, FieldQsarParams(method="comfa"))
    steric, electro = result.matrix[0].reshape(2, -1)
    assert steric[0] == 30
    assert np.isnan(electro[0])
    np.testing.assert_array_equal(np.isnan(electro), steric >= 30)


@pytest.mark.parametrize("method", ["comfa", "comsia"])
def test_chunking_is_numerically_equivalent_and_inputs_untouched(method):
    mol = molecule("CCO")
    before = mol.ToBinary()
    grid = GridSpec((-4.0, -4.0, -4.0), (6, 6, 6))
    params = FieldQsarParams(method=method)
    chunked = compute_fields([mol], grid, replace(params, chunk_points=7))
    whole = compute_fields([mol], grid, replace(params, chunk_points=grid.n_points))
    np.testing.assert_allclose(chunked.matrix, whole.matrix, atol=1e-12, rtol=0)
    assert mol.ToBinary() == before


def test_gaussian_matches_analytic_shape_not_an_arbitrary_shell_decay():
    mol = molecule()
    grid = GridSpec((3.0, 0.0, 0.0), (4, 2, 2))
    result = compute_fields([mol], grid, FieldQsarParams())
    radius3 = np.array(
        [Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) ** 3 for a in mol.GetAtoms()]
    )
    r2 = ((grid.points[:, None, :] - mol.GetConformer().GetPositions()) ** 2).sum(axis=2)
    np.testing.assert_allclose(result.matrix[0, : grid.n_points], np.exp(-0.3 * r2) @ radius3)
    # Gaussian ratios depend on r and direction, not a universal 100x per 2 Å.
    values = result.matrix[0, : grid.n_points].reshape(grid.shape)[:, 0, 0]
    assert (np.diff(values) < 0).all()
    assert values[-1] / values[0] < 1e-4


def test_charge_fallback_is_opt_in_and_counted(monkeypatch):
    def unavailable(m):
        raise ValueError("no parameters")

    monkeypatch.setitem(CHARGE_MODELS, "mmff94", unavailable)
    mol = molecule()
    grid = GridSpec((3.0, 3.0, 3.0), (2, 2, 2))
    with pytest.raises(ValueError, match="Charge calculation failed"):
        compute_fields([mol], grid, FieldQsarParams(charge_model="mmff94"))
    result = compute_fields(
        [mol], grid, FieldQsarParams(charge_model="mmff94", allow_charge_fallback=True)
    )
    assert result.charges[0]["fallback"]
    assert result.charges[0]["used"] == "gasteiger"


def test_grid_and_memory_budgets_refuse_before_allocation():
    mol = molecule()
    with pytest.raises(ValueError, match="limit"):
        GridSpec.around([mol], FieldQsarParams(spacing=0.1, max_grid_points=100))
    with pytest.raises(ValueError, match="memory budget"):
        compute_fields(
            [mol] * 50, GridSpec((0.0, 0.0, 0.0), (20, 20, 20)), FieldQsarParams(max_memory_mb=1)
        )


@pytest.mark.parametrize(
    "smiles,acceptor",
    [
        ("CN", True),
        ("CN(C)C", True),
        ("CC(=O)N", False),
        ("CS(=O)(=O)N", False),
        ("C[NH3+]", False),
    ],
)
def test_ligand_acceptor_smart_rules(smiles, acceptor):
    mol = molecule(smiles)
    weights = _properties(mol, np.zeros(mol.GetNumAtoms()))
    nitrogen = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7)
    assert bool(weights[nitrogen, 4]) == acceptor


def test_prealigned_symmetry_uses_fixed_frame_not_a_hidden_rotation():
    mol = Chem.RemoveHs(molecule("c1ccccc1"))
    changed = Chem.RenumberAtoms(mol, [3, 4, 5, 0, 1, 2])
    result = align_series([mol, changed], ["a", "b"], FieldQsarParams())
    assert result.report.iloc[1].core_rmsd_A < 1e-6


@pytest.mark.parametrize("fixed_frame", [False, True])
def test_symmetry_budget_counts_distinct_maps_not_duplicate_query_automorphisms(fixed_frame):
    # Benzene has twelve query embeddings on each side: 144 pairs, but only
    # twelve distinct atom maps. All fit within a sixteen-map budget.
    reference = Chem.RemoveHs(molecule("c1ccccc1"))
    query = Chem.RenumberAtoms(reference, [3, 4, 5, 0, 1, 2])
    before = query.ToBinary()
    mapping, rmsd, _ = best_mapping(
        query, reference, FieldQsarParams(max_mappings=16), fixed_frame=fixed_frame
    )
    assert len(mapping) == 6 and rmsd < 1e-6
    assert query.ToBinary() == before


def test_truncated_symmetry_enumeration_still_refuses_instead_of_picking_first_map():
    reference = Chem.RemoveHs(molecule("c1ccccc1"))
    with pytest.raises(ValueError, match="enumeration exceeded its limit"):
        best_mapping(reference, reference, FieldQsarParams(max_mappings=8), fixed_frame=True)


def test_global_raw_screen_ranks_on_magnitude_across_blocks():
    """The documented global_raw behaviour: whichever block carries the larger
    numbers wins, which is only meaningful when the blocks share units."""
    X = np.array([[0, 1, np.nan, 0], [1, 1, 2, 10], [2, 1, 3, 20]], dtype=float)
    params = FieldQsarParams(keep_fraction=0.5, field_screen="global_raw")
    filt = FieldColumnFilter(params).fit(X)
    assert filt.support_.tolist() == [False, False, False, True]
    query = np.array([[0, 1, 2, np.nan]])
    with pytest.raises(ValueError, match="held-out"):
        filt.transform(query)
    assert filt.n_undefined_ == 1


def test_per_field_screen_is_invariant_to_rescaling_one_field():
    """The property global_raw lacks: multiplying a block by any positive
    constant must not change which columns are retained."""
    rng = np.random.default_rng(0)
    width = 40
    X = np.hstack([rng.normal(0, 3.0, (30, width)), rng.normal(0, 3.0, (30, width))])
    params = FieldQsarParams(
        method="comsia", field_subset=("steric", "electrostatic"), max_features=20
    )
    baseline = FieldColumnFilter(params).fit(X).support_
    for factor in (1e-3, 1e3):
        scaled = X.copy()
        scaled[:, width:] *= factor
        assert FieldColumnFilter(params).fit(scaled).support_.tolist() == baseline.tolist()


def test_per_field_screen_keeps_every_requested_field_populated():
    """A field given its own budget cannot be eliminated by its neighbour's units.

    Reproduces the 16 October 2026 qualification failure, where a Gaussian fit
    retained 2000 steric and 0 electrostatic columns.
    """
    rng = np.random.default_rng(1)
    width = 50
    steric = rng.normal(0, 12.0, (40, width))
    electrostatic = rng.normal(0, 0.08, (40, width))
    X = np.hstack([steric, electrostatic])
    params = FieldQsarParams(
        method="comsia", field_subset=("steric", "electrostatic"), max_features=40
    )
    filt = FieldColumnFilter(params).fit(X)
    kept = np.flatnonzero(filt.support_)
    assert int((kept < width).sum()) > 0
    assert int((kept >= width).sum()) > 0
    # keep_fraction 0.3 of 50 columns binds before the 40 // 2 budget.
    assert filt.retained_per_field_ == (15, 15)
    assert filt.empty_fields_ == ()

    global_raw = FieldColumnFilter(
        FieldQsarParams(
            method="comsia",
            field_subset=("steric", "electrostatic"),
            max_features=40,
            field_screen="global_raw",
        )
    ).fit(X)
    eliminated = np.flatnonzero(global_raw.support_)
    assert int((eliminated >= width).sum()) == 0


def test_prealigned_coordinates_preserved_and_translations_not_hidden():
    a, b = molecule("Fc1ccccc1"), molecule("Fc1ccccc1")
    for i, xyz in enumerate(b.GetConformer().GetPositions()):
        b.GetConformer().SetAtomPosition(i, xyz + np.array([2, 0, 0]))
    before = b.GetConformer().GetPositions().copy()
    result = align_series([a, b], ["a", "b"], FieldQsarParams())
    np.testing.assert_allclose(result.molecules[1].GetConformer().GetPositions(), before[:7])
    assert result.report.iloc[1].core_rmsd_A == pytest.approx(2)
    assert result.report.iloc[1].warning
    assert result.common_core_coverage <= 1


def test_symmetry_mapping_recovers_renumbered_ring_without_flip():
    mol = Chem.RemoveHs(molecule("Fc1ccc(Cl)cc1"))
    changed = Chem.RenumberAtoms(mol, list(reversed(range(mol.GetNumAtoms()))))
    mapping, rmsd, _ = best_mapping(changed, mol, FieldQsarParams())
    assert rmsd < 1e-6
    assert len(mapping) == mol.GetNumHeavyAtoms()


def test_generated_alignment_gate_and_unassigned_stereo(monkeypatch):
    monkeypatch.setattr("sarscope.analysis.align3d.preparation_available", lambda: False)
    with pytest.raises(ValueError, match="docking"):
        align_series([molecule()], ["a"], FieldQsarParams(alignment="rigid_mcs"))
    with pytest.raises(ValueError, match="stereochemistry"):
        align_series([molecule("CC(O)c1ccccc1")], ["a"], FieldQsarParams())


def test_assay_labels_recomputed_and_variant_context_not_pooled():
    table = pd.DataFrame(
        {"molecule_id": ["a", "b"], "smiles": ["CCO", "CCN"], "pactivity": [99, 99]}
    )
    evidence = pd.DataFrame(
        [
            {
                "molecule_id": "a",
                "assay_chembl_id": "A",
                "standard_type": "IC50",
                "pactivity": 6.0,
                "record_id": "1",
            },
            {
                "molecule_id": "a",
                "assay_chembl_id": "A",
                "standard_type": "IC50",
                "pactivity": 8.0,
                "record_id": "2",
            },
            {
                "molecule_id": "a",
                "assay_chembl_id": "B",
                "standard_type": "IC50",
                "pactivity": 10.0,
                "record_id": "3",
            },
            {
                "molecule_id": "b",
                "assay_chembl_id": "A",
                "standard_type": "IC50",
                "pactivity": 5.0,
                "record_id": "4",
                "assay_variant_mutation": "V600E",
            },
        ]
    )
    result = dataset_from_assay(table, evidence, "A", "IC50", "Unreported")
    assert result.table.molecule_id.tolist() == ["a"]
    assert result.table.pactivity.tolist() == [7]
    assert result.table.n_measurements.tolist() == [2]
    assert len(assay_contexts(evidence)) == 3


def test_upload_conversion_and_existing_curation_aggregate_replicates():
    raw = pd.DataFrame(
        {"id": ["a", "a", "b"], "smi": ["CCO", "CCO", "CCN"], "IC50": [100, 1000, 10]}
    )
    dataset = dataset_from_upload(
        raw,
        [],
        id_col="id",
        smiles_col="smi",
        value_col="IC50",
        unit="nM",
        endpoint="IC50",
        source_sha256="abc",
    )
    assert dataset.table.molecule_id.tolist() == ["a", "b"]
    assert dataset.table.pactivity.tolist() == [6.5, 8]
    assert len(dataset.accounting) >= 2
    assert dataset.source["input_unit"] == "nM"


@pytest.mark.parametrize("value", ["Active", np.inf, None])
def test_upload_refuses_qualitative_missing_or_infinite_labels(value):
    raw = pd.DataFrame({"id": ["a"], "smi": ["CCO"], "value": [value]})
    with pytest.raises((ValueError, TypeError)):
        dataset_from_upload(
            raw,
            [],
            id_col="id",
            smiles_col="smi",
            value_col="value",
            unit="p",
            endpoint="IC50",
            source_sha256="abc",
        )


def test_bounded_upload_and_malformed_sdf_not_silently_dropped():
    with pytest.raises(ValueError, match="5 MB"):
        upload_table(b"x" * (5 * 1024**2 + 1), "csv")
    with pytest.raises(ValueError, match="could not be parsed"):
        upload_table(b"broken\n\n\n $$$$\n", "sdf")


def test_nested_pls_never_fits_filter_to_heldout_rows(monkeypatch):
    rng = np.random.default_rng(12)
    X = rng.normal(size=(48, 20))
    X[:, 0] = np.arange(48)
    y = 6 + X[:, 1] + 0.4 * X[:, 2]
    fitted_rows = []
    original = FieldColumnFilter.fit

    def record(self, data, y=None):
        fitted_rows.append(set(data[:, 0]))
        return original(self, data, y)

    monkeypatch.setattr(FieldColumnFilter, "fit", record)
    params = ModelParams(
        split="random",
        cv_folds=3,
        regression_algorithms=("field_pls",),
        field_qsar=FieldQsarParams(keep_fraction=1.0, component_cap=3),
    )
    result = evaluate_regression(X, y, ["core"] * len(y), ["series"] * len(y), params)
    holdout = set(X[result.test_index, 0])
    assert all(not rows & holdout for rows in fitted_rows)
    assert np.isfinite(result.cv_predictions).all()
    assert result.fitted_model.selected_components_ <= len(result.train_index) // 5


def test_calibration_heuristics_fail_constant_or_wrong_predictions():
    truth = np.array([5.0, 6.0, 7.0, 8.0])
    assert calibration_diagnostics(truth, truth)["origin_calibration_heuristic"]
    assert not calibration_diagnostics(truth, truth * 0)["origin_calibration_heuristic"]
    assert not calibration_diagnostics(truth, truth[::-1])["origin_calibration_heuristic"]


@pytest.fixture(scope="module")
def synthetic_result():
    # Generated chemistry and artificial activities exercise wiring ONLY.
    fragments = [
        "F",
        "Cl",
        "Br",
        "C",
        "CC",
        "CCC",
        "O",
        "OC",
        "OCC",
        "N",
        "NC",
        "N(C)C",
        "C#N",
        "C(=O)N",
        "C(=O)O",
        "S",
        "SC",
        "C(F)(F)F",
        "C=O",
        "N=O",
    ]
    mols = [Chem.RemoveHs(molecule(f"c1ccccc1{fragment}")) for fragment in fragments]
    reference = mols[0]
    for mol in mols:
        mapping, _, _ = best_mapping(mol, reference, FieldQsarParams())
        rdMolAlign.AlignMol(mol, reference, atomMap=mapping)
    table = pd.DataFrame(
        {
            "molecule_id": [f"synthetic_{i}" for i in range(len(mols))],
            "smiles": [Chem.MolToSmiles(m) for m in mols],
            "pactivity": [
                5 + 0.07 * mol.GetNumHeavyAtoms() + i * 0.015 for i, mol in enumerate(mols)
            ],
            "n_measurements": 1,
            "pactivity_range": 0.0,
        }
    )
    measurements = table[["molecule_id", "smiles", "pactivity"]].copy()
    dataset = FieldDataset(
        table, mols, measurements, pd.DataFrame(), {"kind": "synthetic test only"}
    )
    prepared = prepare_field_series(
        dataset, FieldQsarParams(permutations=2, spacing=3.0, component_cap=1)
    )
    return run_field_qsar(prepared, ModelParams(split="random", cv_folds=3, test_fraction=0.25))


def test_end_to_end_same_split_baseline_and_experimental_guard(synthetic_result):
    result = synthetic_result
    assert np.array_equal(result.regression.test_index, result.baseline.test_index)
    assert np.array_equal(result.regression.train_index, result.baseline.train_index)
    assert len(result.permutations) == 2
    assert result.statistics.iloc[0].permutation_pvalue >= 1 / 3
    assert "not validated" in result.status
    assert result.provenance["benchmark_status"] == BENCHMARK_STATUS
    assert result.contours.summary.absolute_contribution_percent.sum() == pytest.approx(100)


def test_contours_are_raw_coefficient_times_training_sd_and_dx_order(synthetic_result):
    result = synthetic_result
    filt, model = result.regression.fitted_filter, result.regression.fitted_model
    expected = np.zeros(len(result.fields.labels))
    expected[filt.support_] = model.coef_.reshape(-1) * filt.training_sd_[filt.support_]
    np.testing.assert_allclose(np.concatenate(list(result.contours.maps.values())), expected)
    grid = GridSpec((1.0, 2.0, 3.0), (2, 2, 2), spacing=2.0)
    dx = open_dx(grid, np.arange(8))
    assert "origin 1 2 3" in dx
    assert "0 1 2\n3 4 5\n6 7" in dx
    np.testing.assert_array_equal(grid.points[1], [1, 2, 5])


def test_archive_records_failures_and_hashes(synthetic_result):
    with zipfile.ZipFile(io.BytesIO(field_qsar_archive(synthetic_result))) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.sha256.json"))
        for name, expected in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        assert b"not validated" in archive.read("report.html")
        assert b"Native end-to-end 3D-QSAR validation" in archive.read("validation_checks.csv")
        source_audit = pd.read_csv(io.BytesIO(archive.read("source_graph_audit.csv")))
        assert len(source_audit) == len(synthetic_result.predictions)
        assert source_audit.radical_electrons.eq(0).all()
        assert not source_audit.source_graph_review_required.any()
        assert "steric.dx" in archive.namelist()
        assert "contours.pml" in archive.namelist()
    read, mols = upload_table(aligned_sdf(synthetic_result.prepared).encode(), "sdf")
    assert len(mols) == len(synthetic_result.predictions)
    assert "pactivity" in read


def test_archive_can_replay_every_prediction_without_deserialising_a_model(synthetic_result):
    from benchmarks.validate_end_to_end_demo import numeric_replay

    replayed = numeric_replay(field_qsar_archive(synthetic_result))
    np.testing.assert_allclose(
        replayed, synthetic_result.predictions.prediction, atol=1e-10, rtol=0
    )


def test_contour_zero_information_has_no_isosurface(synthetic_result):
    result = synthetic_result
    model = SimpleNamespace(coef_=result.regression.fitted_model.coef_ * 0)
    contours = contribution_maps(
        result.fields, result.regression.fitted_filter, model, result.prepared.params
    )
    assert all(levels == (None, None) for levels in contours.levels.values())

"""Training-only sensitivity controls, not experimental binding-mode evidence."""

import hashlib
import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_field_qsar import molecule
from test_field_qsar import synthetic_result as _synthetic_result

from sarscope import field_sensitivity as sensitivity
from sarscope.analysis.fields import GridSpec
from sarscope.conformers import validate_conformer_stereo
from sarscope.field_qsar_report import field_qsar_archive
from sarscope.params import FieldQsarParams, ModelParams


@pytest.fixture(scope="module")
def field_result():
    return _synthetic_result.__wrapped__()


def test_phase_grid_covers_training_extent_plus_margin_and_refuses_bad_phase():
    mol = molecule("CCO")
    params = FieldQsarParams(grid_phase=(0.5, 0.25, 0.75))
    grid = GridSpec.around([mol], params)
    low = np.array(grid.origin)
    high = low + (np.array(grid.shape) - 1) * grid.spacing
    xyz = mol.GetConformer().GetPositions()
    assert (low <= xyz.min(0) - params.margin).all()
    assert (high >= xyz.max(0) + params.margin).all()
    np.testing.assert_allclose((low / params.spacing) % 1, params.grid_phase)
    for phase in [(0.0, 1.0, 0.0), (np.nan, 0.0, 0.0), (0.0, 0.0)]:
        with pytest.raises(ValueError, match="Grid phase"):
            replace(params, grid_phase=phase)
    with pytest.raises(ValueError, match="enclosing"):
        replace(params, grid_convention="rounded_extent")


def test_training_selector_preserves_fit_order_and_never_uses_heldout_labels_or_poses(field_result):
    source = field_result.prepared
    indices = field_result.regression.train_index.tolist()
    selected = sensitivity.training_dataset(source, indices)
    assert (
        selected.table.molecule_id.tolist()
        == source.dataset.table.iloc[indices].molecule_id.tolist()
    )
    altered_table = source.dataset.table.copy()
    test = field_result.regression.test_index
    altered_table.loc[test, "pactivity"] += 1000
    molecules = [type(m)(m) for m in source.alignment.molecules]
    for index in test:
        mol = molecules[int(index)]
        for atom, point in enumerate(mol.GetConformer().GetPositions() + 1000):
            mol.GetConformer().SetAtomPosition(atom, point)
    altered = replace(
        source,
        dataset=replace(source.dataset, table=altered_table),
        alignment=replace(source.alignment, molecules=molecules),
    )
    other = sensitivity.training_dataset(altered, indices)
    pd.testing.assert_frame_equal(selected.table, other.table)
    for a, b in zip(selected.molecules, other.molecules, strict=True):
        np.testing.assert_array_equal(
            a.GetConformer().GetPositions(), b.GetConformer().GetPositions()
        )
    assert set(selected.measurements.molecule_id) <= set(selected.table.molecule_id)
    assert selected.source["scope"] == "training_only"
    with pytest.raises(ValueError, match="unique"):
        sensitivity.training_dataset(source, [0] * 15)


def test_relative_rigid_perturbations_are_deterministic_preserve_bonds_and_stereo(field_result):
    from rdkit import Chem

    dataset = sensitivity.training_dataset(
        field_result.prepared, field_result.regression.train_index.tolist()
    )
    before = [m.ToBinary() for m in dataset.molecules]
    perturbed = sensitivity.perturb_poses(dataset, seed=1101, translation=0.25, rotation_degrees=5)
    repeat = sensitivity.perturb_poses(dataset, seed=1101, translation=0.25, rotation_degrees=5)
    for index, (a, b, c) in enumerate(
        zip(dataset.molecules, perturbed.molecules, repeat.molecules, strict=True)
    ):
        left, right = a.GetConformer().GetPositions(), b.GetConformer().GetPositions()
        np.testing.assert_allclose(
            np.linalg.norm(left[:, None] - left[None, :], axis=2),
            np.linalg.norm(right[:, None] - right[None, :], axis=2),
            atol=1e-12,
        )
        np.testing.assert_array_equal(right, c.GetConformer().GetPositions())
        validate_conformer_stereo(b, Chem.MolToSmiles(a))
        if index == 0:
            np.testing.assert_array_equal(left, right)
    assert [m.ToBinary() for m in dataset.molecules] == before


def test_real_training_diagnostic_and_archive_do_not_promote_flags_to_validation(field_result):
    dataset = sensitivity.training_dataset(
        field_result.prepared, field_result.regression.train_index.tolist()
    )
    variants = [sensitivity.sensitivity_variants()[i] for i in (0, 2, 9)]
    result = sensitivity.run_sensitivity(
        dataset,
        replace(field_result.prepared.params, max_features=100, component_cap=1),
        ModelParams(split="random", cv_folds=3),
        variants=variants,
    )
    assert result.summary.completed.all()
    assert result.summary.variant.tolist() == [v["name"] for v in variants]
    control = result.summary[result.summary.kind.eq("numerical_control")].iloc[0]
    assert control.oof_change_max < 1e-8
    assert result.provenance["held_out_rows_used"] == 0
    assert not result.provenance["held_out_labels_used"]
    assert not set(result.predictions.molecule_id) & set(
        field_result.predictions.loc[field_result.regression.test_index, "molecule_id"]
    )
    enriched = replace(
        field_result,
        training_stability={
            "summary": result.summary,
            "predictions": result.predictions,
            "contributions": result.contributions,
            "provenance": result.provenance,
        },
    )
    with zipfile.ZipFile(io.BytesIO(field_qsar_archive(enriched))) as archive:
        assert "training_stability/summary.csv" in archive.namelist()
        assert b"Training-only stability" in archive.read("report.html")
        assert b"not validated" in archive.read("report.html")
        manifest = json.loads(archive.read("manifest.sha256.json"))
        for name, digest in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest


def test_sensitivity_refuses_unselected_data_and_accounts_for_all_baseline_failures(
    field_result, monkeypatch
):
    with pytest.raises(ValueError, match="training-only"):
        sensitivity.run_sensitivity(
            field_result.prepared.dataset, field_result.prepared.params, ModelParams(split="random")
        )
    dataset = sensitivity.training_dataset(
        field_result.prepared, field_result.regression.train_index.tolist()
    )

    def fail(*args, **kwargs):
        raise ValueError("Deliberate numerical failure")

    monkeypatch.setattr(sensitivity, "training_fit", fail)
    result = sensitivity.run_sensitivity(
        dataset, field_result.prepared.params, ModelParams(split="random")
    )
    assert len(result.summary) == len(sensitivity.sensitivity_variants())
    assert not result.summary.completed.any()
    assert not result.summary.engineering_stability_flag.any()
    assert result.predictions.empty


def test_frozen_training_sensitivity_keeps_grid_passes_and_pose_failures():
    root = Path(__file__).resolve().parents[1] / "benchmarks/field_qsar/sensitivity-20261016-v2"
    if not (root / "manifest.sha256.json").exists():
        pytest.skip("Local sensitivity campaign not present.")
    for name, digest in json.loads((root / "manifest.sha256.json").read_text()).items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
    table = pd.read_csv(root / "summary.csv")
    assert len(table) == 60 and table.completed.all()
    candidate = table[table.profile.eq("candidate_block_1A")]
    assert candidate[candidate.kind.eq("grid_phase")].engineering_stability_flag.all()
    assert not candidate[candidate.kind.eq("relative_pose")].engineering_stability_flag.all()
    for dataset in ("AChE", "CCR5", "THR"):
        record = json.loads((root / dataset / "candidate_block_1A/provenance.json").read_text())
        assert record["source"]["scope"] == "training_only"
        assert not record["held_out_labels_used"]

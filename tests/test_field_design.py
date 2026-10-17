"""3D analogue scoring reuses the real fitted field model, never RF or a refit."""

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from test_field_qsar import synthetic_result as _synthetic_result

from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_design import (
    field_design_archive,
    field_design_inputs,
    field_model_signature,
    generate_field_analogues,
    predict_field_analogues,
)


@pytest.fixture(scope="module")
def fitted():
    return _synthetic_result.__wrapped__()


@pytest.fixture
def designed(fitted):
    table, _ = field_design_inputs(fitted)
    table = add_scaffolds(table)
    series = decompose_scaffold(table, table.murcko.iloc[0], min_substituent_count=1)
    eligible = [
        int(i)
        for i in fitted.regression.train_index
        if not fitted.prepared.alignment.report.iloc[int(i)].warning
    ]
    parent = table.iloc[eligible[0]].molecule_id
    design = generate_field_analogues(fitted, series, parent, series.positions, max_candidates=10)
    train_ids = set(table.iloc[fitted.regression.train_index].molecule_id)
    design.candidates = (
        design.candidates[design.candidates.known_ids.isin(train_ids)].head(3).copy()
    )
    assert not design.candidates.empty
    return design


def test_unvalidated_model_defaults_to_withholding_and_pre_aligned_route_is_explicit(
    fitted, designed
):
    with pytest.raises(ValueError, match="explicit candidate alignment"):
        predict_field_analogues(fitted, designed)
    prediction = predict_field_analogues(fitted, designed, route="rigid_mcs")
    assert prediction.design.candidates.predicted_pactivity.isna().all()
    assert prediction.design.candidates.prediction_status.str.contains("consent required").all()
    assert not prediction.aligned


def test_predictions_reproduce_displayed_model_without_fitting_or_2d_fallback(
    fitted, designed, monkeypatch
):
    def forbidden(*args, **kwargs):
        pytest.fail("No refit or RF fallback permitted")

    filt, estimator = fitted.regression.fitted_filter, fitted.regression.fitted_model
    monkeypatch.setattr(filt, "fit", forbidden)
    monkeypatch.setattr(estimator, "fit", forbidden)
    monkeypatch.setattr(fitted.baseline.fitted_model, "predict", forbidden)
    monkeypatch.setattr("sarscope.field_design._AlignmentGenerator", forbidden)
    before = field_model_signature(fitted)
    prediction = predict_field_analogues(
        fitted, designed, route="rigid_mcs", allow_exploratory=True
    )
    assert field_model_signature(fitted) == before
    assert prediction.audit["2d_fallback"] is False and prediction.audit["refit"] is False
    scored = prediction.design.candidates
    assert scored.predicted_pactivity.notna().all(), scored.prediction_status.tolist()
    table = fitted.prepared.dataset.table
    parent_idx = table.molecule_id.tolist().index(designed.parent.iloc[0].molecule_id)
    parent_point = fitted.predictions.iloc[parent_idx].prediction
    for row in scored.itertuples():
        index = table.molecule_id.tolist().index(row.known_ids)
        assert row.predicted_pactivity == pytest.approx(fitted.predictions.iloc[index].prediction)
        assert row.predicted_delta_vs_parent == pytest.approx(
            row.predicted_pactivity - parent_point
        )
        parts = prediction.contributions.query("molecule_id == @row.molecule_id")
        assert parts.delta_pactivity_contribution.sum() == pytest.approx(
            row.predicted_delta_vs_parent
        )
    with zipfile.ZipFile(io.BytesIO(field_design_archive(prediction, fitted))) as archive:
        assert "aligned_3d_candidates.sdf" in archive.namelist()
        audit = json.loads(archive.read("3d_scoring_audit.json"))
        assert audit["model_signature"] == before and audit["prediction_model"] == "3D field PLS"
        assert b"prediction_pair_conflict" in archive.read("candidates.csv")
        assert b"pair_evidence_warning" in archive.read("candidates.csv")
        assert "pair_conflict_policy" in audit


def test_changed_model_grid_or_reference_invalidates_old_design(fitted, designed):
    changed = replace(
        fitted,
        fields=replace(
            fitted.fields,
            grid=replace(fitted.fields.grid, spacing=fitted.fields.grid.spacing + 0.1),
        ),
    )
    with pytest.raises(ValueError, match="different 3D model"):
        predict_field_analogues(changed, designed, route="rigid_mcs", allow_exploratory=True)
    molecules = [Chem.Mol(m) for m in fitted.prepared.alignment.molecules]
    conformer = molecules[0].GetConformer()
    position = conformer.GetAtomPosition(0)
    conformer.SetAtomPosition(0, (position.x + 0.01, position.y, position.z))
    changed = replace(
        fitted,
        prepared=replace(
            fitted.prepared, alignment=replace(fitted.prepared.alignment, molecules=molecules)
        ),
    )
    assert field_model_signature(changed) != field_model_signature(fitted)


@pytest.mark.parametrize("failure", ["domain", "grid", "stereo", "undefined"])
def test_geometry_and_domain_failures_stay_withheld_even_in_exploratory_mode(
    fitted, designed, monkeypatch, failure
):
    from types import SimpleNamespace

    if failure == "domain":
        monkeypatch.setattr(
            fitted.domain,
            "classify",
            lambda *args: SimpleNamespace(in_domain=np.array([False])),
        )
    elif failure == "grid":
        monkeypatch.setattr(
            "sarscope.field_design.coordinates", lambda m: np.full((m.GetNumAtoms(), 3), 1e9)
        )
        # Generate a matching signature after the deliberate coordinate perturbation.
        designed.provenance["field_model_signature"] = field_model_signature(fitted)
    elif failure == "stereo":
        designed.candidates.loc[designed.candidates.index[0], "smiles"] = "CC(O)c1ccccc1"
    else:
        monkeypatch.setattr(
            fitted.regression.fitted_filter, "missing_counts", lambda X: np.full(len(X), 100000)
        )
    output = predict_field_analogues(fitted, designed, route="rigid_mcs", allow_exploratory=True)
    row = output.design.candidates.iloc[0]
    assert np.isnan(row.predicted_pactivity)
    assert np.isnan(row.predicted_delta_vs_parent)
    assert row.prediction_status.startswith("withheld:")


def test_measured_example_is_real_focused_and_has_honest_3d_verification():
    import pandas as pd

    from sarscope.field_example import field_example_dataset, field_example_files

    contents = field_example_files("egfr")
    csv = pd.read_csv(io.BytesIO(contents["egfr_3d.csv"]))
    dataset = field_example_dataset("egfr")
    assert len(csv) == len(dataset.table) == 50
    assert dataset.source["assay"] == "CHEMBL680017"
    assert len(add_scaffolds(dataset.table).murcko.unique()) == 1
    assert all(m.GetNumConformers() and m.GetConformer().Is3D() for m in dataset.molecules)
    evidence = dataset.measurements
    assert np.allclose(evidence.pactivity, 9 - np.log10(evidence.standard_value))
    report = json.loads(contents["verification.json"])
    three_d = next(row for row in report["statistics"] if row["algorithm"] == "field_pls")
    assert three_d["test_r2"] < 0 and "not validated" in report["status"]
    assert "CC BY-SA 3.0" in contents["README.md"].decode()


def test_novel_pose_is_generated_against_fitted_reference_and_uses_unchanged_fields(
    fitted, designed, monkeypatch
):
    import sys
    from pathlib import Path

    import sarscope.field_design as module

    # AppTest can leave __main__.__file__ pointing to its temporary unguarded
    # script. Spawn must import a safe real module, not rerun that Streamlit UI.
    monkeypatch.setattr(sys.modules["__main__"], "__file__", str(Path(__file__).resolve()))
    if not module.preparation_available():
        pytest.skip("Optional preparation extra is absent")
    designed.candidates = designed.candidates.iloc[:1].copy()
    index = designed.candidates.index[0]
    designed.candidates.loc[index, ["molecule_id", "smiles", "known_ids"]] = [
        "NEW_3D",
        "C#Cc1ccccc1",
        "",
    ]
    designed.candidates.loc[index, "observed_pactivity"] = np.nan
    real_compute = module.compute_fields
    captured = []

    def same_grid(molecules, grid, params):
        assert grid is fitted.fields.grid
        assert params is fitted.prepared.params
        computed = real_compute(molecules, grid, params)
        assert computed.labels == fitted.fields.labels
        captured.append(computed)
        return computed

    monkeypatch.setattr(module, "compute_fields", same_grid)
    before = field_model_signature(fitted)
    output = predict_field_analogues(fitted, designed, route="rigid_mcs", allow_exploratory=True)
    assert len(captured) == 1
    assert field_model_signature(fitted) == before
    assert output.design.candidates.iloc[0].pose_source.startswith("Generated rigid_mcs")
    row = output.design.candidates.iloc[0]
    if np.isfinite(row.predicted_pactivity):
        X = fitted.regression.fitted_filter.transform(captured[0].matrix)
        direct = fitted.regression.fitted_model.predict(X).reshape(-1)[0]
        assert row.predicted_pactivity == pytest.approx(direct)
    else:
        assert "PCA bounding box" in row.prediction_status

"""Regression coverage for the 17 October 3D-QSAR user-testing findings.

Artificial fixtures test implementation contracts, not scientific validation.
"""

import io
import json
import zipfile
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from test_field_qsar import synthetic_result as _synthetic_result

from sarscope.analysis.field_filter import FieldColumnFilter
from sarscope.field_design import flag_pair_conflicts
from sarscope.field_qsar import (
    _check_field_partitions,
    _pooled_q2,
    field_split_plan,
    run_field_qsar,
)
from sarscope.field_qsar_report import charge_audit, field_qsar_files
from sarscope.params import FieldQsarParams, ModelParams


@pytest.fixture(scope="module")
def fitted():
    return _synthetic_result.__wrapped__()


@pytest.mark.parametrize(
    "model,message",
    [
        (ModelParams(split="scaffold", cv_folds=3), "All compounds share one scaffold"),
        (ModelParams(split="random", test_fraction=0.35, cv_folds=3), "13 training and 7"),
    ],
)
def test_impossible_splits_stop_before_field_calculation(fitted, monkeypatch, model, message):
    monkeypatch.setattr(
        "sarscope.field_qsar.compute_fields", lambda *a: pytest.fail("No fields on invalid split")
    )
    with pytest.raises(ValueError, match=message):
        run_field_qsar(fitted.prepared, model)


def test_holdout_fold_count_explained_separately_from_cv(fitted):
    table = fitted.prepared.dataset.table.copy()
    table.loc[10:, "smiles"] = "c1ccncc1C"
    with pytest.raises(ValueError, match="4 scaffolds.*separate from the 3 CV folds"):
        field_split_plan(table, ModelParams(split="scaffold", test_fraction=0.25, cv_folds=3))


def test_nested_scaffold_feasibility_includes_inner_selection(fitted):
    table = fitted.prepared.dataset.table.copy()
    rings = ["c1ccccc1", "c1ccncc1", "c1ccoc1", "C1CCCCC1"]
    table["smiles"] = [rings[i // 5] for i in range(20)]
    # Four groups allow a 25% hold-out, but outer fitting folds then have only
    # two groups, not enough for three inner component-selection folds.
    with pytest.raises(ValueError, match="component-selection partition has 2 scaffolds"):
        field_split_plan(table, ModelParams(split="scaffold", cv_folds=3, test_fraction=0.25))


def test_comfa_overlap_checked_using_fitting_mask_before_any_pls():
    fields = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0], [3.0, np.nan]])
    params = FieldQsarParams(method="comfa", keep_fraction=1)
    with pytest.raises(ValueError, match="Early CoMFA.*Missing electrostatic descriptors"):
        _check_field_partitions(fields, [(np.array([0, 1, 2]), np.array([3]))], params)
    # Explicit opt-in remains the only way to use fitting means.
    _check_field_partitions(
        fields,
        [(np.array([0, 1, 2]), np.array([3]))],
        replace(params, missing_field_policy="training_mean"),
    )
    filt = FieldColumnFilter(params).fit(fields[:3])
    np.testing.assert_array_equal(filt.support_, [True, True])


def test_opposing_measured_effect_flags_model_gain_without_changing_prediction():
    frame = pd.DataFrame(
        {
            "molecule_id": ["gain", "loss", "unscored", "unsupported"],
            "predicted_pactivity": [9.06, 8.00, np.nan, 8.5],
            "predicted_delta_vs_parent": [0.60, -0.46, np.nan, 0.04],
        }
    )
    pairs = pd.DataFrame(
        {
            "candidate_id": ["gain", "gain", "loss", "unscored"],
            "id_from": ["a", "a", "c", "x"],
            "id_to": ["b", "b", "d", "y"],
            "delta_pactivity": [-3.66, -3.66, -0.5, 1.0],
        }
    )
    result = flag_pair_conflicts(frame, pairs)
    assert result.loc[0, "prediction_pair_conflict"]
    assert result.loc[0, "opposing_measured_pairs"] == 1  # duplicate pair not counted twice
    assert "median Δ -3.66" in result.loc[0, "pair_evidence_warning"]
    assert not result.loc[1, "prediction_pair_conflict"]
    assert pd.isna(result.loc[2, "prediction_pair_conflict"])
    assert pd.isna(result.loc[3, "prediction_pair_conflict"])
    np.testing.assert_allclose(
        result.predicted_pactivity, frame.predicted_pactivity, equal_nan=True
    )
    assert "prediction_pair_conflict" not in frame  # caller's proposals unchanged


def test_cv_dictionary_and_explicit_aliases_match_actual_calculations(fitted):
    scores = fitted.statistics
    np.testing.assert_allclose(scores.cv_fold_mean_r2, scores.cv_r2)
    np.testing.assert_allclose(scores.cv_fold_sd_r2, scores.cv_r2_sd)
    train = fitted.regression.train_index
    assert scores.iloc[0].cv_pooled_q2 == pytest.approx(
        _pooled_q2(
            fitted.prepared.dataset.table.pactivity.to_numpy()[train],
            fitted.regression.cv_predictions,
        )
    )
    files = field_qsar_files(fitted)
    assert b"Legacy alias" in files["metric_definitions.csv"]
    assert b"fold-specific predictive" in files["report.html"]


def test_check_values_have_unambiguous_meaning_and_charge_ids(fitted):
    checks = fitted.checks
    grid = checks[checks.check.str.contains("heavy atoms outside")].iloc[0]
    assert grid.passed == (grid.value == 0)
    assert "zero is a pass" in grid.criterion
    alignment = checks[checks.check.eq("Alignment thresholds met")].iloc[0]
    assert "50%" in alignment.criterion and "1 Å" in alignment.criterion
    origin = checks[checks.check.str.startswith("Origin calibration")].iloc[0]
    assert "Pearson r²" in origin.check and "k=" in origin.details
    audit = charge_audit(fitted)
    assert audit.molecule_id.tolist() == fitted.prepared.dataset.table.molecule_id.tolist()
    assert audit.reason.str.contains("no fallback").all()
    assert len(audit) == 20  # rows are compounds, not components


def test_pls_audit_explains_effective_cap_and_early_stop(fitted):
    model = fitted.regression.fitted_model
    assert model.effective_component_cap_ <= fitted.prepared.params.component_cap
    assert "Effective search cap" in model.selection_stop_reason_
    assert int(model.selection_table_.selected.sum()) == 1


def test_viewer_uses_camera_orientation_without_modifying_coordinates(fitted):
    pytest.importorskip("streamlit")
    from sarscope.field_qsar_ui import viewer_html

    mol = fitted.prepared.alignment.molecules[0]
    block = Chem.MolToMolBlock(mol)
    before = mol.ToBinary()
    page = viewer_html([block])
    payload = json.loads(page.split("const data = ", 1)[1].split(";\n", 1)[0])
    assert payload["blocks"] == [block]
    assert len(payload["orientation"]) == 4
    assert np.linalg.norm(payload["orientation"]) == pytest.approx(1)
    assert "600px" in page and "Reset view" in page
    assert mol.ToBinary() == before


def test_archive_conflict_flag_is_serializable():
    # Pandas nullable Boolean values must remain portable in CSV, including
    # withheld rows; no executable objects are needed in the evidence bundle.
    frame = pd.DataFrame({"molecule_id": ["a"], "predicted_delta_vs_parent": [np.nan]})
    flagged = flag_pair_conflicts(frame, pd.DataFrame())
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("candidates.csv", flagged.to_csv(index=False))
    with zipfile.ZipFile(io.BytesIO(stream.getvalue())) as archive:
        assert b"prediction_pair_conflict" in archive.read("candidates.csv")

"""Actual native default-run checks for the measured CCR5 tutorial, not certification."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

from sarscope.analysis.fields import COMSIA_FIELDS
from sarscope.field_example import field_example_archive, field_example_dataset, field_example_files
from sarscope.field_qsar import prepare_field_series, run_field_qsar
from sarscope.params import FieldQsarParams, ModelParams


@pytest.fixture(scope="module")
def measured_result():
    dataset = field_example_dataset()
    reference = dataset.table.molecule_id.tolist().index(dataset.source["reference_id"])
    # These are the actual fresh UI settings, not fitted coefficients, split
    # indices or a benchmark-only profile imported from verification.json.
    params = FieldQsarParams(reference_index=reference, field_subset=COMSIA_FIELDS)
    model = ModelParams(split="random", cv_folds=3, test_fraction=0.25, seed=42)
    with threadpool_limits(limits=1):
        return run_field_qsar(prepare_field_series(dataset, params), model)


def test_measured_positive_demo_is_offline_complete_and_audits_preparation(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx.Client, "send", lambda *a, **k: pytest.fail("Bundled data must be offline")
    )
    contents = field_example_files()
    data = field_example_dataset()
    csv = pd.read_csv(io.BytesIO(contents["ccr5_3d.csv"]), dtype={"molecule_id": str})
    assert len(csv) == len(data.table) == 75
    assert csv.molecule_id.is_unique and csv.smiles.is_unique
    assert data.source["independent_validation"] is False
    assert data.source["design_comparability_review_required"] is True
    assert data.source["license"] == "CC-BY-SA-4.0"
    assert all(not any(a.GetNumRadicalElectrons() for a in m.GetAtoms()) for m in data.molecules)
    assert all(m.GetNumConformers() and m.GetConformer().Is3D() for m in data.molecules)
    audit = pd.read_csv(io.BytesIO(contents["preparation.csv"]), dtype={"molecule_id": str})
    assert audit.heavy_atom_bonds_preserved.all() and audit.heavy_atom_coordinates_preserved.all()
    assert audit.coordinate_max_change_A.eq(0).all()
    assert audit.source_formal_charge.ne(audit.prepared_formal_charge).any()
    merged = csv.merge(audit, on="molecule_id", validate="one_to_one")
    np.testing.assert_allclose(merged.pactivity, merged.source_activity, atol=0, rtol=0)
    assert audit.original_partition.value_counts().to_dict() == {"train": 63, "test": 12}
    source = json.loads(contents["source.json"])
    builder = (
        Path(__file__).resolve().parents[1] / "examples/analogue_design/build_ccr5_3d_example.py"
    )
    assert hashlib.sha256(builder.read_bytes()).hexdigest() == source["builder_sha256"]
    for name, digest in source["files_sha256"].items():
        assert hashlib.sha256(contents[name]).hexdigest() == digest
    with zipfile.ZipFile(io.BytesIO(field_example_archive())) as archive:
        assert set(archive.namelist()) == set(contents)
        assert b"ShareAlike" in archive.read("LICENSE.txt")
        assert b"below 0.5" in archive.read("README.md")


def test_native_default_3d_model_has_positive_predictions_without_relaxing_checks(measured_result):
    result = measured_result
    stats = result.statistics.query("algorithm == 'field_pls'").iloc[0]
    assert stats.test_r2 > 0.5 and stats.test_rmse < 0.6
    assert 0.4 < stats.cv_pooled_q2 < 0.5  # Borderline failed check must remain failed.
    assert len(result.regression.train_index) == 56 and len(result.regression.test_index) == 19
    assert len(result.permutations) == 19
    assert not result.checks.loc[
        result.checks.check.str.startswith("Nested training CV"), "passed"
    ].iloc[0]
    assert result.checks.loc[result.checks.check.eq("Alignment thresholds met"), "passed"].iloc[0]
    assert "not validated" in result.status
    assert (
        result.baseline.scores.query("algorithm == 'random_forest'").iloc[0].test_r2 > stats.test_r2
    )
    assert result.contours.summary.retained_columns.gt(0).all()
    verified = json.loads(field_example_files()["verification.json"])
    saved = next(s for s in verified["statistics"] if s["algorithm"] == "field_pls")
    for metric in ("test_r2", "test_rmse", "cv_pooled_q2"):
        assert float(stats[metric]) == pytest.approx(saved[metric], abs=1e-6)
    assert verified["primary_seed_changed_after_repeats"] is False
    assert [r["seed"] for r in verified["repeat_splits"]] == [7, 19, 73, 101]
    assert len(verified["development_attempts"]) >= 4


def test_literature_design_cannot_invent_assay_comparability(measured_result):
    from sarscope.field_design import generate_field_analogues

    with pytest.raises(ValueError, match="not independently verified single-assay"):
        generate_field_analogues(measured_result, None, "CCR5_01", ["R1"])


def test_measured_demo_design_controls_reuse_the_displayed_3d_model(measured_result):
    from sarscope.analysis.rgroups import decompose_scaffold
    from sarscope.analysis.scaffolds import add_scaffolds
    from sarscope.field_design import (
        field_design_inputs,
        generate_field_analogues,
        predict_field_analogues,
    )

    table, _ = field_design_inputs(measured_result)
    table = add_scaffolds(table)
    scaffold = table.murcko.value_counts().index[0]
    series = decompose_scaffold(table, scaffold, min_substituent_count=1)
    train_ids = set(table.iloc[measured_result.regression.train_index].molecule_id)
    parent = next(pid for pid in series.members.molecule_id if pid in train_ids)
    designed = generate_field_analogues(
        measured_result,
        series,
        parent,
        series.positions,
        max_candidates=10,
        comparable_series_confirmed=True,
    )
    assert designed.provenance["benchmark_comparability_confirmed_by_user"] is True
    assert designed.provenance["known_compound_scope_label"] == "supplied series"
    # Known measured controls have independently inspectable saved poses and
    # points. This verifies scoring reuse, not novel-compound optimisation.
    designed.candidates = designed.candidates[designed.candidates.known_ids.isin(train_ids)].copy()
    assert not designed.candidates.empty
    scored = predict_field_analogues(
        measured_result, designed, route="rigid_mcs", allow_exploratory=True
    ).design.candidates
    assert scored.predicted_pactivity.notna().all(), scored.prediction_status.tolist()
    saved = measured_result.predictions.set_index("molecule_id").prediction
    for row in scored.itertuples():
        assert row.predicted_pactivity == pytest.approx(saved.loc[row.known_ids], abs=1e-9)


def test_literature_design_ui_requires_separate_comparability_declaration(measured_result):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    script = """
import streamlit as st
from sarscope.field_design_ui import show_field_design
show_field_design(st.session_state.fixture, 'measured_design')
"""
    app = AppTest.from_string(script, default_timeout=30)
    app.session_state["fixture"] = measured_result
    app.run()
    assert not app.exception
    assert any("single comparable assay" in w.value for w in app.warning)
    assert not any(b.label == "Prepare R-positions for 3D design" for b in app.button)
    app.checkbox(key="measured_design_field_design_comparability").check().run()
    assert not app.exception
    assert any(b.label == "Prepare R-positions for 3D design" for b in app.button)
    app.checkbox(key="measured_design_field_design_comparability").uncheck().run()
    assert not any(b.label == "Prepare R-positions for 3D design" for b in app.button)


def test_real_example_ui_passes_the_verified_native_defaults_and_displays_their_result(
    measured_result, monkeypatch
):
    from dataclasses import asdict

    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    import sarscope.field_qsar_ui as ui

    verification = json.loads(field_example_files()["verification.json"])
    calls = []

    def same_native_settings(prepared, model):
        assert json.loads(json.dumps(asdict(prepared.params))) == verification["settings"]
        assert json.loads(json.dumps(asdict(model))) == verification["model_settings"]
        assert prepared.dataset.table.molecule_id.tolist() == (
            measured_result.prepared.dataset.table.molecule_id.tolist()
        )
        calls.append(True)
        # The module fixture runs this exact native backend once; this adapter
        # checks UI wiring without repeating the nineteen-permutation fit.
        return measured_result

    monkeypatch.setattr(ui, "run_field_qsar_stage", same_native_settings)
    app = AppTest.from_string(
        "from sarscope.field_qsar_ui import show_field_qsar\nshow_field_qsar(key='positive3d')",
        default_timeout=60,
    ).run()
    app.checkbox(key="positive3d_example").check().run()
    assert not calls
    app.button(key="positive3d_prepare").click().run()
    assert not app.error and not app.exception
    app.checkbox(key="positive3d_reviewed").check().run()
    app.button(key="positive3d_run").click().run()
    assert not app.exception and not app.error and calls == [True]
    statistics = next(d.value for d in app.dataframe if "algorithm" in d.value.columns)
    assert statistics.set_index("algorithm").loc["field_pls", "test_r2"] > 0.5
    assert any("q² ≈ 0.499" in w.value and "remains failed" in w.value for w in app.warning)


def test_only_registered_examples_are_read_and_weak_case_remains_available():
    with pytest.raises(ValueError, match="registered"):
        field_example_files("../../private")
    assert len(field_example_dataset("egfr").table) == 50


def test_tampered_example_file_is_refused_before_modelling(tmp_path, monkeypatch):
    import sarscope.field_example as module

    contents = field_example_files()
    root = tmp_path / "data" / "field_ccr5_example"
    root.mkdir(parents=True)
    for name, data in contents.items():
        (root / name).write_bytes(data)
    (root / "ccr5_3d.csv").write_bytes(contents["ccr5_3d.csv"] + b"\n")
    monkeypatch.setattr(module, "files", lambda name: tmp_path)
    with pytest.raises(ValueError, match="checksum mismatch"):
        module.field_example_files()

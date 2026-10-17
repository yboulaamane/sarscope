"""Analogue hypotheses retain exact assay evidence and never invent potency gains."""

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

from sarscope.analogue_design import (
    DesignModel,
    design_archive,
    fit_design_model,
    generate_analogues,
    model_checks,
    scope_signature,
    score_analogues,
)
from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.params import AnalogueDesignParams, ModelParams


@pytest.fixture
def data():
    table = add_scaffolds(
        pd.DataFrame(
            [
                ("M1", "Cc1cc(F)ncc1", 6.0),
                ("M2", "Cc1cc(Cl)ncc1", 7.0),
                ("M3", "CCc1cc(F)ncc1", 6.5),
            ],
            columns=["molecule_id", "smiles", "pactivity"],
        ).assign(activity_class="series")
    )
    evidence = table[["molecule_id", "pactivity"]].assign(
        record_id=["1", "2", "3"],
        assay_chembl_id="A1",
        standard_type="IC50",
        standard_relation="=",
        assay_variant_mutation=None,
    )
    series = decompose_scaffold(table, table.murcko.iloc[0], min_substituent_count=1)
    assert series is not None
    return series, table, evidence


def design(data, **settings):
    series, table, evidence = data
    return generate_analogues(
        series, table, evidence, "M3", series.positions, AnalogueDesignParams(**settings)
    )


def strong_model(result):
    return DesignModel(
        None,
        SimpleNamespace(scores=pd.DataFrame()),
        result.provenance["scope_signature"],
        pd.DataFrame({"check": ["mock only"], "value": [1.0], "passed": [True]}),
        {},
    )


def mock_predict(bundle, frame):
    return frame.assign(
        status="predicted",
        canonical_smiles=frame.smiles.map(
            lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=True)
        ),
        in_applicability_domain=True,
        predicted_pactivity=np.arange(len(frame), dtype=float) + 7.0,
        empirical_lower_pactivity=np.arange(len(frame), dtype=float) + 6.5,
        empirical_upper_pactivity=np.arange(len(frame), dtype=float) + 7.5,
    )


def test_missing_combination_is_a_hypothesis_not_a_measured_or_predicted_potency(data):
    result = design(data)
    novel = result.candidates[result.candidates.known_ids.eq("")].iloc[0]
    assert novel.smiles == Chem.MolToSmiles(Chem.MolFromSmiles("CCc1cc(Cl)ncc1"))
    assert novel.observed_median_delta == 1.0
    assert novel.supporting_pairs == novel.disjoint_pairs == 1
    assert pd.isna(novel.observed_pactivity)
    assert "predicted_pactivity" not in result.candidates
    support = result.evidence_pairs[result.evidence_pairs.candidate_id.eq(novel.molecule_id)]
    assert set(support.id_from) | set(support.id_to) == {"M1", "M2"}
    assert set(support.evidence_tier) == {"shared assay"}
    assert len(result.measurements) == 3
    known = result.candidates[result.candidates.known_ids.ne("")].iloc[0]
    assert known.known_ids == "M1" and known.observed_pactivity == 6.0
    assert known.observed_median_delta == -0.5  # reversed measured swap


def test_one_position_only_and_locked_positions_unchanged(data):
    series, table, evidence = data
    parent = series.members.query("molecule_id == 'M3'").iloc[0]
    fluorine_position = next(pos for pos in series.positions if parent[pos].startswith("F"))
    result = generate_analogues(
        series, table, evidence, "M3", [fluorine_position], AnalogueDesignParams()
    )
    assert len(result.candidates) == 1
    assert set(result.candidates.position) == {fluorine_position}
    assert "CC" in result.candidates.iloc[0].smiles  # ethyl remains locked
    assert result.provenance["one_position_per_candidate"] is True
    assert result.parent.iloc[0].observed_pactivity == 6.5


def test_caps_and_fragment_rejection_are_accounted_for(data, monkeypatch):
    result = design(data, max_candidates=1)
    assert len(result.candidates) == 1 and result.provenance["truncated"]
    result = design(data, max_attempts=1)
    assert result.provenance["attempted"] == 1
    result = design(data, max_fragment_heavy_atoms=1)
    assert "outside single-attachment fragment budget" in set(result.accounting.status)
    monkeypatch.setattr("sarscope.analogue_design.assemble_analogue", lambda *args: "")
    result = design(data)
    assert result.candidates.empty
    assert all("round-trip failed" in status for status in result.accounting.status)


@pytest.mark.parametrize(
    "column,value",
    [
        ("standard_type", "Ki"),
        ("assay_chembl_id", "A2"),
        ("assay_variant_mutation", "V600E"),
        ("standard_relation", ">"),
        ("assay_chembl_id", None),
        ("record_id", None),
    ],
)
def test_mixed_incomplete_and_censored_scopes_are_refused(data, column, value):
    series, table, evidence = data
    evidence = evidence.copy()
    evidence.loc[0, column] = value
    with pytest.raises(ValueError):
        generate_analogues(series, table, evidence, "M3", series.positions, AnalogueDesignParams())


def test_pooled_activities_and_stale_parent_structures_are_refused(data):
    series, table, evidence = data
    table = table.copy()
    table.loc[0, "pactivity"] = 10.0
    with pytest.raises(ValueError, match="Reaggregate"):
        generate_analogues(series, table, evidence, "M3", series.positions, AnalogueDesignParams())
    table.loc[0, "pactivity"] = 6.0
    table.loc[2, "smiles"] = "CCO"
    with pytest.raises(ValueError, match="scoped assay records"):
        generate_analogues(series, table, evidence, "M3", series.positions, AnalogueDesignParams())


def test_hash_binds_structures_measurements_and_context_and_is_order_independent(data):
    _, table, evidence = data
    original = scope_signature(table, evidence)
    assert original == scope_signature(table.iloc[::-1], evidence.iloc[::-1])
    for changed in (
        evidence.assign(assay_chembl_id="A2"),
        evidence.assign(pactivity=evidence.pactivity + 1),
        evidence.assign(assay_variant_mutation="V600E"),
    ):
        assert scope_signature(table, changed) != original
    assert scope_signature(table.assign(smiles="CCO"), evidence) != original


def test_scoring_delta_uses_model_parent_not_observed_parent(data, monkeypatch):
    result = design(data)
    monkeypatch.setattr("sarscope.analogue_design.predict_candidates", mock_predict)
    scored = score_analogues(result, strong_model(result))
    assert scored.candidates.predicted_delta_vs_parent.tolist() == [1.0, 2.0]
    assert scored.candidates.model_parent_prediction.tolist() == [7.0, 7.0]
    assert scored.parent.iloc[0].observed_pactivity == 6.5
    assert "predicted_pactivity" not in result.candidates  # no mutation


@pytest.mark.parametrize(
    "failure",
    [
        "candidate_domain",
        "candidate_state",
        "parent_domain",
        "parent_state",
        "candidate_stereo",
        "parent_stereo",
        "nonfinite",
    ],
)
def test_unsafe_predictions_are_withheld(data, monkeypatch, failure):
    result = design(data)
    if failure == "candidate_stereo":
        result.candidates.loc[0, "unresolved_stereo"] = True
    if failure == "parent_stereo":
        result.parent.loc[0, "unresolved_stereo"] = True

    def altered(bundle, frame):
        rows = mock_predict(bundle, frame)
        if failure in ("candidate_domain", "parent_domain"):
            rows.loc[1 if failure.startswith("candidate") else 0, "in_applicability_domain"] = False
        if failure in ("candidate_state", "parent_state"):
            rows.loc[1 if failure.startswith("candidate") else 0, "canonical_smiles"] = "CCO"
        if failure == "nonfinite":
            rows.loc[1, "predicted_pactivity"] = np.nan
        return rows

    monkeypatch.setattr("sarscope.analogue_design.predict_candidates", altered)
    scored = score_analogues(result, strong_model(result))
    assert pd.isna(scored.candidates.iloc[0].predicted_pactivity)
    assert pd.isna(scored.candidates.iloc[0].predicted_delta_vs_parent)
    assert scored.candidates.iloc[0].prediction_status.startswith("withheld:")


def test_failed_qualification_clears_old_predictions_and_never_calls_predict(data, monkeypatch):
    result = design(data)
    result.candidates["predicted_pactivity"] = 99.0
    result.candidates["predicted_delta_vs_parent"] = 88.0
    model = strong_model(result)
    model.checks["passed"] = False
    monkeypatch.setattr(
        "sarscope.analogue_design.predict_candidates",
        lambda *args: pytest.fail("Weak models must not predict"),
    )
    scored = score_analogues(result, model)
    assert scored.candidates.predicted_pactivity.isna().all()
    assert scored.candidates.predicted_delta_vs_parent.isna().all()
    with pytest.raises(ValueError, match="scopes differ"):
        score_analogues(result, replace(model, scope_signature="other"))


def test_model_gates_refuse_poor_small_or_uncalibrated_evaluations():
    result = SimpleNamespace(
        best_algorithm="ridge",
        test_truth=np.arange(10.0),
        baseline_test_predictions=np.full(10, 4.5),
        empirical_half_width=0.5,
        scores=pd.DataFrame(
            [{"algorithm": "ridge", "test_r2": 0.7, "test_rmse": 0.4, "design_pooled_cv_q2": 0.6}]
        ),
    )
    assert model_checks(result, AnalogueDesignParams()).passed.all()
    result.scores.loc[0, "test_r2"] = 0.04
    assert not model_checks(result, AnalogueDesignParams()).passed.all()
    result.scores.loc[0, "test_r2"] = 0.7
    result.empirical_half_width = np.nan
    assert not model_checks(result, AnalogueDesignParams()).passed.all()
    result.empirical_half_width = 0.5
    result.test_truth = np.arange(3.0)
    result.baseline_test_predictions = np.ones(3)
    assert not model_checks(result, AnalogueDesignParams()).passed.all()


def test_small_series_cannot_fit_a_predictive_design_model(data):
    _, table, evidence = data
    with pytest.raises(ValueError, match="30–500"):
        fit_design_model(table, evidence, AnalogueDesignParams(), ModelParams())


def test_archive_keeps_evidence_and_valid_structures_without_invented_activity(data):
    result = design(data)
    with zipfile.ZipFile(io.BytesIO(design_archive(result))) as archive:
        manifest = json.loads(archive.read("manifest.sha256.json"))
        for path, checksum in manifest.items():
            assert hashlib.sha256(archive.read(path)).hexdigest() == checksum
        assert len(pd.read_csv(io.BytesIO(archive.read("retained_measurements.csv")))) == 3
        candidates = pd.read_csv(io.BytesIO(archive.read("candidates.csv")))
        assert candidates.prediction_status.str.startswith("not scored").all()
        sdf = list(Chem.ForwardSDMolSupplier(io.BytesIO(archive.read("analogues.sdf"))))
        assert len(sdf) == len(result.candidates) and all(m is not None for m in sdf)
        novel = next(m for m in sdf if not m.GetProp("known_ids"))
        assert not novel.HasProp("observed_pactivity")
        assert not novel.HasProp("predicted_pactivity")
        assert "NOT candidate potency" in archive.read("README.txt").decode()


@pytest.mark.parametrize(
    "settings",
    [
        {"max_candidates": 101},
        {"max_attempts": 2001},
        {"max_scope_compounds": 501},
        {"max_fragment_heavy_atoms": 0},
        {"cliff_similarity": 0},
        {"cliff_activity": np.nan},
        {"minimum_test_r2": 1.0},
    ],
)
def test_invalid_budgets_are_rejected(settings):
    with pytest.raises(ValueError):
        AnalogueDesignParams(**settings)


def test_real_assigned_stereo_survives_a_different_position_swap(data):
    _, table, evidence = data
    table.loc[2, "smiles"] = "C[C@H](O)c1cc(F)ncc1"
    table = add_scaffolds(table.drop(columns="murcko"))
    series = decompose_scaffold(table, table.murcko.iloc[0], min_substituent_count=1)
    parent = series.members.query("molecule_id == 'M3'").iloc[0]
    position = next(pos for pos in series.positions if parent[pos].startswith("F"))
    result = generate_analogues(series, table, evidence, "M3", [position], AnalogueDesignParams())
    expected = Chem.MolToSmiles(Chem.MolFromSmiles("C[C@H](O)c1cc(Cl)ncc1"))
    assert result.candidates.smiles.tolist() == [expected]
    assert not result.candidates.unresolved_stereo.any()


def test_scoped_2d_fit_reuses_real_regression_engine():
    table = pd.DataFrame(
        [
            {
                "molecule_id": f"M{i}_{j}",
                "smiles": f"{'C' * i}c1cc({sub})ncc1",
                "pactivity": 5.0 + i * 0.1 + j * 0.2,
                "activity_class": "series",
            }
            for i in range(1, 9)
            for j, sub in enumerate(["F", "Cl", "Br", "OC", "N"])
        ]
    )
    evidence = table[["molecule_id", "pactivity"]].assign(
        record_id=table.molecule_id,
        assay_chembl_id="A1",
        standard_type="IC50",
        standard_relation="=",
        assay_variant_mutation=None,
    )
    model = fit_design_model(
        table,
        evidence,
        AnalogueDesignParams(),
        ModelParams(
            regression_algorithms=("ridge",),
            split="random",
            cv_folds=3,
            test_fraction=0.25,
            leakage_audit=False,
        ),
    )
    assert model.scope_signature == scope_signature(table, evidence)
    assert len(model.regression.test_truth) == 10
    assert len(model.bundle.train_ids) == 40
    assert model.bundle.features.fingerprint == "ecfp4"
    assert model.regression.cv_predictions is not None
    assert "design_pooled_cv_q2" in model.regression.scores
    assert len(model.checks) == 6


def test_repeated_records_do_not_multiply_support_and_conflicting_records_are_refused(data):
    series, table, evidence = data
    repeated = pd.concat([evidence, evidence], ignore_index=True)
    result = generate_analogues(
        series, table, repeated, "M3", series.positions, AnalogueDesignParams()
    )
    assert result.candidates.supporting_pairs.tolist() == [1, 1]
    repeated.loc[len(evidence), "pactivity"] = 100.0
    with pytest.raises(ValueError, match="conflicting"):
        generate_analogues(series, table, repeated, "M3", series.positions, AnalogueDesignParams())


def test_supporting_structure_and_scope_membership_must_match(data):
    series, table, evidence = data
    changed = table.copy()
    changed.loc[0, "smiles"] = "CCO"
    with pytest.raises(ValueError, match="structures differ"):
        generate_analogues(
            series, changed, evidence, "M3", series.positions, AnalogueDesignParams()
        )
    with pytest.raises(ValueError, match="same compounds"):
        generate_analogues(
            series, table.iloc[:2], evidence, "M1", series.positions, AnalogueDesignParams()
        )
    changed.loc[0, "smiles"] = changed.loc[1, "smiles"]
    with pytest.raises(ValueError, match="duplicate isomeric"):
        generate_analogues(
            series, changed, evidence, "M3", series.positions, AnalogueDesignParams()
        )


def test_opposite_measured_effects_are_visible_not_optimistically_filtered(data):
    _, table, evidence = data
    table = pd.concat(
        [
            table,
            pd.DataFrame(
                [
                    {
                        "molecule_id": "M4",
                        "smiles": "CCCc1cc(F)ncc1",
                        "pactivity": 7.0,
                        "activity_class": "series",
                    },
                    {
                        "molecule_id": "M5",
                        "smiles": "CCCc1cc(Cl)ncc1",
                        "pactivity": 6.0,
                        "activity_class": "series",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    table = add_scaffolds(table.drop(columns="murcko"))
    evidence = table[["molecule_id", "pactivity"]].assign(
        record_id=table.molecule_id,
        assay_chembl_id="A1",
        standard_type="IC50",
        standard_relation="=",
        assay_variant_mutation=None,
    )
    series = decompose_scaffold(table, table.murcko.iloc[0], min_substituent_count=1)
    result = design((series, table, evidence), cliff_similarity=0.2, cliff_activity=0.1)
    novel = result.candidates[result.candidates.known_ids.eq("")].iloc[0]
    assert novel.supporting_pairs == novel.disjoint_pairs == 2
    assert novel.observed_median_delta == 0.0
    assert novel.direction_agreement == 0.5 and novel.contradictory_effects
    assert novel.near_activity_cliff


def test_hydrogen_is_a_supported_substitution(data):
    _, table, evidence = data
    table.loc[1, "smiles"] = "Cc1ccncc1"
    table = add_scaffolds(table.drop(columns="murcko"))
    series = decompose_scaffold(table, table.murcko.iloc[0], min_substituent_count=1)
    result = design((series, table, evidence))
    novel = result.candidates[result.candidates.known_ids.eq("")].iloc[0]
    assert novel.smiles == Chem.MolToSmiles(Chem.MolFromSmiles("CCc1ccncc1"))
    assert novel.fragment_to == "H"

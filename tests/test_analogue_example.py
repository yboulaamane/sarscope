"""The optional measured example is traceable, offline and actually usable."""

import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from sarscope.analogue_design import fit_design_model, generate_analogues, score_analogues
from sarscope.analogue_example import example_archive, example_dataset, example_files
from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.params import AnalogueDesignParams, ModelParams


def test_example_csv_is_measured_unique_and_matches_source_records():
    contents = example_files()
    csv = pd.read_csv(io.BytesIO(contents["egfr_ic50.csv"]))
    source = json.loads(contents["source.json"])
    assert csv.columns.tolist() == ["molecule_id", "smiles", "pactivity"]
    assert len(csv) == 106 and not csv.isna().any().any()
    assert csv.molecule_id.is_unique
    graphs = csv.smiles.map(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    assert graphs.is_unique
    dataset = example_dataset()
    assert dataset.source["assay"] == "CHEMBL944276"
    assert dataset.source["license"] == "CC-BY-SA-3.0"
    assert "four EGFR assay" in source["selection_note"]
    assert "not independently verified" in source["comparability_note"]
    evidence = dataset.measurements
    assert set(evidence.assay_chembl_id) == {"CHEMBL944276"}
    assert set(evidence.standard_type) == {"IC50"}
    assert set(evidence.standard_relation) == {"="}
    assert set(evidence.standard_units) == {"nM"}
    assert not evidence.record_id.str.startswith("upload:").any()
    assert np.allclose(evidence.pactivity, 9 - np.log10(evidence.standard_value))
    expected = evidence.groupby("molecule_id").pactivity.median()
    assert np.allclose(csv.pactivity, csv.molecule_id.map(expected))


def test_example_archive_contains_license_source_checks_and_valid_hashes():
    with zipfile.ZipFile(io.BytesIO(example_archive())) as archive:
        assert archive.testzip() is None
        manifest = json.loads(archive.read("manifest.sha256.json"))
        for name, checksum in manifest.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum
        guide = archive.read("README.md").decode()
        assert "CC BY-SA 3.0" in guide and "not a validation benchmark" in guide
        assert "La Motta" in guide and "10.1021/jm800829v" in guide
        recorded = json.loads(archive.read("verification.json"))
        assert recorded["walkthrough"]["unmeasured"] == 6
        assert all(check["passed"] for check in recorded["scoring_checks"])


def test_frozen_example_runs_through_actual_defaults_without_gate_changes(monkeypatch):
    import httpx

    monkeypatch.setattr(
        httpx.Client, "send", lambda *args, **kwargs: pytest.fail("Offline example")
    )
    dataset = example_dataset()
    table = add_scaffolds(dataset.table)
    recommended = dataset.source["recommended"]
    series = decompose_scaffold(table, recommended["scaffold"], min_substituent_count=1)
    result = generate_analogues(
        series,
        table,
        dataset.measurements,
        recommended["parent_id"],
        series.positions,
        AnalogueDesignParams(),
        source=dataset.source,
    )
    assert len(result.candidates) == 7
    assert result.candidates.known_ids.eq("").sum() == 6
    model = fit_design_model(
        table,
        dataset.measurements,
        AnalogueDesignParams(),
        ModelParams(
            regression_algorithms=("random_forest",),
            split="random",
            test_fraction=0.25,
            cv_folds=3,
            seed=42,
        ),
    )
    assert model.checks.passed.all()
    assert len(model.regression.test_truth) == 27
    selected = model.regression.scores.query("algorithm == 'random_forest'").iloc[0]
    assert selected.test_r2 > 0.5 and selected.design_pooled_cv_q2 > 0.3
    scored = score_analogues(result, model)
    novel = scored.candidates[scored.candidates.known_ids.eq("")]
    assert novel.predicted_pactivity.notna().sum() == 6
    assert novel.observed_pactivity.isna().all()

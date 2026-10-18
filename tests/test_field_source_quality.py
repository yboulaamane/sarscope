"""Source-graph release gates must not depend on a favourable model score."""

from types import SimpleNamespace

import pandas as pd
import pytest
from rdkit import Chem

from sarscope.field_qsar import dataset_from_assay, dataset_from_upload, run_field_qsar
from sarscope.field_source_quality import audit_field_structures, require_closed_shell
from sarscope.params import FieldQsarParams, ModelParams


def test_closed_shell_ions_are_not_confused_with_radicals():
    mols = [Chem.MolFromSmiles(s) for s in ("CCO", "C[NH3+]", "CC(=O)[O-]")]
    result = require_closed_shell(mols, ["neutral", "positive", "negative"])
    assert result.radical_electrons.eq(0).all()
    assert result.formal_charge.tolist() == [0, 1, -1]


def test_audit_preserves_radical_graphs_and_does_not_repair_or_drop_them():
    mol = Chem.MolFromSmiles("[CH2]CO")
    original = Chem.MolToSmiles(mol)
    result = audit_field_structures([mol], ["source-1"])
    assert result.radical_electrons.tolist() == [1]
    assert result.source_graph_review_required.tolist() == [True]
    assert result.radical_atom_indices_zero_based.tolist() == ["0"]
    with pytest.raises(ValueError, match="no automatic repair"):
        require_closed_shell([mol], ["source-1"])
    assert Chem.MolToSmiles(mol) == original


def test_bad_source_lengths_and_parse_failures_are_explicit():
    with pytest.raises(ValueError, match="matching lengths"):
        audit_field_structures([], ["missing"])
    with pytest.raises(ValueError, match="could not be parsed"):
        audit_field_structures([None], ["bad"])


def test_upload_rejects_radicals_before_curation_can_obscure_source():
    raw = pd.DataFrame({"molecule_id": ["bad"], "smiles": ["[CH2]CO"], "pactivity": [6.0]})
    with pytest.raises(ValueError, match="Source-structure review required"):
        dataset_from_upload(
            raw,
            [],
            id_col="molecule_id",
            smiles_col="smiles",
            value_col="pactivity",
            unit="p",
            endpoint="Ki",
            source_sha256="test",
        )


def test_assay_reaggregation_does_not_silently_certify_radical_graphs():
    table = pd.DataFrame({"molecule_id": ["bad"], "smiles": ["[CH2]CO"]})
    evidence = pd.DataFrame(
        {
            "molecule_id": ["bad"],
            "assay_chembl_id": ["A1"],
            "standard_type": ["Ki"],
            "standard_relation": ["="],
            "pactivity": [6.0],
        }
    )
    with pytest.raises(ValueError, match="Source-structure review required"):
        dataset_from_assay(table, evidence, "A1", "Ki", "Unreported")


def test_native_scoring_checks_actual_prepared_structures_before_any_fit():
    prepared = SimpleNamespace(
        params=FieldQsarParams(),
        alignment=SimpleNamespace(molecules=[Chem.MolFromSmiles("[CH2]CO")]),
        dataset=SimpleNamespace(table=pd.DataFrame({"molecule_id": ["bad"]})),
    )
    with pytest.raises(ValueError, match="Native 3D-QSAR requires closed-shell"):
        run_field_qsar(prepared, ModelParams())

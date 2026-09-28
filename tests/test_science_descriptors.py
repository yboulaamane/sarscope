"""Specification for sarscope.analysis.descriptors."""

import pandas as pd
import pytest
from rdkit import Chem

from sarscope.analysis.descriptors import CORE_DESCRIPTORS, add_descriptors, compute_descriptors

pytestmark = pytest.mark.science

VEMURAFENIB = "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2c[nH]c3ncc(-c4ccc(Cl)cc4)cc23)c1F"


def test_keys_are_exactly_the_core_descriptors():
    assert tuple(compute_descriptors(Chem.MolFromSmiles("CCO"))) == CORE_DESCRIPTORS


def test_vemurafenib_values():
    d = compute_descriptors(Chem.MolFromSmiles(VEMURAFENIB))
    assert d["MW"] == pytest.approx(489.931, abs=1e-3)
    assert d["logP"] == pytest.approx(5.5442, abs=1e-4)
    assert d["TPSA"] == pytest.approx(91.92, abs=1e-2)
    assert (d["RB"], d["NumHDonors"]) == (7, 2)


def test_acceptors_are_rdkit_numhacceptors_not_sorbents_nocount():
    # NumHAcceptors applies Lipinski's rules and gives 4; Lipinski.NOCount,
    # which Sorbent uses for catalogue matching, counts N+O and gives 6.
    assert compute_descriptors(Chem.MolFromSmiles(VEMURAFENIB))["NumHAcceptors"] == 4


def test_values_are_floats():
    d = compute_descriptors(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O"))
    assert all(isinstance(v, float) for v in d.values())


def test_add_descriptors_returns_a_copy_with_new_columns():
    table = pd.DataFrame({"molecule_id": ["a", "b"], "smiles": ["CCO", VEMURAFENIB]})
    out = add_descriptors(table)
    assert list(table.columns) == ["molecule_id", "smiles"]
    assert list(out.columns) == ["molecule_id", "smiles", *CORE_DESCRIPTORS]
    assert out.loc[1, "NumHAcceptors"] == 4


def test_add_descriptors_names_the_bad_molecule():
    table = pd.DataFrame({"molecule_id": ["good", "bad"], "smiles": ["CCO", "C1CC"]})
    with pytest.raises(ValueError, match="bad"):
        add_descriptors(table)

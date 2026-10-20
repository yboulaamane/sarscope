"""Published-equation controls and independent RDKit public-MIF oracle."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from sarscope.analysis.fields import GridSpec, compute_fields
from sarscope.analysis.mmff_fields import buffered_14_7, carbon_probe_parameters, energy_chunk
from sarscope.params import FieldQsarParams


def settings(**kwargs):
    return FieldQsarParams(
        method="comfa",
        field_definition="mmff94_mif",
        charge_model="mmff94",
        electrostatic_cutoff=30.0,
        **kwargs,
    )


def molecule(smiles="CO"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    return mol


def test_buffered_14_7_well_and_far_limit():
    radius, epsilon = np.array([3.9, 2.8]), np.array([0.07, 0.03])
    assert np.allclose(buffered_14_7(radius, radius, epsilon), -epsilon)
    assert np.isfinite(buffered_14_7(np.zeros(2), radius, epsilon)).all()
    assert (buffered_14_7(10 * radius, radius, epsilon) < 0).all()
    assert (abs(buffered_14_7(10 * radius, radius, epsilon)) < 1e-6).all()


def test_coulomb_charge_sign_zero_and_dielectric():
    common = dict(
        dielectric=1.0,
        dielectric_model="constant",
        steric_cutoff=30.0,
        electrostatic_cutoff=30.0,
    )
    args = [np.array([[9.0, 25.0], [16.0, 36.0]]), np.array([3.0, 3.0]), np.array([0.03, 0.02])]
    charge = np.array([0.1, -0.1])
    pos = energy_chunk(*args, charge, **common)["electrostatic"]
    neg = energy_chunk(*args, -charge, **common)["electrostatic"]
    zero = energy_chunk(*args, charge * 0, **common)["electrostatic"]
    half = energy_chunk(*args, charge, **(common | {"dielectric": 2.0}))["electrostatic"]
    assert np.array_equal(zero, np.zeros(2))
    assert np.array_equal(neg, -pos)
    assert np.allclose(half, pos / 2)
    assert pos[0] == pytest.approx(332.0716 * (0.1 / 3.05 - 0.1 / 5.05))
    origin = energy_chunk(*[np.zeros((1, 2)), *args[1:]], charge, **common)
    assert all(np.isfinite(values).all() for values in origin.values())


def test_mmff_fields_finite_no_mutation_and_chunk_invariance():
    mol = molecule("C[NH3+]")
    original = Chem.MolToMolBlock(mol)
    grid = GridSpec((0.0, 0.0, 0.0), (4, 3, 5), 0.8)
    first = compute_fields([mol], grid, settings(chunk_points=2))
    second = compute_fields([mol], grid, settings(chunk_points=200))
    assert np.array_equal(first.matrix, second.matrix)
    assert np.isfinite(first.matrix).all()
    assert np.max(abs(first.matrix)) <= 30
    assert Chem.MolToMolBlock(mol) == original
    assert first.charges[0]["probe_mmff_type"] == 1
    assert len(first.charges[0]["mmff_atom_types"]) == mol.GetNumAtoms()
    assert "no cross-field NaN mask" in first.convention


@pytest.mark.parametrize(
    "changes",
    [
        {"method": "comsia"},
        {"charge_model": "gasteiger"},
        {"allow_charge_fallback": True},
        {"electrostatic_cutoff": None},
        {"missing_field_policy": "training_mean"},
        {"dielectric_model": "distance"},
    ],
)
def test_mmff_profile_refuses_inconsistent_settings(changes):
    with pytest.raises(ValueError, match="MMFF94 MIF"):
        replace(settings(), **changes)


@pytest.mark.parametrize("smiles", ["CO", "c1ccncc1", "C[NH3+]", "CC(F)(F)F"])
def test_steric_matches_public_rdkit_mif(smiles):
    mif = pytest.importorskip("rdkit.Chem.rdMIF")
    mol = molecule(smiles)
    radius, epsilon, _ = carbon_probe_parameters(mol)
    oracle = mif.MMFFVdWaals(mol, probeAtomType=1, scaling=True, cutoff=0.0001)
    xyz = mol.GetConformer().GetPositions()
    for point in (np.array([5.0, 6.0, 7.0]), np.array([-6.0, 3.0, -4.0])):
        distance = np.linalg.norm(xyz - point, axis=1)
        ours = buffered_14_7(distance, radius, epsilon).sum()
        # The installed rdMIF docstring says kJ/mol, but its MMFF VDW result
        # numerically follows the public MMFF epsilon (kcal/mol) without x4.184.
        # The independent analytic well-depth test fixes our unit convention.
        expected = oracle(*point, 1000.0)
        assert ours == pytest.approx(expected, abs=1e-10, rel=1e-8)

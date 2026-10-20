"""Geometry checks for the optional offline comparator, not a production route."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from benchmarks.compare_o3a_alignment import overlay


@pytest.mark.parametrize("method", ["o3a_mmff", "o3a_crippen"])
@pytest.mark.parametrize("smiles", ["C[C@H](O)c1ccccc1", "C/C=C/c1ccccc1"])
def test_o3a_rigid_stereo_and_nonmutation(method, smiles):
    reference = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(reference, randomSeed=42) == 0
    probe = Chem.Mol(reference)
    transform = np.array(
        [[0.0, -1.0, 0.0, 8.0], [1.0, 0.0, 0.0, -3.0], [0.0, 0.0, 1.0, 5.0], [0.0, 0.0, 0.0, 1.0]]
    )
    AllChem.TransformMol(probe, transform)
    before = probe.GetConformer().GetPositions().copy()
    ref_before = reference.GetConformer().GetPositions().copy()
    aligned, audit = overlay(probe, reference, method)
    replay, _ = overlay(probe, reference, method)
    assert np.array_equal(before, probe.GetConformer().GetPositions())
    assert np.array_equal(ref_before, reference.GetConformer().GetPositions())
    assert audit["identity_and_stereo_preserved"]
    assert audit["distance_error_A"] < 1e-8
    assert audit["rotation_determinant"] == pytest.approx(1)
    assert np.allclose(aligned.GetConformer().GetPositions(), ref_before, atol=1e-3)
    assert np.array_equal(
        aligned.GetConformer().GetPositions(), replay.GetConformer().GetPositions()
    )


@pytest.mark.parametrize("method", ["o3a_mmff", "o3a_crippen"])
def test_o3a_requires_3d(method):
    mol = Chem.MolFromSmiles("CCO")
    with pytest.raises(ValueError, match="3D"):
        overlay(mol, mol, method)


def test_o3a_rejects_unknown_route():
    with pytest.raises(ValueError, match="Choose"):
        overlay(Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("CC"), "best")

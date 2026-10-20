"""Coordinate-level validation of generated alignment, not binding-pose claims.

The rigid-fit oracle is a NumPy SVD/Kabsch implementation, not rdMolAlign.
Bond lengths, tetrahedral volumes and alkene geometry are measured directly.
"""

import time
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem

from sarscope.analysis import align3d
from sarscope.conformers import embed_ligand_conformer, validate_conformer_stereo
from sarscope.params import FieldQsarParams

PAIRS = [
    ("aromatic", "Nc1ccccc1", "CNc1ccc(F)cc1"),
    ("chiral", "C[C@H](O)c1ccccc1", "C[C@H](O)c1ccc(F)cc1"),
    ("alkene_E", "C/C=C/c1ccccc1", "C/C=C/c1ccc(F)cc1"),
    ("alkene_Z", "C/C=C\\c1ccccc1", "C/C=C\\c1ccc(F)cc1"),
    ("fused", "c1ccc2ccccc2c1", "Cc1ccc2ccccc2c1"),
    # Replace a bridgehead H by OH while retaining the parent 3D neighbour
    # ordering. The written @/@@ flips when H becomes an explicit O branch.
    ("bridged", "C1C[C@H]2CC[C@@H]1C2", "C1C[C@@]2(O)CC[C@@H]1C2"),
]


@pytest.mark.parametrize("reported_ids", [[], [-1], [123]])
def test_failed_constrained_embedding_reports_no_pose_instead_of_bad_id(monkeypatch, reported_ids):
    reference = embed_ligand_conformer("Oc1ccccc1", 42)[0]

    def fail_embedding(molecule, *args):
        molecule.RemoveAllConformers()
        return reported_ids

    monkeypatch.setattr(align3d.AllChem, "EmbedMultipleConfs", fail_embedding)
    with pytest.raises(ValueError, match="Core-constrained embedding produced no conformer"):
        align3d._generate(
            "Nc1ccccc1", Chem.MolToMolBlock(reference), FieldQsarParams(alignment="constrained")
        )


def kabsch(probe, reference):
    """Proper rotation only; row-vector convention and independent centring."""
    left, right = probe - probe.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(left.T @ right)
    rotation = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    residual = left @ rotation - right
    return float(np.sqrt(np.square(residual).sum() / len(left))), rotation


def transformed(mol, *, mirror=False):
    result = Chem.Mol(mol)
    rotation = np.array([[0.36, -0.48, 0.8], [0.8, 0.60, 0], [-0.48, 0.64, 0.60]])
    assert np.linalg.det(rotation) == pytest.approx(1)
    xyz = mol.GetConformer().GetPositions() @ rotation + [7.25, -3.5, 1.75]
    if mirror:
        xyz[:, 0] *= -1
    for i, point in enumerate(xyz):
        result.GetConformer().SetAtomPosition(i, point)
    return result


def distances(mol):
    xyz = mol.GetConformer().GetPositions()
    return np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)


def signed_volumes(mol):
    """Four-neighbour tetrahedra in fixed atom-index order, independent of CIP."""
    xyz = mol.GetConformer().GetPositions()
    return {
        atom.GetIdx(): float(np.linalg.det(xyz[neighbors[:3]] - xyz[neighbors[3]]))
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        and len(neighbors := sorted(n.GetIdx() for n in atom.GetNeighbors())) == 4
    }


def check_geometry(mol):
    """Broad coordinate sanity bounds, not a force-field or bioactive-pose test."""
    xyz = mol.GetConformer().GetPositions()
    assert np.isfinite(xyz).all()
    table = Chem.GetPeriodicTable()
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        radii = sum(table.GetRcovalent(mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
        length = float(np.linalg.norm(xyz[i] - xyz[j]))
        assert 0.6 * radii < length < 1.45 * radii, (i, j, length, radii)
        if bond.GetStereo() in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            a, b = bond.GetStereoAtoms()
            axis = xyz[j] - xyz[i]
            axis /= np.linalg.norm(axis)
            vectors = [xyz[a] - xyz[i], xyz[b] - xyz[j]]
            vectors = [v - v.dot(axis) * axis for v in vectors]
            cosine = np.dot(*vectors) / np.prod([np.linalg.norm(v) for v in vectors])
            assert cosine < -0.5 if bond.GetStereo() == Chem.BondStereo.STEREOE else cosine > 0.5


@pytest.mark.parametrize("smiles", ["Fc1ccc(Cl)cc1", "C[C@H](O)c1ccccc1", "c1ccc2ccccc2c1"])
def test_rigid_symmetry_fit_agrees_with_independent_svd_oracle(smiles):
    reference = Chem.RemoveHs(embed_ligand_conformer(smiles, 42)[0])
    probe = transformed(
        Chem.RenumberAtoms(reference, list(reversed(range(reference.GetNumAtoms()))))
    )
    before = probe.GetConformer().GetPositions().copy()
    mapping, rmsd, transform = align3d.best_mapping(probe, reference, FieldQsarParams())
    expected, rotation = kabsch(
        before[[i for i, _ in mapping]],
        reference.GetConformer().GetPositions()[[j for _, j in mapping]],
    )
    assert rmsd == pytest.approx(expected, abs=2e-6)
    assert rmsd < 2e-6
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    np.testing.assert_array_equal(before, probe.GetConformer().GetPositions())


def test_reflections_cannot_be_used_to_claim_a_zero_rigid_rmsd():
    reference = Chem.RemoveHs(embed_ligand_conformer("F[C@](Cl)(Br)I", 42)[0])
    probe = transformed(reference, mirror=True)
    mapping, rmsd, transform = align3d.best_mapping(probe, reference, FieldQsarParams())
    expected, _ = kabsch(
        probe.GetConformer().GetPositions()[[i for i, _ in mapping]],
        reference.GetConformer().GetPositions()[[j for _, j in mapping]],
    )
    assert rmsd == pytest.approx(expected, abs=2e-6)
    assert rmsd > 0.1
    assert np.linalg.det(transform[:3, :3]) == pytest.approx(1)
    with pytest.raises(ValueError, match="stereochemistry"):
        align3d.align_series([reference, probe], ["ref", "mirror"], FieldQsarParams())


@pytest.mark.parametrize("route", ["rigid_mcs", "constrained"])
@pytest.mark.parametrize(("label", "ref_smiles", "smiles"), PAIRS, ids=[p[0] for p in PAIRS])
def test_generated_routes_preserve_graph_stereo_and_independently_measured_geometry(
    route, label, ref_smiles, smiles
):
    params = FieldQsarParams(alignment=route)
    reference = transformed(embed_ligand_conformer(ref_smiles, 7)[0])
    original, _ = embed_ligand_conformer(smiles, params.seed)
    block, _ = align3d._generate(smiles, Chem.MolToMolBlock(reference), params)
    output = Chem.MolFromMolBlock(block, removeHs=False)
    assert Chem.MolToSmiles(Chem.RemoveHs(output)) == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    validate_conformer_stereo(output, smiles)
    check_geometry(output)
    assert signed_volumes(original).keys() == signed_volumes(output).keys()
    for atom, volume in signed_volumes(original).items():
        assert volume * signed_volumes(output)[atom] > 0
    if route == "rigid_mcs":
        # V2000 coordinate serialization is rounded to four decimal places.
        np.testing.assert_allclose(distances(original), distances(output), atol=2e-4, rtol=0)
    else:
        heavy, ref = Chem.RemoveHs(output), Chem.RemoveHs(reference)
        mapping, rmsd, _ = align3d.best_mapping(heavy, ref, params, fixed_frame=True)
        assert rmsd < 2e-4
        assert len(mapping) >= 3


@pytest.mark.parametrize("route", ["rigid_mcs", "constrained"])
def test_real_series_replays_is_activity_blind_and_keeps_nonfirst_reference(route, monkeypatch):
    # These routes use RDKit, not Meeko/Gemmi. Their optional-extra UI gate is
    # tested separately; keep the geometry tests runnable in the base CI job.
    monkeypatch.setattr(align3d, "preparation_available", lambda: True)
    reference = embed_ligand_conformer("Oc1ccccc1", 42)[0]
    reference = Chem.RemoveHs(reference)
    inputs = [Chem.MolFromSmiles("Fc1ccccc1"), reference, Chem.MolFromSmiles("Nc1ccccc1")]
    ids = ["fluoro", "reference", "amino"]
    params = FieldQsarParams(alignment=route, reference_index=1, alignment_timeout=90)
    before = reference.GetConformer().GetPositions().copy()
    first = align3d.align_series(inputs, ids, params)
    for i, mol in enumerate(inputs):
        mol.SetDoubleProp("pactivity", 1000.0 - i * 100)
    order = [2, 0, 1]
    second = align3d.align_series(
        [inputs[i] for i in order], [ids[i] for i in order], replace(params, reference_index=2)
    )
    by_id = {mol.GetProp("_Name"): mol for mol in second.molecules}
    for original in first.molecules:
        np.testing.assert_array_equal(
            original.GetConformer().GetPositions(),
            by_id[original.GetProp("_Name")].GetConformer().GetPositions(),
        )
    np.testing.assert_array_equal(reference.GetConformer().GetPositions(), before)
    np.testing.assert_allclose(first.molecules[1].GetConformer().GetPositions(), before, atol=1e-4)
    assert list(first.report.molecule_id) == ids


def test_mcs_does_not_force_aromatic_and_saturated_rings_into_a_shared_core():
    left = Chem.RemoveHs(embed_ligand_conformer("CCc1ccccc1", 42)[0])
    right = Chem.RemoveHs(embed_ligand_conformer("CCC1CCCCC1", 42)[0])
    with pytest.raises(ValueError, match="three common heavy atoms"):
        align3d.best_mapping(left, right, FieldQsarParams())


@pytest.mark.parametrize("route", ["rigid_mcs", "constrained"])
def test_conflicting_bridgehead_state_is_not_silently_changed(route):
    # This assignment inverts only one bridgehead relative to the realizable
    # parent. Refusal is the expected result, not a missing successful example.
    invalid_state = "C1C[C@]2(O)CC[C@@H]1C2"
    with pytest.raises(ValueError, match="Stereo constraints failed"):
        align3d._generate(invalid_state, None, FieldQsarParams(alignment=route))


@pytest.mark.parametrize("bad", ["nan", "inf", "stereo", "components"])
def test_bad_supplied_reference_is_rejected_without_mutating_input(bad):
    mol = embed_ligand_conformer("C[C@H](O)c1ccccc1", 42)[0]
    if bad in ("nan", "inf"):
        mol.GetConformer().SetAtomPosition(0, (float(bad), 0, 0))
    elif bad == "stereo":
        mol = transformed(mol, mirror=True)
    else:
        mol = Chem.CombineMols(mol, Chem.MolFromSmiles("[Cl-]"))
    with pytest.raises(ValueError):
        align3d.align_series([mol], ["bad"], FieldQsarParams(alignment="rigid_mcs"))


def _hung_geometry_worker(connection):
    """A deliberately nonresponding job tests real spawn/termination, not mocks."""
    connection.send(("ready", None))
    connection.recv()
    while True:
        time.sleep(0.1)


def test_real_nonresponding_job_is_terminated_at_the_configured_budget(monkeypatch):
    monkeypatch.setattr(align3d, "_worker", _hung_geometry_worker)
    params = FieldQsarParams(molecule_timeout=3, alignment_timeout=30)
    worker = align3d._AlignmentGenerator(params, time.monotonic() + 30)
    started = time.monotonic()
    with pytest.raises(ValueError, match="conformer/MCS job exceeded its 3s"):
        with worker:
            worker.generate("Oc1ccccc1", None)
    assert not worker.process.is_alive()
    assert worker.process.exitcode is not None and worker.process.exitcode < 0
    assert time.monotonic() - started < 12

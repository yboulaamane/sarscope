"""Embedding checks, not assay or stereochemical identity validation."""

import pytest

from sarscope.conformers import embed_ligand_conformer, validate_conformer_stereo
from sarscope.docking import DockingError

REPORTED_PAIR = [
    "Cc1ncc(NC(=O)c2ccnc(C(C)(C)F)c2)cc1-c1cnc(OC2CCOCC2)c(N2[C@@H]3CC[C@@H]2COC3)c1",
    "Cc1ncc(NC(=O)c2ccnc(C(F)(F)F)c2)cc1-c1cnc(OC2CCOCC2)c(N2[C@@H]3CC[C@@H]2COC3)c1",
]


@pytest.mark.parametrize("smiles", REPORTED_PAIR)
def test_reported_bridgehead_assignments_fail_with_stereo_diagnostics(smiles):
    with pytest.raises(DockingError, match="Stereo constraints failed") as error:
        embed_ligand_conformer(smiles, 42)
    assert "FINAL_CHIRAL_BOUNDS" in str(error.value)
    assert "3 bounded strategies" in str(error.value)
    assert "No assigned stereochemistry was changed" in str(error.value)


@pytest.mark.parametrize("smiles", REPORTED_PAIR)
def test_diagnostic_alternative_embeds_and_prepares_without_mutating_supplied_state(smiles):
    pytest.importorskip("meeko")
    import json

    from rdkit import Chem

    from sarscope.docking_worker import _prepare_ligand

    # A diagnostic alternate, NOT a correction/identification of the experimental compound.
    alternative = smiles.replace("CC[C@@H]2COC3", "CC[C@H]2COC3")
    molecule, pdbqt = _prepare_ligand(alternative, 42)
    assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == alternative
    validate_conformer_stereo(molecule, alternative)
    metadata = json.loads(molecule.GetProp("sarscope_embedding"))
    assert metadata["stereo_verified_from_3d"] and metadata["uff_converged"]
    assert "ROOT" in pdbqt and "TORSDOF" in pdbqt
    assert "CC[C@@H]2COC3" in smiles


def test_first_embedding_failure_retries_random_coordinates_with_chirality_enforced(monkeypatch):
    from rdkit.Chem import AllChem

    real = AllChem.EmbedMolecule
    calls = []

    def fail_first(molecule, params):
        calls.append((params.useRandomCoords, params.enforceChirality, params.numThreads))
        return -1 if len(calls) == 1 else real(molecule, params)

    monkeypatch.setattr(AllChem, "EmbedMolecule", fail_first)
    molecule, metadata = embed_ligand_conformer("C[C@H](O)C(=O)O", 42)
    assert calls == [(False, True, 1), (True, True, 1)]
    assert len(metadata["attempts"]) == 2
    assert metadata["method"] == "ETKDGv3 random coordinates"
    validate_conformer_stereo(molecule, "C[C@H](O)C(=O)O")


def test_retry_plan_is_bounded_and_seed_does_not_overflow(monkeypatch):
    from rdkit.Chem import AllChem

    seeds = []

    def fail(molecule, params):
        seeds.append(params.randomSeed)
        assert params.enforceChirality and not params.ignoreSmoothingFailures
        return -1

    monkeypatch.setattr(AllChem, "EmbedMolecule", fail)
    with pytest.raises(DockingError, match="3 bounded strategies"):
        embed_ligand_conformer("CC", 2147483647)
    assert len(seeds) == 3 and all(0 < seed <= 2147483647 for seed in seeds)


def test_coordinate_derived_stereo_check_rejects_mirrored_geometry():
    from rdkit import Chem

    smiles = "C[C@H](O)C(=O)O"
    molecule, _ = embed_ligand_conformer(smiles, 42)
    conformer = molecule.GetConformer()
    for i in range(molecule.GetNumAtoms()):
        point = conformer.GetAtomPosition(i)
        conformer.SetAtomPosition(i, (-point.x, point.y, point.z))
    assert (
        Chem.MolToSmiles(Chem.RemoveHs(molecule)) == smiles
    )  # Copied tags alone hide the inversion.
    with pytest.raises(DockingError, match="3D geometry does not preserve"):
        validate_conformer_stereo(molecule, smiles)

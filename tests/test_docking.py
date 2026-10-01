"""Synthetic, offline tests; docking smoke results are not scientific validation."""

import io
import json
import math
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from tempfile import gettempdir

import pytest

from sarscope import docking
from sarscope.docking import DockingError, DockingSettings


def receptor_records():
    pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n"
    pdbqt = pdb.splitlines()[0][:66] + "    +0.000 C\n"
    return pdb, pdbqt


def ligands():
    return [
        {"molecule_id": "synthetic_A", "smiles": "Cc1ccccc1", "pactivity": 6.0},
        {"molecule_id": "synthetic_B", "smiles": "Oc1ccccc1", "pactivity": 8.0},
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"center": (math.nan, 0, 0)},
        {"center": (0, 0)},
        {"size": (26, 12, 12)},
        {"size": (12, 0, 12)},
        {"exhaustiveness": 17},
        {"exhaustiveness": 1.5},
        {"seed": True},
        {"seed": 0},
        {"n_poses": 4},
        {"timeout_seconds": 301},
    ],
)
def test_job_limits(kwargs):
    with pytest.raises(DockingError):
        DockingSettings(**({"center": (0, 0, 0)} | kwargs))


def test_receptor_validation_strips_headers_and_checks_box():
    pdb, pdbqt = receptor_records()
    safe = docking.validate_receptor_pair(
        "HEADER ignored\n" + pdb, pdbqt, DockingSettings((0, 0, 0))
    )
    assert "HEADER" not in safe and safe.endswith("END\n")
    with pytest.raises(DockingError, match="box contains no"):
        docking.validate_receptor_pair(pdb, pdbqt, DockingSettings((100, 0, 0)))


@pytest.mark.parametrize(
    "transform, message",
    [
        (lambda s: s.replace("   0.000", "   1.000", 1), "coordinates disagree"),
        (lambda s: s.replace(" CA ", " CB "), "identical heavy-atom"),
        (lambda s: s.replace("ALA", "HOH"), "protein-only"),
        (lambda s: s[:16] + "A" + s[17:], "alternate"),
        (lambda s: "ROOT\n" + s, "single rigid"),
        (lambda s: s + s, "serial numbers"),
        (lambda s: s.replace("+0.000", "   nan"), "finite"),
    ],
)
def test_receptor_mismatch_and_unsafe_records(transform, message):
    pdb, pdbqt = receptor_records()
    with pytest.raises(DockingError, match=message):
        docking.validate_receptor_pair(pdb, transform(pdbqt), DockingSettings((0, 0, 0)))


def test_contacts_compare_residue_and_type_not_ligand_atom_indices():
    left = [
        {"protein_residue": "ALA1.A", "interaction": "Hydrophobic", "distance_angstrom": 3.5},
        {"protein_residue": "ALA1.A", "interaction": "Hydrophobic", "distance_angstrom": 3.0},
        {"protein_residue": "SER2.A", "interaction": "HBDonor", "distance_angstrom": 2.5},
    ]
    right = [
        {"protein_residue": "ALA1.A", "interaction": "Hydrophobic", "distance_angstrom": 3.2},
        {"protein_residue": "SER2.A", "interaction": "HBAcceptor", "distance_angstrom": 2.8},
    ]
    changes = docking.interaction_changes(left, right)
    assert [x["change_A_to_B"] for x in changes] == ["retained", "gained", "lost"]
    assert changes[0]["distance_A_Angstrom"] == 3.0
    assert changes[0]["distance_B_Angstrom"] == 3.2
    assert docking.interaction_changes([], []) == []


def test_request_hash_invalidates_all_inputs():
    request = {"ligands": ligands(), "settings": asdict(DockingSettings((0, 0, 0)))}
    initial = docking.docking_request_key(request)
    assert initial == docking.docking_request_key(dict(reversed(list(request.items()))))
    request["ligands"][0]["pactivity"] += 0.1
    assert initial != docking.docking_request_key(request)


@pytest.mark.parametrize("values", [[], ligands()[:1], ligands() * 2])
def test_exactly_two_ligands(values):
    with pytest.raises(DockingError, match="exactly two"):
        docking.run_pair_docking(values, "", "", DockingSettings((0, 0, 0)), receptor_label="test")


def test_one_job_per_host(monkeypatch):
    filelock = pytest.importorskip("filelock")
    monkeypatch.setattr(docking, "docking_unavailable_reason", lambda: None)
    monkeypatch.setattr(docking, "vina_binary", lambda: "/unused/vina")
    pdb, pdbqt = receptor_records()
    with filelock.FileLock(str(Path(gettempdir()) / "sarscope-pair-docking.lock")):
        with pytest.raises(DockingError, match="Another pair"):
            docking.run_pair_docking(
                ligands(), pdb, pdbqt, DockingSettings((0, 0, 0)), receptor_label="test"
            )


def test_timeout_kills_worker_and_child_processes(monkeypatch, tmp_path):
    pytest.importorskip("filelock")
    monkeypatch.setattr(docking, "docking_unavailable_reason", lambda: None)
    monkeypatch.setattr(docking, "vina_binary", lambda: "/unused/vina")
    real_popen = subprocess.Popen
    child_pid = tmp_path / "child.pid"
    processes = []
    script = (
        "import subprocess,sys,time; from pathlib import Path; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"Path({str(child_pid)!r}).write_text(str(p.pid)); time.sleep(60)"
    )

    def replace_worker(command, **kwargs):
        process = real_popen([sys.executable, "-c", script], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(docking.subprocess, "Popen", replace_worker)
    pdb, pdbqt = receptor_records()
    with pytest.raises(DockingError, match="total timeout"):
        docking.run_pair_docking(
            ligands(),
            pdb,
            pdbqt,
            DockingSettings((0, 0, 0), timeout_seconds=1),
            receptor_label="test",
        )
    assert processes[0].poll() is not None
    assert child_pid.exists()
    # A killed, not-yet-reaped grandchild can remain a zombie briefly on Linux.
    stat = Path(f"/proc/{child_pid.read_text()}/stat")
    if stat.exists():
        assert stat.read_text().split()[2] in {"Z", "X"}
    else:
        with pytest.raises(ProcessLookupError):
            os.kill(int(child_pid.read_text()), 0)


@pytest.mark.parametrize("smiles", ["CC.Cl", "CC(O)C(=O)O", "CC=CC", "C1CCCCCCCC1", "[Na+]"])
def test_ambiguous_or_unsupported_ligand_preparation(smiles):
    pytest.importorskip("meeko")
    from sarscope.docking_worker import _prepare_ligand

    with pytest.raises(DockingError):
        _prepare_ligand(smiles, 42)


@pytest.mark.parametrize("exploratory", [False, True])
def test_pair_docking_real_engine_smoke_and_artifact(exploratory):
    if docking.docking_unavailable_reason():
        pytest.skip(
            "Install the optional docking extra and the Vina executable for this smoke test"
        )
    from meeko import MoleculePreparation, PDBQTWriterLegacy, Polymer, ResidueChemTemplates
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.AddHs(Chem.MolFromFASTA("AA"))
    embedding = AllChem.ETKDGv3()
    embedding.randomSeed = 42
    assert AllChem.EmbedMolecule(molecule, embedding) == 0
    AllChem.UFFOptimizeMolecule(molecule)
    polymer = Polymer.from_pdb_string(
        Chem.MolToPDBBlock(Chem.RemoveHs(molecule)),
        chem_templates=ResidueChemTemplates.create_from_defaults(),
        mk_prep=MoleculePreparation(),
    )
    receptor, flexible = PDBQTWriterLegacy.write_from_polymer(polymer)
    assert not flexible
    phases = []
    values = ligands()
    if exploratory:
        values[0] |= {"smiles": "C[C@H](O)C(=O)O", "source_smiles": "CC(O)C(=O)O"}
    provenance = {
        "site": {"kind": "bound_ligand", "id": "synthetic reference"},
        "reference_pdb": "REMARK synthetic reference for artifact test only\nEND\n",
        "excluded_essential_components": [],
    }
    result = docking.run_pair_docking(
        values,
        polymer.to_pdb(),
        receptor,
        DockingSettings((0, 0, 0), (12, 12, 12), exhaustiveness=1, n_poses=1, timeout_seconds=60),
        receptor_label="Synthetic AA dipeptide: technical smoke only",
        on_progress=phases.append,
        receptor_provenance=provenance,
    )
    assert phases
    assert result["manifest"]["cpu"] == 1
    assert result["manifest"]["analysed_pose_rank"] == 1
    assert result["manifest"]["receptor_preparation"] == provenance
    assert len(result["ligands"]) == 2
    if exploratory:
        assert result["ligands"][0]["stereochemistry"]["status"] == "exploratory_selection"
        assert result["manifest"]["ligands"][0]["source_smiles"] == "CC(O)C(=O)O"
        assert any("exploratory stereoisomer" in w for w in result["manifest"]["warnings"])
    for pose in result["ligands"]:
        assert math.isfinite(pose["score_kcal_mol"])
        restored = Chem.MolFromMolBlock(pose["sdf"])
        assert Chem.MolToSmiles(restored) == Chem.MolToSmiles(Chem.MolFromSmiles(pose["smiles"]))
        assert restored.GetConformer().Is3D()
    # The synthetic phenol pose exercises both implicit H-bonds and network HTML.
    assert result["ligands"][1]["interactions"]
    assert "<script" in result["ligands"][1]["diagram_html"]
    with zipfile.ZipFile(io.BytesIO(docking.docking_result_zip(result))) as archive:
        assert "ligand_A_prepared.pdbqt" in archive.namelist()
        assert "ligand_B.html" in archive.namelist()
        assert "interaction_changes.csv" in archive.namelist()
        assert archive.read("reference_ligand.pdb").decode() == provenance["reference_pdb"]
        assert json.loads(archive.read("manifest.json"))["schema"] == "sarscope-pair-docking-v1"

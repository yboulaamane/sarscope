"""Offline structure/site tests; all coordinates are generated, not benchmark evidence."""

import hashlib
import io
import json
import math
import tarfile
import zipfile
from pathlib import Path

import httpx
import pytest

from sarscope.docking import DockingError, DockingSettings, validate_receptor_pair
from sarscope.receptor import (
    box_from_coordinates,
    check_site_components,
    inspect_structure,
    prepare_receptor,
)
from sarscope.sources.pdb import PdbClient, normalise_pdb_id


@pytest.fixture
def synthetic_cif():
    gemmi = pytest.importorskip("gemmi")
    from rdkit import Chem
    from rdkit.Chem import AllChem

    protein = Chem.AddHs(Chem.MolFromFASTA("AA"))
    AllChem.EmbedMolecule(protein, randomSeed=42)
    AllChem.UFFOptimizeMolecule(protein)
    structure = gemmi.read_pdb_string(Chem.MolToPDBBlock(Chem.RemoveHs(protein)))
    structure[0][0].name = "LONG"
    ligand_chain = gemmi.Chain("L")
    ligand = gemmi.Residue()
    ligand.name, ligand.seqid, ligand.het_flag = "LIG", gemmi.SeqId(5, " "), "H"
    for i in range(6):
        atom = gemmi.Atom()
        atom.name, atom.element = f"C{i + 1}", gemmi.Element("C")
        atom.pos = gemmi.Position(math.cos(i), math.sin(i), 3.0)
        ligand.add_atom(atom)
    ligand_chain.add_residue(ligand)
    structure[0].add_chain(ligand_chain)
    structure.setup_entities()
    return structure.make_mmcif_document().as_string()


@pytest.mark.parametrize("value", ["https://evil.example", "../../x", "1ABC/", "AF-P12345", ""])
def test_only_experimental_pdb_ids_are_accepted(value):
    with pytest.raises(DockingError):
        normalise_pdb_id(value)
    assert normalise_pdb_id(" 1abc ") == "1ABC"


def test_rcsb_search_is_bounded_and_uses_exact_target_mapping():
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "search.rcsb" in str(request.url):
            return httpx.Response(200, json={"result_set": [{"identifier": "1ABC"}]})
        return httpx.Response(
            200,
            json={
                "data": {
                    "entries": [
                        {
                            "rcsb_id": "1ABC",
                            "struct": {"title": "Synthetic construct"},
                            "rcsb_entry_info": {"resolution_combined": [2.0]},
                            "polymer_entities": [
                                {
                                    "entity_poly": {"type": "polypeptide(L)"},
                                    "rcsb_polymer_entity": {"pdbx_mutation": "G12D"},
                                    "rcsb_polymer_entity_container_identifiers": {
                                        "auth_asym_ids": ["A"],
                                        "uniprot_ids": ["P12345"],
                                    },
                                }
                            ],
                        }
                    ]
                }
            },
        )

    with PdbClient(transport=httpx.MockTransport(respond)) as client:
        rows = client.search("P12345")
    assert requests[0]["request_options"]["paginate"]["rows"] == 20
    assert requests[0]["request_options"]["results_content_type"] == ["experimental"]
    assert rows[0]["matched_chains"] == ["A"]
    assert rows[0]["chain_mutations"] == {"A": "G12D"}


def test_empty_search_and_graphql_errors_are_handled():
    with PdbClient(transport=httpx.MockTransport(lambda _: httpx.Response(204))) as client:
        assert client.search("P12345") == []
    with PdbClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"errors": [{}]}))
    ) as client:
        with pytest.raises(DockingError, match="metadata query failed"):
            client.metadata(["1ABC"])


def test_box_is_derived_from_extents_not_a_protein_centroid_and_never_clipped():
    box = box_from_coordinates([[10, 20, 30], [14, 26, 38]])
    assert box["center"] == [12, 23, 34]
    assert box["size"] == [12, 14, 16]
    assert box["fits_browser_limit"]
    huge = box_from_coordinates([[0, 0, 0], [30, 0, 0]])
    assert huge["size"][0] == 38 and not huge["fits_browser_limit"]
    with pytest.raises(DockingError):
        box_from_coordinates([[math.nan, 0, 0]])


def test_chain_mapping_reference_selection_and_provenance(synthetic_cif):
    structure = inspect_structure(synthetic_cif, ["LONG"])
    assert structure["provenance"]["chain_mapping"] == {"LONG": "A"}
    assert (
        structure["provenance"]["source_cif_sha256"]
        == hashlib.sha256(synthetic_cif.encode()).hexdigest()
    )
    assert "LIG" not in structure["protein_pdb"]
    assert len(structure["references"]) == 1
    ref = structure["references"][0]
    assert ref["id"] == "L:5:LIG"
    assert ref["kind"] == "bound_ligand" and ref["fits_browser_limit"]
    with pytest.raises(DockingError, match="absent"):
        inspect_structure(synthetic_cif, ["Z"])


def test_nearby_cofactor_guard_applies_even_if_selected_as_reference():
    provenance = {
        "site": {"id": "A:5:ZN"},
        "excluded_essential_components": [
            {"id": "A:5:ZN", "name": "ZN", "coordinates": [[0, 0, 0]]}
        ],
    }
    with pytest.raises(DockingError, match="Excluded metal/cofactor"):
        check_site_components(provenance, DockingSettings((0, 0, 0)))
    check_site_components(provenance, DockingSettings((100, 0, 0)))


def test_meeko_receptor_preparation_real_smoke(synthetic_cif):
    pytest.importorskip("meeko")
    structure = inspect_structure(synthetic_cif, ["LONG"])
    result = prepare_receptor(structure["protein_pdb"], structure["provenance"])
    validate_receptor_pair(
        result["protein_pdb"], result["receptor_pdbqt"], DockingSettings((0, 0, 0))
    )
    assert result["provenance"]["meeko_version"]
    assert "no pKa" in result["provenance"]["preparation"]


def test_meeko_does_not_silently_delete_a_residue_missing_heavy_atoms(synthetic_cif):
    pytest.importorskip("meeko")
    import gemmi

    structure = gemmi.make_structure_from_block(gemmi.cif.read_string(synthetic_cif).sole_block())
    structure[0][0][0].remove_atom("CB", "\x00")
    inspected = inspect_structure(structure.make_mmcif_document().as_string(), ["LONG"])
    with pytest.raises(DockingError, match="Meeko preparation failed"):
        prepare_receptor(inspected["protein_pdb"], inspected["provenance"])


def test_docking_rechecks_excluded_components_against_the_actual_edited_grid():
    from sarscope.docking import run_pair_docking

    pdb = "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C  \nEND\n"
    pdbqt = pdb.splitlines()[0][:66] + "    +0.000 C\n"
    provenance = {
        "site": {"center": [100, 0, 0]},  # Initially proposed far from the excluded metal.
        "excluded_essential_components": [{"id": "A:5:ZN", "coordinates": [[0, 0, 0]]}],
    }
    ligands = [
        {"molecule_id": "A", "smiles": "CC", "pactivity": 6.0},
        {"molecule_id": "B", "smiles": "CCC", "pactivity": 8.0},
    ]
    with pytest.raises(DockingError, match="Excluded metal/cofactor"):
        run_pair_docking(
            ligands,
            pdb,
            pdbqt,
            DockingSettings((0, 0, 0)),
            receptor_label="synthetic test",
            receptor_provenance=provenance,
        )


def test_fpocket_output_parser_records_scores_geometry_and_oversize(tmp_path):
    from sarscope.receptor_worker import parse_fpocket_output

    (tmp_path / "pockets").mkdir()
    (tmp_path / "protein_info.txt").write_text(
        "Pocket 1 :\n Score : 0.5\n Druggability Score : 0.7\n Volume : 100\n"
    )
    atoms = [
        "ATOM      1    O STP     1      10.000  20.000  30.000    0.00     4.14\n",
        "ATOM      2    O STP     1      40.000  20.000  30.000    0.00     4.14\n",
    ]
    (tmp_path / "pockets" / "pocket1_vert.pqr").write_text("".join(atoms))
    rows = parse_fpocket_output(tmp_path)
    assert rows[0]["kind"] == "predicted_pocket"
    assert rows[0]["score"] == 0.5
    assert rows[0]["druggability_score"] == 0.7
    assert rows[0]["center"] == [25, 20, 30]
    assert not rows[0]["fits_browser_limit"]


def test_engine_installer_checks_package_and_binary_before_execution(monkeypatch, tmp_path):
    zstandard = pytest.importorskip("zstandard")
    from sarscope import receptor_worker as worker

    binary = b"synthetic executable - never executed"
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        item = tarfile.TarInfo("bin/fpocket")
        item.size = len(binary)
        archive.addfile(item, io.BytesIO(binary))
    package = io.BytesIO()
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr(
            "pkg-synthetic.tar.zst", zstandard.ZstdCompressor().compress(buffer.getvalue())
        )
    data = package.getvalue()
    monkeypatch.delenv("SARSCOPE_FPOCKET_BINARY", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    monkeypatch.setattr(worker.shutil, "which", lambda _: None)
    monkeypatch.setattr(worker.platform, "system", lambda: "Linux")
    monkeypatch.setattr(worker.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(worker, "FPOCKET_PACKAGE_SHA256", hashlib.sha256(data).hexdigest())
    monkeypatch.setattr(worker, "FPOCKET_BINARY_SHA256", hashlib.sha256(binary).hexdigest())
    calls = []

    def download(*args, **kwargs):
        calls.append(args)
        return httpx.Client(
            transport=httpx.MockTransport(lambda _: httpx.Response(200, content=data))
        ).stream("GET", "https://example.test")

    monkeypatch.setattr(httpx, "stream", download)
    path, identity = worker._install_fpocket()
    assert Path(path).read_bytes() == binary
    assert identity["binary_sha256"] == hashlib.sha256(binary).hexdigest()
    worker._install_fpocket()
    assert len(calls) == 1  # Valid cached executable needs no network request.
    monkeypatch.setattr(worker, "FPOCKET_BINARY_SHA256", "wrong checksum")
    with pytest.raises(DockingError, match="checksum mismatch"):
        worker._install_fpocket()


def test_fpocket_real_engine_smoke_when_explicitly_configured(synthetic_cif):
    import os

    cached = Path.home() / ".cache" / "sarscope" / "fpocket-4.2.2" / "fpocket"
    if not os.environ.get("SARSCOPE_FPOCKET_BINARY") and not cached.exists():
        pytest.skip("Explicitly configure fpocket to run the offline native-engine smoke test")
    from sarscope.receptor import predict_pockets

    structure = inspect_structure(synthetic_cif, ["LONG"])
    result = predict_pockets(structure["protein_pdb"])
    assert result["engine"]["binary_sha256"]
    assert isinstance(result["pockets"], list)  # A tiny synthetic peptide can have no pockets.


def test_consistent_alternate_location_choice_and_modified_residue_rejection(synthetic_cif):
    import gemmi

    from sarscope.docking import _atoms

    structure = gemmi.make_structure_from_block(gemmi.cif.read_string(synthetic_cif).sole_block())
    residue = structure[0][0][0]
    original = residue[0]
    original.altloc, original.occ = "A", 0.3
    selected = original.clone()
    selected.altloc, selected.occ = "B", 0.7
    selected.pos.x += 0.2
    residue.add_atom(selected)
    inspected = inspect_structure(structure.make_mmcif_document().as_string(), ["LONG"])
    assert inspected["provenance"]["alternate_locations"] == ["LONG:1:ALA=B"]
    atoms = _atoms(inspected["protein_pdb"], pdbqt=False)
    assert abs(atoms[("A", "1", "", "ALA", selected.name)][0] - selected.pos.x) < 0.001
    residue.name = "MSE"
    with pytest.raises(DockingError, match="unsupported protein residue"):
        inspect_structure(structure.make_mmcif_document().as_string(), ["LONG"])


def test_native_fpocket_detects_an_offline_synthetic_cavity():
    import os

    import numpy as np

    from sarscope.receptor import predict_pockets

    cached = Path.home() / ".cache" / "sarscope" / "fpocket-4.2.2" / "fpocket"
    if not os.environ.get("SARSCOPE_FPOCKET_BINARY") and not cached.exists():
        pytest.skip("Native fpocket not configured/cached; no download from this offline test")
    pytest.importorskip("filelock")
    # An artificial atom shell, NOT a physical protein or pocket-validation dataset.
    rng = np.random.default_rng(42)
    vectors = rng.normal(size=(180, 3))
    xyz = vectors / np.linalg.norm(vectors, axis=1)[:, None] * rng.uniform(4.5, 6.2, (180, 1))
    protein = (
        "".join(
            f"ATOM  {i:5d}  CA  ALA A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00           C  \n"
            for i, (x, y, z) in enumerate(xyz, 1)
        )
        + "END\n"
    )
    result = predict_pockets(protein)
    assert result["pockets"]
    assert all(p["kind"] == "predicted_pocket" for p in result["pockets"])
    assert all(np.isfinite(p["center"]).all() for p in result["pockets"])

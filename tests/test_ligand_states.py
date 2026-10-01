"""Stereo choices preserve source chemistry and do not fabricate assay assignments."""

import pytest

from sarscope.docking import DockingError, DockingSettings
from sarscope.ligand_states import enumerate_ligand_states, inspect_ligand, validate_ligand_state


@pytest.mark.parametrize(
    "smiles,kind", [("CC(O)C(=O)O", "Atom_Tetrahedral"), ("CC=CC", "Bond_Double")]
)
def test_preflight_identifies_atom_and_double_bond_stereo(smiles, kind):
    assert inspect_ligand(smiles)["unassigned_features"][0]["type"] == kind
    alternatives = enumerate_ligand_states(smiles)
    assert len(alternatives) == 2
    assert alternatives == enumerate_ligand_states(smiles)
    for selected in alternatives:
        state = validate_ligand_state(smiles, selected)
        assert state["source_smiles"] == smiles
        assert state["status"] == "exploratory_selection"
        assert not state["experimental_configuration_verified"]
        assert not inspect_ligand(selected)["unassigned_features"]


def test_existing_stereochemistry_is_retained_during_enumeration():
    source = "C[C@H](O)CC=CC"
    alternatives = enumerate_ligand_states(source)
    assert len(alternatives) == 2
    for selected in alternatives:
        validate_ligand_state(source, selected)
    assert validate_ligand_state("C[C@H](O)C(=O)O", "O=C(O)[C@@H](O)C")["status"] == "as_supplied"


@pytest.mark.parametrize(
    "source,selected",
    [
        ("C[C@H](O)CC(N)C", "C[C@@H](O)C[C@H](N)C"),
        ("C/C=C/CC(O)C", "C/C=C\\C[C@H](O)C"),
        ("CC(O)C(=O)O", "C[C@H](N)C(=O)O"),
        ("CC(O)C(=O)O", "C[C@H](O)C(=O)[O-]"),
        ("CC(O)C(=O)O", "C[13C@H](O)C(=O)O"),
    ],
)
def test_manual_state_cannot_change_existing_stereo_graph_charge_or_isotopes(source, selected):
    with pytest.raises(DockingError, match="must retain"):
        validate_ligand_state(source, selected)


def test_unresolved_states_and_large_enumerations_stop_before_embedding():
    with pytest.raises(DockingError, match="Resolve unspecified"):
        validate_ligand_state("CC=CC", "CC=CC")
    with pytest.raises(DockingError, match="browser limit"):
        enumerate_ligand_states("FC(Cl)C(Br)C(O)C(N)C")


def test_backend_retains_source_activity_and_derives_state_provenance(monkeypatch):
    from sarscope import docking

    values = [
        {
            "molecule_id": "A",
            "smiles": "C[C@H](O)C(=O)O",
            "source_smiles": "CC(O)C(=O)O",
            "pactivity": 6.0,
        },
        {"molecule_id": "B", "smiles": "CC", "pactivity": 8.0},
    ]
    captured = []
    monkeypatch.setattr(docking, "validate_receptor_pair", lambda *args: "protein")
    monkeypatch.setattr(docking, "docking_unavailable_reason", lambda: None)
    monkeypatch.setattr(docking, "vina_binary", lambda: "/unused/vina")

    def worker(request, **kwargs):
        captured.append(request)
        return {"ligands": [v | {"interactions": []} for v in request["ligands"]]}

    monkeypatch.setattr(docking, "run_structure_job", worker)
    result = docking.run_pair_docking(
        values, "", "", DockingSettings((0, 0, 0)), receptor_label="test"
    )
    state = captured[0]["ligands"][0]
    assert state["source_smiles"] == "CC(O)C(=O)O"
    assert state["pactivity"] == values[0]["pactivity"]
    assert state["stereochemistry"]["status"] == "exploratory_selection"
    assert result["ligands"][1]["stereochemistry"]["status"] == "as_supplied"
    assert "stereochemistry" not in values[0]  # No mutation of source records.
    values[0]["smiles"] = "CC(O)C(=O)O"
    with pytest.raises(DockingError, match="Resolve unspecified"):
        docking.run_pair_docking(values, "", "", DockingSettings((0, 0, 0)), receptor_label="test")
    assert len(captured) == 1

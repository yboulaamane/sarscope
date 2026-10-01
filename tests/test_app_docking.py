"""The pair-docking controls are explicit, gated, and never run on page load."""

import io

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sarscope import docking_ui  # noqa: E402

PAIR = [
    {"molecule_id": "synthetic_A", "smiles": "Cc1ccccc1", "pactivity": 6.0},
    {"molecule_id": "synthetic_B", "smiles": "Oc1ccccc1", "pactivity": 8.0},
]


def panel():
    return AppTest.from_string(
        "from sarscope.docking_ui import show_cliff_docking\n"
        f"show_cliff_docking({PAIR!r}, key='test_docking')"
    ).run()


def test_docking_is_disabled_by_default_and_not_imported_on_startup(monkeypatch):
    def must_not_run(*args, **kwargs):
        pytest.fail("Docking availability and work must not run before user opt-in")

    monkeypatch.setattr(docking_ui, "docking_unavailable_reason", must_not_run)
    monkeypatch.setattr(docking_ui, "run_pair_docking", must_not_run)
    app = panel()
    assert not app.exception
    assert not app.button
    assert app.checkbox[0].value is False


def test_missing_dependencies_are_friendly_and_do_not_break_landscape(monkeypatch):
    monkeypatch.setattr(docking_ui, "docking_unavailable_reason", lambda: "Vina not installed")
    app = panel()
    app.checkbox[0].check().run()
    assert not app.exception
    assert app.info[0].value == "Vina not installed"
    assert not app.button


def result_fixture():
    ligand = {"sdf": "", "pdbqt": "", "prepared_pdbqt": "", "diagram_html": "", "interactions": []}
    return {
        "ligands": [
            ligand | PAIR[0] | {"score_kcal_mol": -5.0},
            ligand | PAIR[1] | {"score_kcal_mol": -6.0},
        ],
        "manifest": {"warnings": []},
        "protein_pdb": "",
        "receptor_pdbqt": "",
        "interaction_changes": [],
    }


def test_user_button_only_and_cached_results_hide_when_settings_change(monkeypatch):
    monkeypatch.setattr(docking_ui, "docking_unavailable_reason", lambda: None)
    calls = []

    def run(*args, **kwargs):
        calls.append((args, kwargs))
        return result_fixture()

    class Upload:
        size = 100

        def getvalue(self):
            return b"technical test"

    def upload(label, **kwargs):
        return None if "reference" in label.lower() else Upload()

    monkeypatch.setattr(docking_ui, "run_pair_docking", run)
    monkeypatch.setattr(docking_ui.st, "file_uploader", upload)
    app = panel()
    app.checkbox(key="test_docking_enabled").check().run()
    assert app.button(key="test_docking_run").disabled
    app.text_input(key="test_docking_label").set_value("Synthetic receptor")
    for axis in "xyz":
        app.number_input(key=f"test_docking_center_{axis}").set_value(0.0)
    app.checkbox(key="test_docking_confirmed").check().run()
    assert not app.exception
    assert not app.button(key="test_docking_run").disabled
    assert not calls
    app.session_state["test_docking_3d"] = False
    app.button(key="test_docking_run").click().run()
    assert not app.exception
    assert len(calls) == 1
    assert [m.value for m in app.metric][:4] == [
        "-5.00 kcal/mol",
        "-6.00 kcal/mol",
        "+2.00",
        "-1.00 kcal/mol",
    ]
    app.button(key="test_docking_run").click().run()
    assert len(calls) == 1
    app.number_input(key="test_docking_seed").set_value(43).run()
    assert not app.metric
    assert any("previous result is hidden" in i.value for i in app.info)
    assert len(calls) == 1


def test_reference_box_requires_bound_3d_not_a_2d_depiction():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdDepictor

    molecule = Chem.MolFromSmiles("Cc1ccccc1")

    def sdf_bytes():
        buffer = io.StringIO()
        writer = Chem.SDWriter(buffer)
        writer.write(molecule)
        writer.close()
        return buffer.getvalue().encode()

    rdDepictor.Compute2DCoords(molecule)
    with pytest.raises(docking_ui.DockingError, match="bound 3D"):
        docking_ui.reference_box(sdf_bytes())
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, randomSeed=42)
    center, size = docking_ui.reference_box(sdf_bytes())
    assert len(center) == len(size) == 3
    assert all(6 <= s <= 25 for s in size)


def test_diagrams_are_iframe_isolated(monkeypatch):
    import streamlit.components.v1 as components

    calls = []
    monkeypatch.setattr(components, "html", lambda *a, **k: calls.append((a, k)))
    docking_ui._html("<p>Library-generated HTML</p>", height=550)
    assert calls[0][1] == {"height": 550, "scrolling": True}

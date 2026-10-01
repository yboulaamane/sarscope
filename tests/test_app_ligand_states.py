"""Undefined stereo is actionable before jobs; changes invalidate previous choices/results."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sarscope import docking_ui  # noqa: E402
from sarscope.ligand_states import validate_ligand_state  # noqa: E402

PAIR = [
    {"molecule_id": "A", "smiles": "CC(O)C(=O)O", "pactivity": 6.0},
    {"molecule_id": "B", "smiles": "Oc1ccccc1", "pactivity": 8.0},
]


def panel(monkeypatch, pair=PAIR):
    monkeypatch.setattr(docking_ui, "docking_unavailable_reason", lambda: None)
    app = AppTest.from_string(
        "from sarscope.docking_ui import show_cliff_docking\n"
        f"show_cliff_docking({pair!r}, key='stereo')"
    ).run()
    app.checkbox(key="stereo_enabled").check().run()
    assert not app.exception
    return app


def test_generate_choose_accept_then_dock_preserves_source_and_hides_stale_results(monkeypatch):
    calls = []

    class Upload:
        size = 100

        def getvalue(self):
            return b"technical test"

    monkeypatch.setattr(
        docking_ui.st,
        "file_uploader",
        lambda label, **kwargs: None if "reference" in label.lower() else Upload(),
    )

    def run(values, *args, **kwargs):
        calls.append(values)
        poses = [
            value
            | {
                "stereochemistry": validate_ligand_state(
                    value.get("source_smiles", value["smiles"]), value["smiles"]
                ),
                "score_kcal_mol": -5.0 - i,
                "interactions": [],
                "diagram_html": "",
                "sdf": "",
                "pdbqt": "",
                "prepared_pdbqt": "",
            }
            for i, value in enumerate(values)
        ]
        return {
            "ligands": poses,
            "manifest": {"warnings": []},
            "protein_pdb": "",
            "receptor_pdbqt": "",
            "interaction_changes": [],
        }

    monkeypatch.setattr(docking_ui, "run_pair_docking", run)
    app = panel(monkeypatch)
    assert len(app.warning) == 1
    assert app.button(key="stereo_run").disabled
    assert not calls
    next(b for b in app.button if b.label.startswith("Generate alternatives")).click().run()
    choice = next(s for s in app.selectbox if s.label.startswith("Docking stereoisomer"))
    assert choice.value is None
    assert len(choice.options) == 2
    choice.set_value(choice.options[0]).run()
    accept = next(c for c in app.checkbox if c.label.startswith("I accept ligand A"))
    assert not accept.value
    accept.check().run()
    assert not app.exception
    app.radio(key="stereo_mode").set_value("Upload prepared files").run()
    app.text_input(key="stereo_label").set_value("Synthetic receptor")
    for axis in "xyz":
        app.number_input(key=f"stereo_center_{axis}").set_value(0.0)
    app.checkbox(key="stereo_confirmed").check().run()
    assert not app.button(key="stereo_run").disabled
    app.session_state["stereo_3d"] = False
    app.button(key="stereo_run").click().run()
    assert not app.exception
    assert calls[0][0]["source_smiles"] == PAIR[0]["smiles"]
    assert calls[0][0]["smiles"] != PAIR[0]["smiles"]
    assert calls[0][0]["pactivity"] == 6.0
    assert calls[0][1] == PAIR[1]
    assert any("not assessed" in w.value for w in app.warning)
    assert not any("ranks the experimentally" in i.value for i in app.info)
    assert app.metric[2].label.startswith("Source")
    choice = next(s for s in app.selectbox if s.label.startswith("Docking stereoisomer"))
    choice.set_value(choice.options[1]).run()
    assert not app.metric  # New stereo choice needs a new explicit acceptance.
    assert not next(c for c in app.checkbox if c.label.startswith("I accept ligand A")).value
    assert app.button(key="stereo_run").disabled
    assert app.number_input(key="stereo_center_x").value == 0.0
    assert app.text_input(key="stereo_label").value == "Synthetic receptor"
    assert len(calls) == 1


def test_manual_stereo_is_validated_and_large_enumeration_keeps_manual_controls(monkeypatch):
    app = panel(monkeypatch, [PAIR[0] | {"smiles": "FC(Cl)C(Br)C(O)C(N)C"}, PAIR[1]])
    next(b for b in app.button if b.label.startswith("Generate alternatives")).click().run()
    assert not app.exception
    assert any("browser limit" in w.value for w in app.warning)
    next(r for r in app.radio if r.label.startswith("Stereo input")).set_value(
        "Provide fully specified SMILES"
    ).run()
    manual = next(t for t in app.text_input if t.label.startswith("Fully specified"))
    manual.set_value("CC").run()
    assert not app.exception
    assert any("must retain" in e.value for e in app.error)
    assert app.button(key="stereo_run").disabled


def test_ez_choice_is_offered_without_automatically_assigning_configuration(monkeypatch):
    app = panel(monkeypatch, [PAIR[0] | {"smiles": "CC=CC"}, PAIR[1]])
    next(b for b in app.button if b.label.startswith("Generate alternatives")).click().run()
    choice = next(s for s in app.selectbox if s.label.startswith("Docking stereoisomer"))
    assert len(choice.options) == 2 and choice.value is None
    choice.set_value(choice.options[0]).run()
    assert not app.exception
    assert not next(c for c in app.checkbox if c.label.startswith("I accept ligand A")).value

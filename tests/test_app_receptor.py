"""No uploads, automatic site boxes, explicit actions and stale-input guards."""

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from sarscope import docking_ui, receptor_ui  # noqa: E402

TARGET = {"target_chembl_id": "CHEMBL123", "target_components": [{"accession": "P12345"}]}
ENTRY = {
    "pdb_id": "1ABC",
    "title": "Synthetic receptor",
    "resolution_A": 2.0,
    "method": "X-RAY DIFFRACTION",
    "ligands": "LIG",
    "chain_mutations": {"X": "G12D"},
    "matched_chains": ["X"],
    "protein_chains": ["X"],
}
SITE = {
    "id": "X:10:LIG",
    "name": "LIG",
    "kind": "bound_ligand",
    "pdb": "",
    "center": [5.0, 6.0, 7.0],
    "size": [12.0, 14.0, 16.0],
    "fits_browser_limit": True,
}


def setup_panel(monkeypatch, *, references=True):
    calls = []
    monkeypatch.setattr(docking_ui, "docking_unavailable_reason", lambda: None)

    def search(accession):
        calls.append("search")
        return [ENTRY]

    class Client:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def metadata(self, ids, **kwargs):
            calls.append("metadata")
            return [ENTRY]

        def coordinates(self, identifier):
            calls.append("coordinates")
            return "Synthetic mmCIF"

    def inspect(cif, chains):
        calls.append("inspect")
        return {
            "protein_pdb": "Synthetic protein",
            "references": [SITE] if references else [],
            "provenance": {"excluded_essential_components": []},
        }

    def prepare(protein, provenance, **kwargs):
        calls.append("prepare")
        return {
            "protein_pdb": protein,
            "receptor_pdbqt": "Prepared synthetic receptor",
            "provenance": provenance,
        }

    def predict(protein, **kwargs):
        calls.append("predict")
        return {
            "pockets": [
                SITE
                | {
                    "id": "fpocket_1",
                    "kind": "predicted_pocket",
                    "rank": 1,
                    "score": 0.3,
                    "druggability_score": 0.6,
                    "volume_A3": 200,
                }
            ],
            "engine": {"distribution": "synthetic test"},
            "protein_sha256": "test",
            "method": "synthetic test",
        }

    monkeypatch.setattr(receptor_ui, "search_structures", search)
    monkeypatch.setattr(receptor_ui, "PdbClient", Client)
    monkeypatch.setattr(receptor_ui, "inspect_structure", inspect)
    monkeypatch.setattr(receptor_ui, "prepare_receptor", prepare)
    monkeypatch.setattr(receptor_ui, "predict_pockets", predict)
    monkeypatch.setattr(receptor_ui, "_preview", lambda *args: None)
    script = (
        "from sarscope.docking_ui import show_cliff_docking\n"
        "pair=[{'molecule_id':'A','smiles':'CC','pactivity':6.},"
        "{'molecule_id':'B','smiles':'CCC','pactivity':8.}]\n"
        f"show_cliff_docking(pair,key='auto',target={TARGET!r})"
    )
    return AppTest.from_string(script).run(), calls


def fetch_structure(app):
    app.checkbox(key="auto_enabled").check().run()
    assert not app.get("file_uploader")
    app.button(key="auto_search").click().run()
    app.selectbox(key="auto_candidate").set_value("1ABC").run()
    app.button(key="auto_fetch").click().run()
    assert not app.exception


def review_and_prepare(app):
    review = next(c for c in app.checkbox if c.label.startswith("I reviewed"))
    review.check().run()
    app.button(key="auto_prepare").click().run()
    assert not app.exception


def test_bound_reference_box_and_preparation_are_automatic_but_button_gated(monkeypatch):
    app, calls = setup_panel(monkeypatch)
    assert not calls
    fetch_structure(app)
    assert calls == ["search", "metadata", "coordinates", "inspect"]
    assert app.selectbox(key="auto_site").value == "X:10:LIG"
    assert app.button(key="auto_prepare").disabled
    assert app.button(key="auto_run").disabled
    review_and_prepare(app)
    assert calls[-1] == "prepare"
    assert "predict" not in calls
    for axis, center, size in zip("xyz", SITE["center"], SITE["size"], strict=True):
        assert app.number_input(key=f"auto_center_{axis}").value == center
        assert app.number_input(key=f"auto_size_{axis}").value == size
    assert not any(c.label.startswith("Docking ranks") for c in app.info)
    app.text_input(key="auto_pdb_id").set_value("2XYZ").run()
    assert app.button(key="auto_run").disabled  # Old preparation cannot dock a new requested PDB.
    assert calls.count("prepare") == calls.count("inspect") == 1


def test_apo_structure_offers_explicit_pocket_prediction_and_records_its_origin(monkeypatch):
    app, calls = setup_panel(monkeypatch, references=False)
    fetch_structure(app)
    assert "predict" not in calls
    app.button(key="auto_predict").click().run()
    assert not app.exception
    assert calls[-1] == "predict"
    assert any("not validated binding sites" in w.value for w in app.warning)
    app.selectbox(key="auto_site").set_value("fpocket_1").run()
    review_and_prepare(app)
    _, result = app.session_state["auto_prepared"]
    assert result["provenance"]["site"]["kind"] == "predicted_pocket"
    assert result["provenance"]["pocket_engine"]["method"] == "synthetic test"
    assert app.number_input(key="auto_center_x").value == 5.0


def test_wrong_target_structure_is_rejected_before_coordinate_download(monkeypatch):
    app, calls = setup_panel(monkeypatch)
    original = ENTRY["matched_chains"]
    monkeypatch.setitem(ENTRY, "matched_chains", [])
    app.checkbox(key="auto_enabled").check().run()
    app.text_input(key="auto_pdb_id").set_value("1ABC").run()
    app.button(key="auto_fetch").click().run()
    assert not app.exception
    assert "coordinates" not in calls
    assert any("no chain mapped" in e.value for e in app.error)
    assert original == ["X"]


def test_reference_box_overlapping_a_removed_cofactor_stops_before_preparation(monkeypatch):
    app, calls = setup_panel(monkeypatch)
    previous = receptor_ui.inspect_structure

    def with_cofactor(cif, chains):
        structure = previous(cif, chains)
        structure["provenance"]["excluded_essential_components"] = [
            {"id": "X:5:ZN", "coordinates": [SITE["center"]]}
        ]
        return structure

    monkeypatch.setattr(receptor_ui, "inspect_structure", with_cofactor)
    fetch_structure(app)
    assert any("Excluded metal/cofactor" in e.value for e in app.error)
    assert not any(b.label == "Prepare receptor · Meeko" for b in app.button)
    assert "prepare" not in calls

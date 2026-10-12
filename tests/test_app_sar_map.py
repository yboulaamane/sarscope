"""SAR controls recompute scoped evidence and invalidate old maps."""

import io
import json
import zipfile

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

SCRIPT = """
import pandas as pd
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.sar_ui import show_sar_map
from sarscope.params import ClassScheme
table=add_scaffolds(pd.DataFrame([
 {"molecule_id":"M1","smiles":"Fc1ccccc1","pactivity":8.,"activity_class":"potent","group":1},
 {"molecule_id":"M2","smiles":"Cc1ccccc1","pactivity":6.,"activity_class":"intermediate","group":2},
 {"molecule_id":"M3","smiles":"Clc1ccccc1","pactivity":7.,"activity_class":"active","group":1},
]))
sar=decompose_scaffold(table,"c1ccccc1",min_substituent_count=1)
evidence=pd.DataFrame([
 {"molecule_id":row.molecule_id,"pactivity":row.pactivity,"record_id":str(i),
  "assay_chembl_id":"ASSAY","standard_type":"IC50","standard_relation":"=",
  "assay_variant_mutation":None,"document_year":2020,"source_origin":"Literature"}
 for i,row in enumerate(table.itertuples())
])
show_sar_map([sar],table,evidence,classes=ClassScheme())
"""


def test_map_is_opt_in_and_filter_changes_require_rebuilding():
    app = AppTest.from_string(SCRIPT, default_timeout=15).run()
    assert not app.exception
    assert "sar_map_result" not in app.session_state
    next(
        button for button in app.button if button.label == "Build interactive SAR map"
    ).click().run()
    assert not app.exception
    assert app.session_state["sar_map_result"][1].series.n_molecules == 3
    assert any(item.label == "Inspect transformation" for item in app.selectbox)
    next(item for item in app.multiselect if item.label == "SAR endpoint").set_value([]).run()
    assert not app.exception
    assert any("Rebuild the map" in item.value for item in app.info)
    next(button for button in app.button if button.label == "Update SAR map").click().run()
    assert not app.exception
    assert any("No compounds remain" in item.value for item in app.info)


def test_evidence_archive_contains_scope_measurements_and_structures():
    from sarscope.sar_ui import sar_map_archive

    app = AppTest.from_string(SCRIPT, default_timeout=15).run()
    next(
        button for button in app.button if button.label == "Build interactive SAR map"
    ).click().run()
    result = app.session_state["sar_map_result"][1]
    evidence = result.pairs.copy()
    structures = result.series.members.assign(smiles="c1ccccc1")
    archive = sar_map_archive(
        result,
        {"filters": {"standard_type": ["IC50"]}},
        "<svg/>",
        evidence=evidence,
        structures=structures,
    )
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        assert bundle.testzip() is None
        scope = json.loads(bundle.read("scope.json"))
        assert scope["filters"]["standard_type"] == ["IC50"]
        assert set(scope["retained_record_ids"]) == {"0", "1", "2"}
        assert "retained_measurements.csv" in bundle.namelist()
        assert "series_structures.csv" in bundle.namelist()
        assert bundle.read("sar_map.svg") == b"<svg/>"

import json

from sarscope import __version__
from sarscope.params import RunParams
from sarscope.provenance import chembl_source, collect, table_source


def test_collect_is_json_serialisable_and_complete():
    source = chembl_source(
        {"target_chembl_id": "CHEMBL5145", "pref_name": "B-raf", "organism": "Homo sapiens"},
        "ChEMBL_37",
        11017,
    )
    record = json.loads(json.dumps(collect(RunParams(), source)))
    assert record["sarscope"] == __version__
    assert record["source"]["release"] == "ChEMBL_37"
    assert record["packages"]["rdkit"] is not None
    assert record["packages"]["sorbent"] is not None
    assert record["params"]["classes"]["floor_label"] == "inactive"


def test_table_source_hashes_the_file(tmp_path):
    path = tmp_path / "x.csv"
    path.write_bytes(b"abc")
    digest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert table_source(path)["sha256"] == digest

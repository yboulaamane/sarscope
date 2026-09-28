import pandas as pd
import pytest

from sarscope.curate import MEASUREMENT_COLUMNS
from sarscope.sources.table import read_activity_table


def test_reads_p_values(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("molecule_id,smiles,pactivity\nm1,CCO,6.5\nm2,c1ccccc1,7.25\n")
    frame = read_activity_table(path)
    assert tuple(frame.columns) == MEASUREMENT_COLUMNS
    assert frame["pactivity"].tolist() == [6.5, 7.25]
    assert frame["record_id"].tolist() == ["a.csv:2", "a.csv:3"]


def test_converts_concentrations_and_custom_columns(tmp_path):
    path = tmp_path / "b.tsv"
    path.write_text("id\tSMILES\tIC50_nM\nx\tCCO\t100\ny\tCCN\t1000\n")
    frame = read_activity_table(
        path, id_col="id", smiles_col="SMILES", value_col="IC50_nM", unit="nM"
    )
    assert frame["pactivity"].tolist() == pytest.approx([7.0, 6.0])
    assert frame["molecule_id"].tolist() == ["x", "y"]


def test_drops_incomplete_rows_and_keeps_line_numbers(tmp_path):
    path = tmp_path / "c.csv"
    path.write_text("molecule_id,smiles,pactivity\nm1,CCO,6\nm2,,7\nm3,CCC,\nm4,CCN,5\n")
    frame = read_activity_table(path)
    assert frame["molecule_id"].tolist() == ["m1", "m4"]
    assert frame["record_id"].tolist() == ["c.csv:2", "c.csv:5"]


def test_missing_column_names_what_was_found(tmp_path):
    path = tmp_path / "d.csv"
    path.write_text("id,smiles,value\n1,CCO,5\n")
    with pytest.raises(ValueError, match=r"missing column\(s\) \['molecule_id', 'pactivity'\]"):
        read_activity_table(path)


def test_non_positive_concentration_is_an_error(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("molecule_id,smiles,ic50\nm1,CCO,0\n")
    with pytest.raises(ValueError, match="positive"):
        read_activity_table(path, value_col="ic50", unit="nM")


def test_returns_a_plain_frame(tmp_path):
    path = tmp_path / "f.csv"
    path.write_text("molecule_id,smiles,pactivity\nm1,CCO,6\n")
    assert isinstance(read_activity_table(path), pd.DataFrame)

import pytest

from sarscope import __main__ as cli
from sarscope import __version__


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_run_needs_exactly_one_source(tmp_path, capsys):
    assert cli.main(["run", "--out", str(tmp_path)]) == 2
    assert "either a target ID or --input" in capsys.readouterr().err


def test_build_params_maps_flags(tmp_path):
    args = cli.parser().parse_args(
        [
            "run", "CHEMBL5145", "--out", str(tmp_path), "--variant", "V600E",
            "--keep-censored", "--split", "random", "--algorithms", "extra_trees",
            "--max-year", "2022", "--no-leakage-audit",
        ]
    )  # fmt: skip
    params = cli.build_params(args)
    assert params.curation.variant == "V600E"
    assert ">" in params.curation.relations
    assert params.curation.max_document_year == 2022
    assert params.model.split == "random"
    assert params.model.algorithms == ("extra_trees",)
    assert params.model.leakage_audit is False


def test_build_params_defaults_are_wild_type_and_exact_values(tmp_path):
    args = cli.parser().parse_args(["run", "CHEMBL5145", "--out", str(tmp_path)])
    params = cli.build_params(args)
    assert params.curation.variant is None
    assert params.curation.relations == ("=",)


def test_cache_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SARSCOPE_CACHE", str(tmp_path))
    assert cli.default_cache_dir() == tmp_path


def test_input_column_flags_reach_the_reader(tmp_path, monkeypatch):
    """A user's CSV in nM must be readable without editing it first."""
    from sarscope import pipeline

    captured = {}

    def fake_run_table(path, params, **kwargs):
        captured.update(path=path, **kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(pipeline, "run_table", fake_run_table)
    csv = tmp_path / "mine.csv"
    csv.write_text("id,SMILES,IC50_nM\nm1,CCO,100\n")
    with pytest.raises(SystemExit):
        cli.main(
            [
                "run", "--input", str(csv), "--out", str(tmp_path / "out"),
                "--input-id-col", "id", "--input-smiles-col", "SMILES",
                "--input-value-col", "IC50_nM", "--input-unit", "nM",
            ]
        )  # fmt: skip
    assert captured["id_col"] == "id"
    assert captured["smiles_col"] == "SMILES"
    assert captured["value_col"] == "IC50_nM"
    assert captured["unit"] == "nM"


def test_input_defaults_are_the_p_scale(tmp_path):
    args = cli.parser().parse_args(["run", "--input", "x.csv", "--out", str(tmp_path)])
    assert (args.input_id_col, args.input_smiles_col) == ("molecule_id", "smiles")
    assert (args.input_value_col, args.input_unit) == ("pactivity", "p")

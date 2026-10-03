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
            "--source-ids", "7", "37",
        ]
    )  # fmt: skip
    params = cli.build_params(args)
    assert params.curation.variant == "V600E"
    assert ">" in params.curation.relations
    assert params.curation.max_document_year == 2022
    assert params.curation.source_ids == (7, 37)
    assert params.model.split == "random"
    assert params.model.algorithms == ("extra_trees",)
    assert params.model.leakage_audit is False


def test_build_params_defaults_are_unannotated_and_exact_values(tmp_path):
    args = cli.parser().parse_args(["run", "CHEMBL5145", "--out", str(tmp_path)])
    params = cli.build_params(args)
    assert params.curation.variant is None
    assert params.curation.relations == ("=",)


def test_cli_does_not_conflate_explicit_wild_type_with_missing_annotation(tmp_path):
    args = cli.parser().parse_args(
        ["run", "CHEMBL5145", "--out", str(tmp_path), "--variant", "wild-type"]
    )
    assert cli.build_params(args).curation.variant == "wild-type"


def test_source_holdout_cli_flags(tmp_path):
    args = cli.parser().parse_args(
        ["run", "CHEMBL5145", "--out", str(tmp_path), "--split", "source", "--source-test-id", "37"]
    )
    params = cli.build_params(args)
    assert params.model.split == "source"
    assert params.model.source_test_id == 37


def test_separate_qualitative_and_frozen_benchmark_commands(tmp_path):
    screen = cli.parser().parse_args(["screen", "1000", "--out", str(tmp_path / "screen")])
    assert screen.aid == 1000
    assert screen.snapshot is None
    freeze = cli.parser().parse_args(["benchmark-freeze", "--out", str(tmp_path / "frozen")])
    assert freeze.out == tmp_path / "frozen"
    bench = cli.parser().parse_args(
        ["benchmark-run", "--frozen", str(tmp_path / "frozen"), "--out", str(tmp_path / "result")]
    )
    assert bench.frozen == tmp_path / "frozen"
    assert bench.validation == "scaffold"
    origin = cli.parser().parse_args(
        [
            "benchmark-run", "--frozen", str(tmp_path / "frozen"),
            "--out", str(tmp_path / "origin"), "--validation", "origin",
        ]
    )  # fmt: skip
    assert origin.validation == "origin"


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

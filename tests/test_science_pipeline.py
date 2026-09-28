"""Specification for sarscope.pipeline and sarscope.report, offline, end to end."""

import json

import pandas as pd
import pytest

from sarscope.analysis.descriptors import PAPER_DESCRIPTORS
from sarscope.curate import CURATED_COLUMNS, CurationResult, CurationStep
from sarscope.params import LandscapeParams, ModelParams, RunParams
from sarscope.pipeline import analyse
from sarscope.report import write_report

pytestmark = pytest.mark.science

CORES = [
    "c1ccc({})cc1",
    "c1ccnc({})c1",
    "c1ccc2[nH]c({})cc2c1",
    "c1ccc2cc({})ccc2c1",
    "C1CCC({})CC1",
    "c1cc({})sc1",
    "c1cnc({})nc1",
    "O=C1CCC({})N1",
]
SUBSTITUENTS = ["C", "CC", "CCC", "CO", "CN", "Cl", "F", "OC", "C(F)(F)F", "C#N"]


def label(p: float) -> str:
    return "potent" if p >= 8 else "active" if p >= 7 else "intermediate" if p >= 6 else "inactive"


@pytest.fixture
def curation() -> CurationResult:
    rows = []
    for i, (core, sub) in enumerate((c, s) for c in CORES for s in SUBSTITUENTS):
        p = 4.5 + ((i * 37) % 50) / 10
        rows.append(
            {
                "molecule_id": f"M{i}",
                "merged_ids": f"M{i}",
                "smiles": core.format(sub),
                "inchikey": None,
                "pactivity": p,
                "n_measurements": 1,
                "pactivity_range": 0.0,
                "activity_class": label(p),
                "group": 1 if p >= 7 else 2,
            }
        )
    table = pd.DataFrame(rows, columns=list(CURATED_COLUMNS))
    return CurationResult(table=table, steps=[CurationStep("input", 80, 80, 80)])


def params(**model) -> RunParams:
    defaults = {"algorithms": ("extra_trees",), "cv_folds": 3, "leakage_audit": False}
    return RunParams(
        landscape=LandscapeParams(fingerprints=("ecfp4", "maccs")),
        model=ModelParams(**{**defaults, **model}),
    )


def test_analyse_fills_every_result(curation):
    results = analyse(curation, params())
    assert set(PAPER_DESCRIPTORS) <= set(results.table.columns)
    assert {"murcko", "skeleton"} <= set(results.table.columns)
    assert len(results.table) == 80
    assert set(results.landscapes) == {"ecfp4", "maccs"}
    assert results.models is not None and results.domain is not None
    assert 0.0 <= results.domain.coverage <= 1.0
    assert results.skipped == {}


def test_impossible_modelling_is_skipped_not_fatal(curation):
    results = analyse(curation, params(cv_folds=50))
    assert results.models is None and results.domain is None
    assert "model" in results.skipped
    assert not results.diversity.empty  # everything else still ran


def test_report_folder(curation, tmp_path):
    results = analyse(curation, params())
    out = tmp_path / "report"
    html = write_report(results, out)
    assert html == out / "report.html"
    for name in [
        "provenance.json",
        "tables/curation_log.csv",
        "tables/curated_dataset.csv",
        "tables/table2_descriptor_profile.csv",
        "tables/table3_pca_loadings.csv",
        "tables/table4_scaffold_diversity.csv",
        "tables/scaffold_enrichment.csv",
        "tables/cliffs_ecfp4.csv",
        "tables/cliffs_maccs.csv",
        "tables/table6_models.csv",
        "figures/fig5_pca.png",
        "figures/fig8_sas_maccs.png",
    ]:
        assert (out / name).is_file(), name
    text = html.read_text()
    assert "data:image/png;base64," in text
    assert 'src="figures/' not in text  # self-contained
    assert json.loads((out / "provenance.json").read_text())["params"]["model"]["cv_folds"] == 3


def test_report_refuses_an_unrelated_non_empty_folder(curation, tmp_path):
    (tmp_path / "thesis.docx").write_text("do not touch")
    with pytest.raises(FileExistsError):
        write_report(analyse(curation, params()), tmp_path)


def test_report_overwrites_its_own_previous_output(curation, tmp_path):
    results = analyse(curation, params())
    write_report(results, tmp_path / "r")
    write_report(results, tmp_path / "r")

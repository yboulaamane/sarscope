from pathlib import Path

import pandas as pd
import pytest

from sarscope.ml_selection import select_ml_table


def test_drop_intermediate_is_ml_only_and_preserves_other_labels():
    curated = pd.DataFrame(
        {
            "molecule_id": ["p", "a", "m", "i"],
            "activity_class": ["potent", "active", "intermediate", "inactive"],
            "pactivity": [8.5, 7.4, 6.5, 5.4],
        },
        index=[10, 11, 12, 13],
    )
    selected = select_ml_table(curated, drop_intermediate=True)
    assert selected["activity_class"].tolist() == ["potent", "active", "inactive"]
    assert selected["pactivity"].tolist() == [8.5, 7.4, 5.4]
    assert selected.index.tolist() == [0, 1, 2]
    assert curated["activity_class"].tolist() == ["potent", "active", "intermediate", "inactive"]


def test_default_ml_selection_retains_all_compounds():
    curated = pd.DataFrame({"activity_class": ["active", "intermediate"]})
    assert len(select_ml_table(curated)) == 2


def test_drop_intermediate_refuses_empty_ml_dataset():
    curated = pd.DataFrame({"activity_class": ["intermediate"]})
    with pytest.raises(ValueError, match="leaves no compounds"):
        select_ml_table(curated, drop_intermediate=True)


def test_browser_ml_stage_applies_selection_without_changing_upstream_table(monkeypatch):
    from sarscope.curate import CurationResult
    from sarscope.params import RunParams

    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from streamlit_app import ScaffoldStage, run_ml_stage

    curated = pd.DataFrame(
        {
            "molecule_id": ["p", "a", "m", "i"],
            "activity_class": ["potent", "active", "intermediate", "inactive"],
            "murcko": ["P", "A", "M", "I"],
        }
    )
    scaffold = ScaffoldStage(curated, pd.DataFrame(), pd.DataFrame())
    stage = run_ml_stage(
        scaffold,
        CurationResult(curated),
        RunParams(),
        classification=True,
        regression=False,
        landscape=None,
        drop_intermediate=True,
    )
    assert stage.drop_intermediate
    assert stage.table["activity_class"].tolist() == ["potent", "active", "inactive"]
    assert curated["activity_class"].tolist() == [
        "potent", "active", "intermediate", "inactive"
    ]
    assert stage.skipped  # tiny fixture is intentionally too small to fit ML

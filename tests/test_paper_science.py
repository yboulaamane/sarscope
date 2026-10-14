"""Scientific-paper diagnostics must not treat assay or stereo mixtures as controls."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PAPER = Path(__file__).resolve().parents[1] / "paper"
spec = importlib.util.spec_from_file_location(
    "sar_paper_analysis", PAPER / "analyse_sar_evidence.py"
)
assert spec is not None and spec.loader is not None
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_multiple_testing_adjustment_preserves_order_and_bounds():
    np.testing.assert_allclose(analysis.bh_adjust(np.array([0.04, 0.01, 0.03])), [0.04, 0.03, 0.04])
    with pytest.raises(ValueError):
        analysis.bh_adjust(np.array([np.nan]))


def test_direction_rule_requires_magnitude_consistency_and_no_large_conflict():
    assert analysis.directional(np.array([1, 1, 1]), 0.5).item()
    assert not analysis.directional(np.array([1, 1, -1]), 0.5).item()
    assert not analysis.directional(np.array([0.1, 0.1, 0.1]), 0.5).item()


def test_compound_omission_removes_every_incident_pair():
    pairs = pd.DataFrame(
        {"id_from": ["a", "a", "b"], "id_to": ["b", "c", "c"], "delta_pactivity": [1.0, 3.0, 2.0]}
    )
    result = analysis.leave_one_compound_out(pairs)
    assert result["loco_valid_omissions"] == 3
    assert result["loco_median_min"] == 1
    assert result["loco_median_max"] == 3
    assert result["loco_sign_retention"] == 1


def test_stereo_audit_separates_enantiomers_and_unassigned_core():
    directory = PAPER / "results/drd2"
    members = pd.read_csv(directory / "series_members.csv")
    molecules = pd.read_csv(directory / "molecules.csv")
    scope = json.loads((directory / "scope.json").read_text())
    selected, audit, core = analysis.stereo_scope(members, molecules, scope)
    assert len(selected) == 23
    assert audit.isomeric_core.nunique() == 3
    assert set(audit[~audit.retained].molecule_id) == {
        "CHEMBL125455",
        "CHEMBL126493",
        "CHEMBL2109988",
    }
    assert core.count("@") > 0
    cycles = analysis.double_changes(selected)
    quartet = cycles[
        (cycles.position_a == "R2")
        & (cycles.position_b == "R4")
        & cycles.a0.isin(["H", "Cl[*:2]"])
        & cycles.a1.isin(["H", "Cl[*:2]"])
        & cycles.b0.isin(["H", "Cl[*:4]"])
        & cycles.b1.isin(["H", "Cl[*:4]"])
    ]
    assert len(quartet) == 1
    assert quartet.iloc[0].delta_delta_pIC50 == pytest.approx(-0.3689947652)


def test_complete_cycles_never_silently_collapse_ambiguous_cells():
    table = pd.DataFrame(
        [
            {"molecule_id": f"M{i}", "R1": a, "R2": b, "pactivity": value}
            for i, (a, b, value) in enumerate(
                [("A", "X", 1), ("A", "Y", 2), ("B", "X", 3), ("B", "Y", 4)]
            )
        ]
    )
    assert len(analysis.double_changes(table)) == 1
    duplicate = table.iloc[[0]].copy()
    duplicate.molecule_id = "OTHER_STEREO_STATE"
    assert analysis.double_changes(pd.concat([table, duplicate])).empty

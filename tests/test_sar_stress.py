"""SAR claims are challenged with controlled comparisons, not confident prose."""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from rdkit import Chem

from sarscope.analysis.rgroups import ScaffoldSar, _substituent_stats, decompose_scaffold
from sarscope.analysis.sar_map import build_sar_map
from sarscope.analysis.sar_stress import assemble_analogue, stress_test_sar


def cohort(cells=None):
    if cells is None:
        cells = [
            ("a", "C[*:1]", "C[*:2]", 6),
            ("b", "F[*:1]", "C[*:2]", 7),
            ("c", "C[*:1]", "CC[*:2]", 6),
            ("d", "F[*:1]", "CC[*:2]", 7),
            ("e", "C[*:1]", "CCC[*:2]", 6),
            ("f", "F[*:1]", "CCC[*:2]", 7),
        ]
    members = pd.DataFrame(cells, columns=["molecule_id", "R1", "R2", "pactivity"])
    members["activity_class"] = "active"
    series = ScaffoldSar(
        "c1ccccc1",
        "c1cc([*:1])ccc1[*:2]",
        len(members),
        0,
        members,
        _substituent_stats(members, ["R1", "R2"], 1),
    )
    evidence = members[["molecule_id", "pactivity"]].copy()
    evidence["record_id"] = evidence.molecule_id
    evidence["assay_chembl_id"] = "ASSAY"
    evidence["standard_type"] = "IC50"
    evidence["standard_relation"] = "="
    evidence["assay_variant_mutation"] = None
    evidence["document_chembl_id"] = evidence.molecule_id.map(
        {"a": "D1", "b": "D1", "c": "D2", "d": "D2", "e": "D3", "f": "D3"}
    )
    return series, evidence


def analyse(series, evidence, **kwargs):
    result = build_sar_map(series, evidence)
    claims = result.transformations.query("position == 'R1'")
    shared = claims[claims.evidence_tier == "shared assay"]
    claim = (shared if not shared.empty else claims).iloc[0]
    return result, stress_test_sar(result, claim, evidence, **kwargs)


def test_omissions_remove_all_incident_edges_and_do_not_pool_only_assay():
    series, evidence = cohort()
    _, stress = analyse(series, evidence)
    assert stress.verdict == "Direction retained in the tested subsets"
    for omitted in ["a", "b", "c", "d", "e", "f"]:
        check = stress.checks[
            (stress.checks.test == "Leave one compound out") & (stress.checks.omitted == omitted)
        ].iloc[0]
        assert check.pairs == 2
        assert check.median_delta == 1
    check = stress.checks.query("test == 'Leave one publication out' and omitted == 'D1'").iloc[0]
    assert check.pairs == 2
    assert stress.checks.query("test == 'Leave one assay out'").iloc[0].pairs == 0
    assert any("no matched pairs" in note for note in stress.limitations)


def test_compound_reuse_is_not_independent_support():
    series, evidence = cohort(
        [("a", "C[*:1]", "C[*:2]", 6), ("b", "F[*:1]", "C[*:2]", 7), ("c", "F[*:1]", "C[*:2]", 8)]
    )
    _, stress = analyse(series, evidence)
    assert stress.checks.iloc[0].pairs == 2
    assert stress.checks.iloc[0].disjoint_pairs == 1
    assert stress.checks.query("omitted == 'a'").iloc[0].pairs == 0
    assert stress.verdict == "Too sparse to assess transferability"


def test_publication_removal_recomputes_raw_measurement_medians():
    series, evidence = cohort([("a", "C[*:1]", "C[*:2]", 6), ("b", "F[*:1]", "C[*:2]", 7)])
    extra = evidence.iloc[[0]].copy()
    extra["record_id"], extra["document_chembl_id"], extra["pactivity"] = "a2", "D2", 10
    evidence = pd.concat([evidence, extra], ignore_index=True)
    _, stress = analyse(series, evidence)
    assert stress.checks.iloc[0].median_delta == -1
    check = stress.checks.query("test == 'Leave one publication out' and omitted == 'D2'").iloc[0]
    assert check.median_delta == 1
    assert not check.retains_direction


def test_copied_values_do_not_weight_medians_or_multiply_pair_support():
    series, evidence = cohort()
    extra = evidence.iloc[[0]].copy()
    extra["pactivity"], extra["record_id"] = 10, "new"
    copied = pd.concat([evidence.iloc[[0]]] * 20, ignore_index=True)
    result, _ = analyse(series, pd.concat([evidence, extra, copied], ignore_index=True))
    pair = result.pairs.query("position == 'R1' and id_from == 'a'").iloc[0]
    assert pair.pactivity_from == 8
    assert set(pair.record_ids_from.split(";")) == {"a", "new"}
    assert result.transformations.query("position == 'R1'").iloc[0].supporting_pairs == 3


def test_equal_rounded_values_with_distinct_record_ids_are_not_assumed_duplicate():
    series, evidence = cohort()
    extra = evidence.iloc[[0]].copy()
    extra["pactivity"], extra["record_id"] = 10, "new"
    rounded = evidence.iloc[[0]].copy()
    rounded["record_id"] = "distinct_experiment"
    result, _ = analyse(series, pd.concat([evidence, extra, rounded], ignore_index=True))
    pair = result.pairs.query("position == 'R1' and id_from == 'a'").iloc[0]
    assert pair.pactivity_from == 6
    assert result.transformations.query("position == 'R1'").iloc[0].supporting_pairs == 3


def test_opposing_assays_produce_retest_suggestion_not_potency_prediction():
    series, evidence = cohort()
    other = evidence.copy()
    other["assay_chembl_id"] = "OTHER"
    other["record_id"] = "other_" + other.record_id
    other["pactivity"] = other.molecule_id.map({"a": 8, "b": 6, "c": 8, "d": 6, "e": 8, "f": 6})
    result, stress = analyse(series, pd.concat([evidence, other], ignore_index=True))
    assert stress.verdict == "Mixed or below the selected effect threshold"
    assert stress.checks.iloc[0].pairs == 3
    assert stress.checks.iloc[0].assays == 2
    assert stress.suggestions.iloc[0].action == "Retest conflicting pair"
    assert "other_" in stress.suggestions.iloc[0].record_ids
    assert stress.suggestions.iloc[0].proposed_smiles == ""
    assert result.transformations.query("position == 'R1'").iloc[0].evidence_strength == "mixed"


def test_missing_cycle_suggests_a_pattern_with_three_observed_corners():
    series, evidence = cohort(
        [("a", "C[*:1]", "C[*:2]", 6), ("b", "F[*:1]", "C[*:2]", 8), ("c", "C[*:1]", "CC[*:2]", 7)]
    )
    _, stress = analyse(series, evidence)
    assert stress.cycles.empty
    suggestion = stress.suggestions.iloc[0]
    assert suggestion.action == "Complete missing analogue"
    assert json.loads(suggestion.proposed_pattern) == {"R1": "F[*:1]", "R2": "CC[*:2]"}
    assert suggestion.supporting_ids == "a;b;c"
    assert set(suggestion.record_ids.split(";")) == {"a", "b", "c"}
    assert suggestion.proposed_smiles == ""  # Legacy input: no invented chemistry.


def test_existing_but_unmeasured_fourth_corner_is_not_called_a_new_molecule():
    series, evidence = cohort(
        [
            ("a", "C[*:1]", "C[*:2]", 6),
            ("b", "F[*:1]", "C[*:2]", 8),
            ("c", "C[*:1]", "CC[*:2]", 7),
            ("d", "F[*:1]", "CC[*:2]", 6),
        ]
    )
    evidence.loc[evidence.molecule_id == "d", "assay_chembl_id"] = "OTHER"
    _, stress = analyse(series, evidence)
    assert stress.cycles.empty
    assert stress.suggestions.iloc[0].action == "Measure existing analogue"
    assert stress.suggestions.iloc[0].existing_ids == "d"


def test_complete_cycle_is_same_assay_and_reports_descriptive_interaction():
    series, evidence = cohort(
        [
            ("a", "C[*:1]", "C[*:2]", 6),
            ("b", "F[*:1]", "C[*:2]", 8),
            ("c", "C[*:1]", "CC[*:2]", 7),
            ("d", "F[*:1]", "CC[*:2]", 6),
        ]
    )
    _, stress = analyse(series, evidence)
    assert len(stress.cycles) == 1
    assert stress.cycles.iloc[0].interaction_delta == 3
    assert stress.cycles.iloc[0].interaction_delta == (
        stress.cycles.iloc[0].delta_second - stress.cycles.iloc[0].delta_first
    )
    assert stress.cycles.iloc[0].variant == "Unreported"
    evidence.loc[evidence.molecule_id == "c", "assay_variant_mutation"] = "V600E"
    _, stress = analyse(series, evidence)
    assert stress.cycles.empty
    assert not (stress.suggestions.action == "Complete missing analogue").any()


def test_ambiguous_cycle_cells_are_not_silently_averaged():
    series, evidence = cohort(
        [
            ("a", "C[*:1]", "C[*:2]", 6),
            ("b", "F[*:1]", "C[*:2]", 8),
            ("c", "C[*:1]", "CC[*:2]", 7),
            ("d", "F[*:1]", "CC[*:2]", 6),
            ("duplicate", "F[*:1]", "CC[*:2]", 9),
        ]
    )
    _, stress = analyse(series, evidence)
    assert stress.cycles.empty
    assert stress.suggestions.empty


def test_threshold_sensitivity_and_bounded_checks_are_explicit():
    series, evidence = cohort()
    _, stress = analyse(series, evidence, max_omissions_per_axis=1, max_cycle_visits=1)
    assert stress.settings["omissions_truncated"]
    assert stress.settings["cycle_search_truncated"]
    assert any("partial sensitivity" in note for note in stress.limitations)
    _, stress = analyse(series, evidence, thresholds=(0.5, 1.5))
    assert stress.thresholds.interpretation.tolist() == ["increase", "mixed / small change"]
    with pytest.raises(ValueError):
        analyse(series, evidence, thresholds=(float("nan"),))


def test_unmatched_evidence_abstains_and_suggests_no_virtual_potency():
    series, evidence = cohort()
    result, stress = analyse(series, evidence.assign(assay_chembl_id=None))
    assert set(result.pairs.evidence_tier) == {"pooled / unmatched"}
    assert stress.verdict == "Insufficient shared-assay evidence"
    assert stress.contexts.empty and stress.suggestions.empty


def test_production_drd2_map_uses_largest_identical_stereo_core_and_audits_exclusions():
    table = pd.read_csv(Path(__file__).resolve().parents[1] / "paper/results/drd2/molecules.csv")
    original = pd.read_csv(
        Path(__file__).resolve().parents[1] / "paper/results/drd2/series_members.csv"
    )
    scaffold = table[table.molecule_id.isin(original.molecule_id)].iloc[0].murcko
    series = decompose_scaffold(table, scaffold, min_substituent_count=1)
    result = build_sar_map(series, pd.DataFrame())
    assert result.series.n_molecules == 23
    assert result.excluded_core_molecules == 3
    assert result.series.members.core_smiles.nunique() == 1
    assert not {"CHEMBL125455", "CHEMBL126493", "CHEMBL2109988"} & set(
        result.series.members.molecule_id
    )
    assert result.series.core == result.series.members.iloc[0].core_smiles


def test_molzip_proposals_preserve_stereo_and_require_original_round_trip():
    member = pd.Series(
        {
            "core_smiles": "C[C@H](N)[*:1]",
            "rgroup_fragments": json.dumps({"R1": "C[*:1]"}),
            "molecule_smiles": "C[C@H](N)C",
        }
    )
    # Original above is achiral (two identical methyls); invalid stereo is not a template.
    assert assemble_analogue(member, "R1", "F[*:1]") == ""
    member["rgroup_fragments"] = json.dumps({"R1": "CC[*:1]"})
    member["molecule_smiles"] = "C[C@H](N)CC"
    proposal = assemble_analogue(member, "R1", "F[*:1]")
    assert proposal
    assert len(Chem.FindMolChiralCenters(Chem.MolFromSmiles(proposal), includeUnassigned=True)) == 1
    assert "@" in proposal
    member["molecule_smiles"] = "C[C@@H](N)CC"
    assert assemble_analogue(member, "R1", "F[*:1]") == ""


def test_stress_archive_is_traceable_and_hashes_match():
    from sarscope.sar_ui import sar_stress_archive

    series, evidence = cohort()
    result, stress = analyse(series, evidence)
    with zipfile.ZipFile(io.BytesIO(sar_stress_archive(stress, result, evidence))) as archive:
        assert archive.testzip() is None
        metadata = json.loads(archive.read("evidence_card.json"))
        assert metadata["claim"]["position"] == "R1"
        for name, checksum in metadata["sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == checksum

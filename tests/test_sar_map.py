"""Controlled SAR comparisons must retain assay context and sparse evidence."""

import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from sarscope.analysis.rgroups import ScaffoldSar, _substituent_stats
from sarscope.analysis.sar_map import build_sar_map, filter_series
from sarscope.sar_depict import sar_map_svg


def data():
    members = pd.DataFrame(
        [
            {
                "molecule_id": f"M{i}{suffix}",
                "pactivity": pooled,
                "activity_class": "active",
                "R1": fragment,
                "R2": context,
            }
            for i, context in enumerate(["C[*:2]", "CC[*:2]", "CCC[*:2]"])
            for suffix, fragment, pooled in [("a", "C[*:1]", 9.0), ("b", "F[*:1]", 5.0)]
        ]
    )
    series = ScaffoldSar(
        "c1ccccc1",
        "c1cc([*:1])ccc1[*:2]",
        len(members),
        0,
        members,
        _substituent_stats(members, ["R1", "R2"], 1),
    )
    evidence = pd.DataFrame(
        [
            {
                "molecule_id": row.molecule_id,
                "record_id": str(i),
                "pactivity": value,
                "assay_chembl_id": "ASSAY1",
                "standard_type": "IC50",
                "standard_relation": "=",
                "assay_variant_mutation": None,
                "document_year": 2020,
                "source_origin": "literature",
            }
            for i, row in enumerate(members.itertuples())
            for value in [6.0 if row.R1.startswith("C") else 7.0]
        ]
    )
    return series, evidence


def test_position_effect_uses_shared_assay_values_and_holds_other_sites_constant():
    series, evidence = data()
    result = build_sar_map(series, evidence)
    pairs = result.pairs[result.pairs.position == "R1"]
    assert len(pairs) == 3
    assert pairs.delta_pactivity.tolist() == [1.0, 1.0, 1.0]
    row = result.transformations[result.transformations.position == "R1"].iloc[0]
    assert row.median_delta == 1.0
    assert row.disjoint_pairs == 3
    assert row.distinct_contexts == 3
    assert row.evidence_strength == "strong"
    assert row.interpretation == "increase"


def test_endpoints_and_variants_are_never_pooled_in_matched_deltas():
    series, evidence = data()
    ki = evidence.copy()
    ki["standard_type"] = "Ki"
    ki["pactivity"] = ki.molecule_id.map(
        lambda identifier: 9.0 if identifier.endswith("a") else 6.0
    )
    ki["record_id"] = "ki_" + ki.record_id
    result = build_sar_map(series, pd.concat([evidence, ki]))
    rows = result.transformations[result.transformations.position == "R1"]
    assert dict(zip(rows.endpoint, rows.median_delta, strict=True)) == {"IC50": 1.0, "Ki": -3.0}
    mutated = evidence.copy()
    mutated.loc[mutated.molecule_id.str.endswith("b"), "assay_variant_mutation"] = "V600E"
    result = build_sar_map(series, mutated)
    pairs = result.pairs[result.pairs.position == "R1"]
    assert set(pairs.evidence_tier) == {"pooled / unmatched"}
    assert result.transformations.query("position == 'R1'").iloc[0].evidence_strength == "limited"


def test_repeated_assays_do_not_multiply_compound_pair_support():
    series, evidence = data()
    other = evidence.copy()
    other["assay_chembl_id"] = "ASSAY2"
    other["record_id"] = "other_" + other.record_id
    result = build_sar_map(series, pd.concat([evidence, other]))
    summary = result.transformations.query("position == 'R1'").iloc[0]
    assert len(result.pairs.query("position == 'R1'")) == 6
    assert summary.supporting_pairs == 3
    assert summary.disjoint_pairs == 3
    assert summary.assays == 2
    other["pactivity"] = other.molecule_id.map(
        lambda identifier: 9.0 if identifier.endswith("a") else 6.0
    )
    result = build_sar_map(series, pd.concat([evidence, other]))
    assert result.transformations.query("position == 'R1'").iloc[0].evidence_strength == "mixed"


def test_filter_reaggregates_measurements_and_empty_selection_is_empty():
    series, evidence = data()
    scoped, records = filter_series(series, evidence, {"standard_type": ["IC50"]})
    assert scoped.members.query("molecule_id == 'M0a'").iloc[0].pactivity == 6.0
    assert len(records) == 6
    empty, records = filter_series(series, evidence, {"standard_type": []})
    assert empty.members.empty
    assert build_sar_map(empty, records).pairs.empty


def test_comparison_limit_caps_evidence_and_unmatched_values_stay_limited():
    series, evidence = data()
    limited = build_sar_map(series, evidence, max_pairs=1)
    assert limited.pair_limit_reached
    assert len(limited.pairs) == 1
    assert set(limited.transformations.evidence_strength) == {"limited"}
    unmatched = build_sar_map(series, pd.DataFrame())
    assert set(unmatched.transformations.evidence_strength) <= {"limited", "mixed"}


def test_svg_has_real_attachment_sites_and_is_well_formed():
    series, evidence = data()
    result = build_sar_map(series, evidence)
    svg = sar_map_svg(result, selected="R1")
    if svg is None:
        pytest.skip("RDKit drawing unavailable")
    tree = ET.fromstring(svg)
    targets = [
        node.attrib["data-position"] for node in tree.iter() if "data-position" in node.attrib
    ]
    assert targets == ["R1", "R2"]
    assert "3 pairs" in svg
    assert "Invariant core" in svg

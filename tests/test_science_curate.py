"""Specification for sarscope.curate."""

import math

import pandas as pd
import pytest

from sarscope.curate import (
    CURATED_COLUMNS,
    MEASUREMENT_COLUMNS,
    _standardise_one,
    assign_classes,
    curate_chembl,
    curate_table,
    filter_chembl_records,
    source_safe_table,
    standardize_and_aggregate,
    time_safe_table,
)
from sarscope.params import ClassScheme, CurationParams, ModelParams, RunParams

pytestmark = pytest.mark.science

STEP_ORDER = [
    "standard_type",
    "relation",
    "units",
    "assay_type",
    "source_origin",
    "variant",
    "potential_duplicate",
    "validity",
    "bao_format",
    "assay_id",
    "target_confidence",
    "document_year",
    "structure",
    "value",
]


def ids(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["record_id"])


# -- filter_chembl_records ------------------------------------------------


def test_a_clean_record_passes_and_is_converted(make_record):
    frame, _ = filter_chembl_records([make_record(activity_id=7)], CurationParams())
    assert tuple(frame.columns) == MEASUREMENT_COLUMNS
    row = frame.iloc[0]
    assert row["record_id"] == "7"
    assert row["pactivity"] == pytest.approx(7.0)  # 100 nM


def test_molecule_id_is_the_parent(make_record):
    record = make_record(molecule_chembl_id="CHEMBL_SALT", parent_molecule_chembl_id="CHEMBL_P")
    frame, _ = filter_chembl_records([record], CurationParams())
    assert frame["molecule_id"].tolist() == ["CHEMBL_P"]


@pytest.mark.parametrize(
    ("override", "step"),
    [
        ({"standard_type": "Ki"}, "standard_type"),
        ({"standard_relation": ">"}, "relation"),
        ({"standard_relation": None}, "relation"),
        ({"standard_units": "ug.mL-1"}, "units"),
        ({"assay_type": "F"}, "assay_type"),
        ({"assay_variant_mutation": "V600E"}, "variant"),
        ({"potential_duplicate": 1}, "potential_duplicate"),
        ({"data_validity_comment": "Outside typical range"}, "validity"),
        ({"canonical_smiles": None}, "structure"),
        ({"standard_value": None}, "value"),
        ({"standard_value": "0"}, "value"),
    ],
)
def test_each_default_filter_removes_its_record(make_record, override, step):
    records = [make_record(activity_id=1), make_record(activity_id=2, **override)]
    frame, steps = filter_chembl_records(records, CurationParams())
    assert ids(frame) == ["1"]
    removed = {s.name: s.removed for s in steps}
    assert removed[step] == 1
    assert sum(removed.values()) == 1


def test_every_step_is_logged_in_order_and_counts_chain(make_record):
    _, steps = filter_chembl_records([make_record()], CurationParams())
    assert [s.name for s in steps] == STEP_ORDER
    for before, after in zip(steps, steps[1:], strict=False):
        assert after.records_in == before.records_out


def test_variant_selection(make_record):
    records = [
        make_record(activity_id=1),
        make_record(activity_id=2, assay_variant_mutation="V600E"),
        make_record(activity_id=3, assay_variant_mutation="V600K"),
        make_record(activity_id=4, assay_variant_mutation="T790M"),
    ]
    for variant, expected in [
        (None, ["1"]),
        ("V600E", ["2"]),
        ("T790M", ["4"]),
        ("any", ["1", "2", "3", "4"]),
    ]:
        frame, _ = filter_chembl_records(records, CurationParams(variant=variant))
        assert ids(frame) == expected, variant


def test_integrated_origins_can_be_filtered_without_new_downloads(make_record):
    records = [
        make_record(activity_id=1, src_id=1),
        make_record(activity_id=2, src_id=7),
        make_record(activity_id=3, src_id=37),
    ]
    frame, steps = filter_chembl_records(records, CurationParams(source_ids=(7, 37)))
    assert ids(frame) == ["2", "3"]
    assert next(step for step in steps if step.name == "source_origin").removed == 1
    assert frame.attrs["source_id_by_record"] == {"2": 7, "3": 37}
    assert frame.attrs["context_by_record"]["2"]["source_origin"].startswith("PubChem")


def test_empty_origin_filter_explains_why_curation_stopped(make_record):
    params = RunParams(curation=CurationParams(source_ids=(37,)))
    with pytest.raises(ValueError, match="no measurements remain after curation filters"):
        curate_chembl([make_record(src_id=1)], params)


def test_source_holdout_removes_shared_structures_before_aggregation(make_record):
    records = [
        make_record(
            activity_id=1,
            molecule_chembl_id="CHEMBL1",
            parent_molecule_chembl_id="CHEMBL1",
            canonical_smiles="CCO",
            src_id=1,
            standard_value="1000",
        ),
        make_record(
            activity_id=2,
            molecule_chembl_id="CHEMBL2",
            parent_molecule_chembl_id="CHEMBL2",
            canonical_smiles="CCN",
            src_id=1,
        ),
        make_record(
            activity_id=3,
            molecule_chembl_id="CHEMBL3",
            parent_molecule_chembl_id="CHEMBL3",
            canonical_smiles="CCO",
            src_id=7,
            standard_value="1",
        ),
        make_record(
            activity_id=4,
            molecule_chembl_id="CHEMBL4",
            parent_molecule_chembl_id="CHEMBL4",
            canonical_smiles="CCC",
            src_id=7,
        ),
    ]
    params = RunParams(model=ModelParams(split="source", source_test_id=7))
    curated = curate_chembl(records, params)
    safe = source_safe_table(curated, params)
    assert len(safe) == 3
    assert safe["source_test"].tolist() == [False, False, True]
    assert safe.loc[safe["smiles"] == "CCO", "pactivity"].iloc[0] == pytest.approx(6.0)
    assert safe.loc[safe["source_test"], "smiles"].tolist() == ["CCC"]


def test_optional_filters_are_off_by_default_and_work_when_set(make_record):
    records = [
        make_record(activity_id=1),
        make_record(activity_id=2, bao_label="cell-based format", document_year=2024),
        make_record(activity_id=3, document_year=None),
    ]
    frame, _ = filter_chembl_records(records, CurationParams())
    assert ids(frame) == ["1", "2", "3"]

    frame, _ = filter_chembl_records(
        records, CurationParams(bao_formats=("single protein format",))
    )
    assert ids(frame) == ["1", "3"]

    frame, _ = filter_chembl_records(records, CurationParams(max_document_year=2022))
    assert ids(frame) == ["1"]  # no year is dropped when the year filter is on


def test_censored_values_can_be_kept_on_request(make_record):
    records = [make_record(activity_id=1), make_record(activity_id=2, standard_relation=">")]
    frame, _ = filter_chembl_records(records, CurationParams(relations=("=", ">")))
    assert ids(frame) == ["1", "2"]


# -- standardize_and_aggregate --------------------------------------------


def measurements(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [(m, s, p, f"r{i}") for i, (m, s, p) in enumerate(rows)], columns=MEASUREMENT_COLUMNS
    )


def test_salt_and_free_base_merge_into_one_molecule():
    frame = measurements(
        [
            ("CHEMBL10", "c1ccc2[nH]ccc2c1.Cl", 6.0),
            ("CHEMBL9", "c1ccc2[nH]ccc2c1", 7.0),
            ("CHEMBL9", "c1ccc2[nH]ccc2c1", 9.0),
        ]
    )
    molecules, rejected, _ = standardize_and_aggregate(frame, CurationParams())
    assert len(molecules) == 1 and rejected.empty
    row = molecules.iloc[0]
    assert row["molecule_id"] == "CHEMBL9"  # (len, str) order, not plain string order
    assert set(row["merged_ids"].split(";")) == {"CHEMBL9", "CHEMBL10"}
    assert row["smiles"] == "c1ccc2[nH]ccc2c1"
    assert row["pactivity"] == pytest.approx(7.0)  # median of 6, 7, 9
    assert row["n_measurements"] == 3
    assert row["pactivity_range"] == pytest.approx(3.0)


def test_mean_aggregation():
    frame = measurements([("m1", "CCO", 6.0), ("m1", "CCO", 7.0), ("m1", "CCO", 9.0)])
    molecules, _, _ = standardize_and_aggregate(frame, CurationParams(aggregate="mean"))
    assert molecules.iloc[0]["pactivity"] == pytest.approx(22 / 3)


def test_standardization_cache_separates_tautomer_choices():
    _standardise_one.cache_clear()
    first = _standardise_one("CC(C)C(=O)O", True)
    assert _standardise_one("CC(C)C(=O)O", True) == first
    assert _standardise_one.cache_info().hits == 1
    _standardise_one("CC(C)C(=O)O", False)
    assert _standardise_one.cache_info().misses == 2
    _standardise_one.cache_clear()


def test_standardization_progress_and_aggregation_keep_first_source_year():
    frame = measurements(
        [
            ("CHEMBL10", "CCO.Cl", 6.0),
            ("CHEMBL9", "CCO", 7.0),
            ("CHEMBL9", "CCO", 8.0),
        ]
    )
    frame.attrs["document_year_by_record"] = {"r0": 2022, "r1": 2019, "r2": None}
    updates: list[tuple[int, int]] = []
    molecules, _, _ = standardize_and_aggregate(
        frame, CurationParams(), progress=lambda done, total: updates.append((done, total))
    )
    assert updates[-1] == (2, 2)
    assert molecules.iloc[0]["molecule_id"] == "CHEMBL9"
    assert molecules.iloc[0]["pactivity"] == pytest.approx(7.0)
    assert molecules.iloc[0]["document_year"] == 2019


def test_unparseable_structures_are_rejected_with_a_reason():
    frame = measurements([("m1", "CCO", 6.0), ("m2", "not a smiles", 7.0)])
    molecules, rejected, steps = standardize_and_aggregate(frame, CurationParams())
    assert molecules["molecule_id"].tolist() == ["m1"]
    assert rejected["molecule_id"].tolist() == ["m2"]
    assert rejected.iloc[0]["reason"]
    assert next(s for s in steps if s.name == "standardisation").removed == 1


def test_replicate_range_filter():
    frame = measurements([("m1", "CCO", 5.0), ("m1", "CCO", 8.0), ("m2", "CCN", 6.0)])
    molecules, _, steps = standardize_and_aggregate(frame, CurationParams(max_replicate_range=1.0))
    assert molecules["molecule_id"].tolist() == ["m2"]
    assert "replicate_range" in [s.name for s in steps]


# -- assign_classes --------------------------------------------------------


def test_class_boundaries_are_inclusive_lower_bounds():
    p = pd.Series([8.0, 7.999, 7.0, 6.0, 5.999, 10.0], index=list("abcdef"))
    classes, groups = assign_classes(p, ClassScheme())
    assert classes.tolist() == ["potent", "active", "active", "intermediate", "inactive", "potent"]
    assert groups.tolist() == [1, 1, 1, 2, 2, 1]
    assert list(classes.index) == list("abcdef")


def test_custom_scheme():
    scheme = ClassScheme(bounds=(("hit", 6.0),), floor_label="miss", group1=("hit",))
    classes, groups = assign_classes(pd.Series([6.5, 5.0]), scheme)
    assert classes.tolist() == ["hit", "miss"]
    assert groups.tolist() == [1, 2]


def test_nan_is_an_error():
    with pytest.raises(ValueError):
        assign_classes(pd.Series([7.0, math.nan]), ClassScheme())


# -- end to end ------------------------------------------------------------


def test_curate_chembl_produces_the_curated_contract(make_record):
    records = [
        make_record(activity_id=1, standard_value="5"),  # 8.30 -> potent
        make_record(activity_id=2, parent_molecule_chembl_id="CHEMBL2", canonical_smiles="CCO"),
        make_record(activity_id=3, standard_relation=">"),
    ]
    result = curate_chembl(records, RunParams())
    assert tuple(result.table.columns) == CURATED_COLUMNS
    assert sorted(result.table["activity_class"]) == ["active", "potent"]
    names = [s.name for s in result.steps]
    assert names[: len(STEP_ORDER)] == STEP_ORDER
    assert "aggregation" in names


def test_assay_context_is_retained_and_can_be_filtered(make_record):
    records = [
        make_record(
            activity_id=11,
            assay_chembl_id="CHEMBL_A",
            document_chembl_id="CHEMBL_D",
            confidence_score=9,
            standard_value="100",
        ),
        make_record(
            activity_id=12,
            assay_chembl_id="CHEMBL_B",
            confidence_score=6,
            standard_value="1000",
        ),
    ]
    result = curate_chembl(records, RunParams())
    assert set(result.evidence["record_id"]) == {"11", "12"}
    assert set(result.evidence["assay_chembl_id"]) == {"CHEMBL_A", "CHEMBL_B"}
    assert result.evidence.set_index("record_id").loc["11", "document_chembl_id"] == "CHEMBL_D"
    assert result.table.iloc[0]["pactivity"] == pytest.approx(6.5)

    restricted = curate_chembl(
        records,
        RunParams(curation=CurationParams(assay_ids=("CHEMBL_A",), min_confidence_score=9)),
    )
    assert restricted.evidence["record_id"].tolist() == ["11"]
    assert restricted.evidence.iloc[0]["activity_url"].endswith("/activity/11.json")
    assert restricted.evidence.iloc[0]["assay_url"].endswith("/assay/CHEMBL_A.json")
    assert restricted.table.iloc[0]["pactivity"] == pytest.approx(7.0)
    assert [
        step.removed for step in restricted.steps if step.name in {"assay_id", "target_confidence"}
    ] == [1, 0]


def test_time_safe_labels_ignore_later_measurement_of_training_compound(make_record):
    records = [
        make_record(
            activity_id=1,
            molecule_chembl_id="A",
            parent_molecule_chembl_id="A",
            canonical_smiles="CCO",
            standard_value="1000",
            document_year=2018,
        ),
        make_record(
            activity_id=2,
            molecule_chembl_id="A",
            parent_molecule_chembl_id="A",
            canonical_smiles="CCO",
            standard_value="1",
            document_year=2021,
        ),
        make_record(
            activity_id=3,
            molecule_chembl_id="B",
            parent_molecule_chembl_id="B",
            canonical_smiles="CCN",
            standard_value="100",
            document_year=2021,
        ),
    ]
    params = RunParams(model=ModelParams(split="time", time_cutoff=2019))
    curated = curate_chembl(records, params)
    assert curated.table.set_index("molecule_id").loc["A", "pactivity"] == pytest.approx(7.5)
    safe = time_safe_table(curated, params).set_index("molecule_id")
    assert safe.loc["A", "pactivity"] == pytest.approx(6.0)
    assert safe.loc["A", "n_measurements"] == 1
    assert safe.loc["B", "document_year"] == 2021
    assert len(safe) == 2


def test_curate_table_produces_the_curated_contract():
    frame = measurements([("a", "CCO", 8.5), ("b", "CCN", 5.0)])
    result = curate_table(frame, RunParams())
    assert tuple(result.table.columns) == CURATED_COLUMNS
    assert result.table.set_index("molecule_id")["group"].to_dict() == {"a": 1, "b": 2}

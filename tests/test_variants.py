"""Protein-variant normalization must be conservative and auditable."""

from sarscope.variants import normalize_variant, variant_matches


def test_equivalent_substitution_notations_share_one_canonical_form():
    for text in ("V600E", "p.Val600Glu", "Val600Glu", "BRAF (V600E)"):
        annotation = normalize_variant(text, source="chembl_structured")
        assert annotation.normalized_mutations == ("V600E",)
        assert annotation.display_hgvs == ("p.Val600Glu",)
        assert annotation.status == "normalized_variant"


def test_multiple_substitutions_are_deduplicated_and_position_sorted():
    annotation = normalize_variant("V600E; T790M; p.Val600Glu")
    assert annotation.normalized_mutations == ("V600E", "T790M")
    assert annotation.variant_class == "multiple_substitutions"


def test_missing_annotation_is_not_conflated_with_explicit_wild_type():
    assert normalize_variant(None).status == "unannotated"
    assert normalize_variant("wild-type").status == "explicit_wild_type"
    assert variant_matches(None, None)
    assert not variant_matches(None, "wild-type")


def test_ambiguous_variant_text_is_preserved_but_not_invented():
    annotation = normalize_variant("exon 19 deletion mutant", source="database_text")
    assert annotation.status == "variant_unparsed"
    assert annotation.raw_annotation == "exon 19 deletion mutant"
    assert annotation.normalized_mutations == ()
    assert (
        normalize_variant("HEK293T reporter assay", source="database_text").status == "unannotated"
    )


def test_sequence_check_warns_on_wrong_reference_residue():
    sequence = "A" * 599 + "G" + "A" * 10
    annotation = normalize_variant("V600E", reference_sequence=sequence, accession="P15056")
    assert annotation.sequence_validation == "warning"
    assert "reference has G" in annotation.warnings[0]
    assert annotation.accession == "P15056"


def test_matching_normalizes_substitutions_but_requires_full_multi_mutation_set():
    assert variant_matches("p.Val600Glu", "V600E")
    assert variant_matches("V600E; T790M", "p.Thr790Met / p.Val600Glu")
    assert not variant_matches("V600E", "V600E; T790M")

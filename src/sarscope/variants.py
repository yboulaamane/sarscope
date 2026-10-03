"""Conservative protein-variant normalization with raw evidence preserved.

Database mutation annotations are not identifiers: the same substitution may
arrive as ``V600E``, ``p.Val600Glu`` or ``BRAF (V600E)``.  This module
normalizes only substitutions that can be parsed without guessing.  Missing
annotations remain distinct from an explicit wild-type statement, and every
result retains the source text.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

_ONE_TO_THREE = {
    "A": "Ala",
    "C": "Cys",
    "D": "Asp",
    "E": "Glu",
    "F": "Phe",
    "G": "Gly",
    "H": "His",
    "I": "Ile",
    "K": "Lys",
    "L": "Leu",
    "M": "Met",
    "N": "Asn",
    "P": "Pro",
    "Q": "Gln",
    "R": "Arg",
    "S": "Ser",
    "T": "Thr",
    "V": "Val",
    "W": "Trp",
    "Y": "Tyr",
    "*": "Ter",
}
_THREE_TO_ONE = {value.upper(): key for key, value in _ONE_TO_THREE.items()}
_AA = "ACDEFGHIKLMNPQRSTVWY"
_ONE_LETTER = re.compile(
    rf"(?<![A-Z0-9])([{_AA}])(\d{{1,5}})([{_AA}\*])(?![A-Z0-9])",
    re.IGNORECASE,
)
_THREE_LETTER = re.compile(
    r"(?<![A-Za-z0-9])(?:p\.)?"
    r"(Ala|Cys|Asp|Glu|Phe|Gly|His|Ile|Lys|Leu|Met|Asn|Pro|Gln|Arg|Ser|Thr|Val|Trp|Tyr)"
    r"(\d{1,5})"
    r"(Ala|Cys|Asp|Glu|Phe|Gly|His|Ile|Lys|Leu|Met|Asn|Pro|Gln|Arg|Ser|Thr|Val|Trp|Tyr|Ter)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_EXPLICIT_WT = re.compile(r"\b(?:wild[ -]?type|wt)\b", re.IGNORECASE)
_VARIANT_WORD = re.compile(
    r"\b(?:mutant|mutation|variant|delet(?:ion|ed)|insert(?:ion|ed)|fusion|"
    r"frameshift|truncat(?:ed|ion)|duplication)\b",
    re.IGNORECASE,
)

VariantStatus = Literal[
    "unannotated", "explicit_wild_type", "normalized_variant", "variant_unparsed"
]


@dataclass(frozen=True)
class VariantAnnotation:
    """Normalized interpretation plus the untouched source annotation."""

    raw_annotation: str | None
    status: VariantStatus
    normalized_mutations: tuple[str, ...] = ()
    display_hgvs: tuple[str, ...] = ()
    variant_class: str = "none"
    evidence_source: str = "unknown"
    confidence: str = "unknown"
    accession: str | None = None
    sequence_validation: str = "not_checked"
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mutation_sort_key(mutation: str) -> tuple[int, str]:
    match = re.search(r"\d+", mutation)
    return (int(match.group()) if match else 10**9, mutation)


def _extract_substitutions(text: str) -> tuple[str, ...]:
    found: set[str] = set()
    for match in _ONE_LETTER.finditer(text.upper()):
        found.add(f"{match.group(1).upper()}{int(match.group(2))}{match.group(3).upper()}")
    for match in _THREE_LETTER.finditer(text):
        ref = _THREE_TO_ONE[match.group(1).upper()]
        alt = _THREE_TO_ONE[match.group(3).upper()]
        found.add(f"{ref}{int(match.group(2))}{alt}")
    return tuple(sorted(found, key=_mutation_sort_key))


def _display_hgvs(mutation: str) -> str:
    match = re.fullmatch(rf"([{_AA}])(\d+)([{_AA}\*])", mutation)
    if match is None:  # pragma: no cover - guarded by the parser
        return mutation
    return f"p.{_ONE_TO_THREE[match.group(1)]}{match.group(2)}{_ONE_TO_THREE[match.group(3)]}"


def normalize_variant(
    raw: Any,
    *,
    source: str = "unknown",
    accession: Any = None,
    reference_sequence: str | None = None,
) -> VariantAnnotation:
    """Normalize confidently parsed substitutions and retain all raw evidence.

    A reference sequence is optional.  When supplied, residue positions and
    reference amino acids are checked; an accession alone is provenance, not
    proof that the residue numbering matches.
    """
    raw_text = None if raw is None else str(raw).strip() or None
    accession_text = None if accession is None else str(accession).strip() or None
    confidence = "structured" if source.startswith("chembl") else "text_derived"
    if raw_text is None:
        return VariantAnnotation(
            None,
            "unannotated",
            evidence_source=source,
            confidence="unknown",
            accession=accession_text,
        )

    mutations = _extract_substitutions(raw_text)
    warnings: list[str] = []
    sequence_validation = "not_checked"
    if mutations and reference_sequence:
        sequence = re.sub(r"\s+", "", reference_sequence).upper()
        for mutation in mutations:
            match = re.fullmatch(rf"([{_AA}])(\d+)([{_AA}\*])", mutation)
            assert match is not None
            expected, position = match.group(1), int(match.group(2))
            if position > len(sequence):
                warnings.append(
                    f"{mutation}: position {position} exceeds reference length {len(sequence)}"
                )
            elif sequence[position - 1] != expected:
                warnings.append(
                    f"{mutation}: reference has {sequence[position - 1]} at position {position}, "
                    f"not {expected}"
                )
        sequence_validation = "warning" if warnings else "consistent"

    if mutations:
        return VariantAnnotation(
            raw_text,
            "normalized_variant",
            mutations,
            tuple(_display_hgvs(mutation) for mutation in mutations),
            "substitution" if len(mutations) == 1 else "multiple_substitutions",
            source,
            confidence,
            accession_text,
            sequence_validation,
            tuple(warnings),
        )
    if _EXPLICIT_WT.search(raw_text):
        return VariantAnnotation(
            raw_text,
            "explicit_wild_type",
            variant_class="wild_type",
            evidence_source=source,
            confidence=confidence,
            accession=accession_text,
        )
    return VariantAnnotation(
        raw_text,
        "variant_unparsed" if _VARIANT_WORD.search(raw_text) else "unannotated",
        variant_class="other" if _VARIANT_WORD.search(raw_text) else "none",
        evidence_source=source,
        confidence=confidence if _VARIANT_WORD.search(raw_text) else "unknown",
        accession=accession_text,
        warnings=("Mutation-related text could not be normalized; inspect the raw annotation.",)
        if _VARIANT_WORD.search(raw_text)
        else (),
    )


def variant_matches(query: str | None, raw: Any, *, source: str = "chembl_structured") -> bool:
    """Match a selection without conflating absent annotations and wild type.

    Parsed multi-substitution annotations require the same complete mutation
    set.  Unparsed annotations fall back to case- and whitespace-normalized
    equality, retaining support for database-specific construct descriptions.
    """
    if query == "any":
        return True
    observed = normalize_variant(raw, source=source)
    if query is None:
        return observed.raw_annotation is None
    wanted = normalize_variant(query, source="user_query")
    if wanted.normalized_mutations:
        return wanted.normalized_mutations == observed.normalized_mutations
    if wanted.status == "explicit_wild_type":
        return observed.status == "explicit_wild_type"

    def collapse(value: Any) -> str:
        return " ".join(str(value).casefold().split())

    return observed.raw_annotation is not None and collapse(query) == collapse(
        observed.raw_annotation
    )

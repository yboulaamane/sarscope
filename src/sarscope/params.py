"""Every analysis choice, in one place, with its default and the reason for it.

Nothing in the analysis layer hard-codes a threshold; it all flows from these
dataclasses, and all of it lands in ``provenance.json`` so a report can always
be traced back to the exact settings that produced it.

Curation gets its own dataclass with no implicit behaviour because that is where
the choices matter most. Measured on BRAF (CHEMBL5145, 11,017 IC50 records),
the curation settings below move the final dataset between roughly 2,900 and
6,700 molecules - a factor of two, decided entirely by options that a typical
write-up describes in one sentence. Every one of them is therefore explicit,
recorded, and counted in the curation log.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

SplitStrategy = Literal["random", "scaffold", "time", "source"]
FingerprintName = Literal["ecfp4", "maccs", "pubchem"]


@dataclass(frozen=True)
class CurationParams:
    """How raw ChEMBL activity records become one value per molecule."""

    #: ChEMBL ``standard_type`` values to keep. Mixing IC50 with Ki or EC50
    #: pools measurements that are not on the same scale, so the default is one.
    standard_types: tuple[str, ...] = ("IC50",)
    #: Only exact measurements. Censored records ("> 10000 nM") carry a bound,
    #: not a value; taking the median of bounds and values together is
    #: meaningless. 15.7% of BRAF IC50 records are censored.
    relations: tuple[str, ...] = ("=",)
    #: Only molar units, converted to -log10(M). Mass units (ug.mL-1) would need
    #: a molecular weight and are rare.
    units: tuple[str, ...] = ("nM",)
    #: ChEMBL assay types: B binding, F functional, A ADMET. A binding IC50 and
    #: a cell-based one measure different things, so the default keeps binding.
    assay_types: tuple[str, ...] = ("B",)
    #: Which ``assay_variant_mutation`` to keep. None means wild-type only (no
    #: variant annotation); a string such as "V600E" keeps only that mutant;
    #: "any" pools everything. 41% of BRAF IC50 records are V600E, and pooling
    #: wild-type and mutant mixes two proteins.
    variant: str | None = None
    #: Drop records ChEMBL flags as likely re-reports of an earlier measurement,
    #: so one value cited twice does not count twice in the median. Removes 22%
    #: of BRAF records. On the unfiltered set no molecule loses all its data;
    #: after the default filters 35 do, because their only wild-type binding
    #: record is the flagged re-report.
    drop_potential_duplicates: bool = True
    #: Drop records carrying a ``data_validity_comment`` ("Outside typical
    #: range", "Potential transcription error").
    drop_flagged_validity: bool = True
    #: Restrict to BAO assay formats (e.g. "single protein format"). None keeps
    #: all; the breakdown is always reported. A cell-based IC50 is a different
    #: measurement from an enzyme IC50, but "assay format" is too vague to
    #: exclude safely by default.
    bao_formats: tuple[str, ...] | None = None
    #: Restrict to explicit ChEMBL assays when a coherent assay subset is needed.
    assay_ids: tuple[str, ...] | None = None
    #: ChEMBL activity src_id values (e.g. 7 PubChem, 37 BindingDB). None keeps all.
    source_ids: tuple[int, ...] | None = None
    #: ChEMBL assay-to-target assignment confidence (0–9); None disables filtering.
    min_confidence_score: int | None = None
    #: Keep only records from documents published up to this year. Useful for
    #: approximating an older ChEMBL release. None keeps all.
    max_document_year: int | None = None
    #: How repeated measurements of one molecule combine. The median resists a
    #: single mistyped value, which a mean does not.
    aggregate: Literal["median", "mean"] = "median"
    #: Molecules whose measurements span more than this many log units are
    #: dropped as irreconcilable. None keeps them, and the spread is always
    #: reported as ``pactivity_range`` either way.
    max_replicate_range: float | None = None
    #: Run the canonical-tautomer step of Sorbent's standardiser. It changes
    #: donor/acceptor counts on some molecules, so it is recorded.
    canonical_tautomer: bool = True


@dataclass(frozen=True)
class ClassScheme:
    """Potency bins on the -log10(M) scale.

    The default is the common four-way split at 100 nM, 1 uM and 10 uM
    (pIC50 8, 7, 6), with Group 1 the potent and active half. Bins are a
    convention, not a fact about the data: change them here and every class
    count, enrichment factor and model label follows.
    """

    #: Lower bounds, highest first. A molecule takes the first label whose
    #: bound it meets; anything below the last bound gets ``floor_label``.
    bounds: tuple[tuple[str, float], ...] = (
        ("potent", 8.0),
        ("active", 7.0),
        ("intermediate", 6.0),
    )
    floor_label: str = "inactive"
    #: Labels forming Group 1, the "active" side for enrichment factors.
    group1: tuple[str, ...] = ("potent", "active")

    @property
    def labels(self) -> tuple[str, ...]:
        return (*(label for label, _ in self.bounds), self.floor_label)


@dataclass(frozen=True)
class LandscapeParams:
    """Structure-activity similarity (SAS) map and activity cliffs."""

    #: Fingerprints to build a SAS map for. Which one you pick changes what
    #: counts as a cliff, so more than one is the honest default: a pair that is
    #: a cliff under every fingerprint is a stronger finding than one that is
    #: not. PubChem needs the optional ``pubchem`` extra, so it is off here.
    fingerprints: tuple[FingerprintName, ...] = ("ecfp4", "maccs")
    #: A pair is a cliff when similarity > this ...
    similarity_threshold: float = 0.9
    #: ... and the potency difference > this, in log units. Two orders of
    #: magnitude between near-identical structures is the usual cliff criterion.
    activity_threshold: float = 2.0
    #: A molecule is a cliff *generator* when its cliff count exceeds
    #: mean + k * SD, taken over molecules with at least one cliff.
    generator_sd: float = 2.0


@dataclass(frozen=True)
class FeatureParams:
    """Fingerprint features and the filter applied before modelling."""

    fingerprint: FingerprintName = "ecfp4"
    #: ECFP bit length.
    ecfp_bits: int = 2048
    #: Drop features with variance below this. On binary bits variance is
    #: p(1 - p), so 0.1 keeps only bits set in roughly 11-89% of molecules -
    #: an aggressive cut that discards both the near-constant and the very rare.
    variance_threshold: float = 0.1
    #: Drop one of each feature pair correlated above this.
    correlation_threshold: float = 0.95


@dataclass(frozen=True)
class MatchedPairParams:
    """Bounds for single-cut matched molecular-pair enumeration."""

    max_variable_heavy_atoms: int = 10
    max_pairs: int = 100_000


@dataclass(frozen=True)
class ModelParams:
    """Classification and continuous-regression bake-offs."""

    features: FeatureParams = field(default_factory=FeatureParams)
    #: Algorithm names from ``analysis.model.ALGORITHMS``. The default is a fast
    #: subset; "all" expands to every algorithm registered there.
    algorithms: tuple[str, ...] = (
        "extra_trees",
        "random_forest",
        "gradient_boosting",
        "nearest_neighbors",
    )
    #: Continuous pActivity estimators from ``analysis.regression``.  Kept
    #: separate from ``algorithms`` because a name such as ``random_forest``
    #: maps to a classifier in one bake-off and a regressor in the other.
    regression_algorithms: tuple[str, ...] = (
        "extra_trees",
        "random_forest",
        "gradient_boosting",
        "nearest_neighbors",
        "svr",
    )
    test_fraction: float = 0.2
    cv_folds: int = 10
    #: "scaffold" keeps every Murcko scaffold wholly on one side of each split,
    #: which is what generalisation to new chemotypes actually means. A random
    #: split scatters one congeneric series across both sides and flatters the
    #: model accordingly. "source" holds out origin-unique compounds and uses
    #: scaffold CV inside the remaining training data.
    split: SplitStrategy = "scaffold"
    #: For ``split="time"``, train on measurements documented on or before
    #: this year and test only on later molecules.
    time_cutoff: int = 2019
    #: For ``split="source"``, reserve compounds measured only in this ChEMBL
    #: origin (e.g. 7 PubChem or 37 BindingDB) as an external-origin test.
    source_test_id: int | None = None
    #: Random oversampling of minority classes, applied to training folds only.
    oversample: bool = True
    #: Also run the naive order of operations (select features and oversample on
    #: everything, then split) and report the score gap. See ``analysis.model``
    #: for why that order leaks.
    leakage_audit: bool = True
    seed: int = 42


@dataclass(frozen=True)
class RunParams:
    curation: CurationParams = field(default_factory=CurationParams)
    classes: ClassScheme = field(default_factory=ClassScheme)
    landscape: LandscapeParams = field(default_factory=LandscapeParams)
    matched_pairs: MatchedPairParams = field(default_factory=MatchedPairParams)
    model: ModelParams = field(default_factory=ModelParams)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dicts for provenance.json. Tuples stay tuples; json writes them as lists."""
        return dataclasses.asdict(self)

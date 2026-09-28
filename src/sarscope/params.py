"""Every analysis choice, in one place, with its default and where it came from.

The reference workflow is Aouidate, *J. Mol. Liq.* 401 (2024) 124705 (BRAF
inhibitors). Where that paper states a value, the default reproduces it and the
comment says "paper". Where it is silent, the default is a choice made here and
the comment says why. Nothing in the analysis layer hard-codes a threshold; it
all flows from these dataclasses, and all of it lands in ``provenance.json`` so
a report can always be traced back to the exact settings that produced it.

Measured on BRAF (CHEMBL5145, ChEMBL 37, documents up to 2022), the curation
choices the paper leaves unstated move the dataset between roughly 3,400 and
6,650 molecules. That spread is why curation gets its own dataclass with no
implicit behaviour, rather than a couple of flags.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Literal

SplitStrategy = Literal["random", "scaffold"]
FingerprintName = Literal["ecfp4", "maccs", "pubchem"]


@dataclass(frozen=True)
class CurationParams:
    """How raw ChEMBL activity records become one value per molecule."""

    #: ChEMBL ``standard_type`` values to keep. Paper: IC50.
    standard_types: tuple[str, ...] = ("IC50",)
    #: Only exact measurements. Censored records ("> 10000 nM") carry a bound,
    #: not a value; taking the median of bounds and values together is
    #: meaningless. 15.7% of BRAF IC50 records are censored. Paper: unstated.
    relations: tuple[str, ...] = ("=",)
    #: Only molar units, converted to -log10(M). Mass units (ug.mL-1) would need
    #: a molecular weight and are rare. Paper: unstated.
    units: tuple[str, ...] = ("nM",)
    #: ChEMBL assay types: B binding, F functional, A ADMET. Paper: unstated.
    assay_types: tuple[str, ...] = ("B",)
    #: Which ``assay_variant_mutation`` to keep. None means wild-type only (no
    #: variant annotation); a string such as "V600E" keeps only that mutant;
    #: "any" pools everything. 41% of BRAF IC50 records are V600E, and pooling
    #: wild-type and mutant mixes two proteins. Paper: unstated.
    variant: str | None = None
    #: Drop records ChEMBL flags as likely re-reports of an earlier measurement,
    #: so one value cited twice does not count twice in the median. Removes 22%
    #: of BRAF records. On the unfiltered set no molecule loses all its data;
    #: after the default filters 35 do, because their only wild-type binding
    #: record is the flagged re-report. Paper: unstated.
    drop_potential_duplicates: bool = True
    #: Drop records carrying a ``data_validity_comment`` ("Outside typical
    #: range", "Potential transcription error"). Paper: unstated.
    drop_flagged_validity: bool = True
    #: Restrict to BAO assay formats (e.g. "single protein format"). None keeps
    #: all; the breakdown is always reported. A cell-based IC50 is a different
    #: measurement from an enzyme IC50, but "assay format" is too vague to
    #: exclude safely by default. Paper: unstated.
    bao_formats: tuple[str, ...] | None = None
    #: Keep only records from documents published up to this year. Useful for
    #: approximating an older ChEMBL release. None keeps all.
    max_document_year: int | None = None
    #: How repeated measurements of one molecule combine. Paper: median.
    aggregate: Literal["median", "mean"] = "median"
    #: Molecules whose measurements span more than this many log units are
    #: dropped as irreconcilable. None keeps them (but the spread is always
    #: reported). Paper: unstated.
    max_replicate_range: float | None = None
    #: Run the canonical-tautomer step of Sorbent's standardiser. It changes
    #: donor/acceptor counts on some molecules, so it is recorded.
    canonical_tautomer: bool = True


@dataclass(frozen=True)
class ClassScheme:
    """Potency bins on the -log10(M) scale.

    Paper: potent >= 8 > active >= 7 > intermediate >= 6 > inactive. Group 1 is
    potent + active, Group 2 is intermediate + inactive.
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

    #: Fingerprints to build a SAS map for. Paper: ECFP4, MACCS, PubChem.
    #: PubChem needs the optional ``pubchem`` extra, so it is off by default.
    fingerprints: tuple[FingerprintName, ...] = ("ecfp4", "maccs")
    #: A pair is a cliff when similarity > this ... Paper: 0.9.
    similarity_threshold: float = 0.9
    #: ... and the potency difference > this, in log units. Paper: 2.0.
    activity_threshold: float = 2.0
    #: A molecule is a cliff *generator* when its cliff count exceeds
    #: mean + k * SD, taken over molecules with at least one cliff. Paper: k=2.
    generator_sd: float = 2.0


@dataclass(frozen=True)
class FeatureParams:
    """Fingerprint features and the filter applied before modelling."""

    fingerprint: FingerprintName = "ecfp4"
    #: ECFP bit length. Paper: unstated.
    ecfp_bits: int = 2048
    #: Drop features with variance below this. Paper: 0.1. On binary bits this
    #: keeps only bits set in roughly 11-89% of molecules.
    variance_threshold: float = 0.1
    #: Drop one of each feature pair correlated above this. Paper: 0.95.
    correlation_threshold: float = 0.95


@dataclass(frozen=True)
class ModelParams:
    """Classifier bake-off."""

    features: FeatureParams = field(default_factory=FeatureParams)
    #: Algorithm names from ``analysis.model.ALGORITHMS``. The default is the
    #: fast subset; "all" in the CLI expands to the paper's full fourteen.
    algorithms: tuple[str, ...] = (
        "extra_trees",
        "random_forest",
        "gradient_boosting",
        "nearest_neighbors",
    )
    #: Paper: 80:20.
    test_fraction: float = 0.2
    #: Paper: 10.
    cv_folds: int = 10
    #: "scaffold" keeps every Murcko scaffold wholly on one side of each split,
    #: which is what generalisation to new chemotypes actually means. The
    #: paper used a random split.
    split: SplitStrategy = "scaffold"
    #: Random oversampling of minority classes, applied to training folds only.
    oversample: bool = True
    #: Also run the paper's order of operations (oversample, then split) and
    #: report the score gap. See ``analysis.model`` for why that order leaks.
    leakage_audit: bool = True
    #: Paper: 42.
    seed: int = 42


@dataclass(frozen=True)
class RunParams:
    curation: CurationParams = field(default_factory=CurationParams)
    classes: ClassScheme = field(default_factory=ClassScheme)
    landscape: LandscapeParams = field(default_factory=LandscapeParams)
    model: ModelParams = field(default_factory=ModelParams)

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dicts for provenance.json. Tuples stay tuples; json writes them as lists."""
        return dataclasses.asdict(self)

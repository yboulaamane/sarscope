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
    #: Which ``assay_variant_mutation`` annotation to keep. None keeps records
    #: without a mutation annotation, not confirmed wild-type protein; a string
    #: keeps the same normalized substitution set for any target (for example,
    #: V600E matches p.Val600Glu); unparsed text falls back to exact matching;
    #: "any" pools all annotations.
    variant: str | None = None
    #: Drop records ChEMBL flags as likely re-reports of an earlier measurement,
    #: so one value cited twice does not count twice in the median. Removes 22%
    #: of BRAF records. On the unfiltered set no molecule loses all its data;
    #: after the default filters 35 do, because their only unannotated binding
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

    The default is a four-way split at 10 nM, 100 nM and 1 uM
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
    #: Continuous descriptors and hybrids get fold-local median imputation and scaling.
    representation: Literal["fingerprint", "rdkit2d", "hybrid", "molfeat2d"] = "fingerprint"
    #: Ordered names are saved with fitted models; empty uses the medicinal-chemistry preset.
    descriptor_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class MatchedPairParams:
    """Bounds for single-cut matched molecular-pair enumeration."""

    max_variable_heavy_atoms: int = 10
    max_pairs: int = 100_000


@dataclass(frozen=True)
class FieldQsarParams:
    """Bounded, experimental field QSAR; not a claim of Tripos equivalence.

    Geometry and charge assumptions are recorded alongside the model. The
    default Gaussian fields avoid the undefined electrostatic regions in LJ
    fields. Limits protect shared browser hosts, not statistical validity.
    """

    method: Literal["comfa", "comsia"] = "comsia"
    #: Public Tripos-5.2 SE parameters, NOT a claim of full vendor parity.
    field_definition: Literal["sarscope", "tripos52_se"] = "sarscope"
    #: Empty selects all fields for the method; explicit subsets support published comparisons.
    field_subset: tuple[str, ...] = ()
    field_scaling: Literal["raw", "block_std", "column_std"] = "raw"
    grid_convention: Literal["enclosing", "rounded_extent"] = "enclosing"
    #: Fractions of one lattice step; controls sampling phase, not molecular alignment.
    grid_phase: tuple[float, float, float] = (0.0, 0.0, 0.0)
    spacing: float = 2.0
    margin: float = 4.0
    charge_model: str = "gasteiger"
    allow_charge_fallback: bool = False
    alpha: float = 0.3
    dielectric: float = 1.0
    dielectric_model: Literal["constant", "distance"] = "constant"
    electrostatic_cutoff: float | None = None
    #: SYBYL documents training-column means for missing prediction descriptors.
    missing_field_policy: Literal["refuse", "training_mean"] = "refuse"
    max_imputed_fraction: float = 0.1
    steric_cutoff: float = 30.0
    minimum_distance: float = 0.3
    probe_radius: float = 1.70
    probe_epsilon: float = 0.1070
    filter_mode: Literal["absolute", "percentile"] = "percentile"
    #: How the retained-column budget is allocated across field blocks.
    #:
    #: ``per_field`` screens inside each block and gives each its own share, so
    #: the retained set does not depend on the units a field happens to carry.
    #: ``global_raw`` ranks every column together on raw standard deviation.
    #:
    #: Raw ranking is defensible only when the blocks share units. It does for
    #: CoMFA, where steric and electrostatic are both kcal/mol, and it does not
    #: for CoMSIA, where steric weights are cubed radii and electrostatic
    #: weights are partial charges. Ranking those together compares angstroms
    #: cubed against elementary charge: the larger-magnitude block wins on its
    #: units alone and can take the whole budget. That happened in the
    #: 16 October 2026 qualification, where the Gaussian fit retained 2000
    #: steric and 0 electrostatic columns while the matched LJ/Coulomb fit,
    #: whose blocks do share units, retained 217 and 170.
    #:
    #: ``global_raw`` is retained to reproduce that frozen evidence exactly.
    field_screen: Literal["per_field", "global_raw"] = "per_field"
    minimum_sd: float = 2.0
    keep_fraction: float = 0.3
    max_features: int = 2000
    component_cap: int = 4
    #: Prespecified literature count, never selected by held-out performance.
    fixed_components: int | None = None
    component_improvement: float = 0.05
    component_selection: Literal["relative_improvement", "max_q2"] = "relative_improvement"
    #: None is for explicitly labelled historical benchmarks only, not native validation.
    component_sample_ratio: int | None = 5
    pls_max_iter: int = 500
    pls_tolerance: float = 1e-6
    alignment: Literal["prealigned", "rigid_mcs", "constrained"] = "prealigned"
    #: A supplied frame need not share a literal core. Skipped MCS QC is reported, not passed.
    assess_prealigned_core: bool = True
    #: Preserve reviewed explicit-H coordinates rather than regenerating them from heavy atoms.
    preserve_input_hydrogens: bool = False
    reference_index: int = 0
    min_core_coverage: float = 0.5
    max_alignment_rmsd: float = 1.0
    mcs_timeout: int = 5
    max_mappings: int = 256
    conformers: int = 8
    molecule_timeout: int = 30
    alignment_timeout: int = 180
    permutation_timeout: int = 120
    max_molecules: int = 100
    max_grid_points: int = 12000
    max_memory_mb: int = 256
    chunk_points: int = 512
    permutations: int = 19
    permutation_alpha: float = 0.05
    contour_percentile: float = 80.0
    seed: int = 42

    def __post_init__(self) -> None:
        import math

        if self.method not in ("comfa", "comsia"):
            raise ValueError("Choose comfa or comsia fields.")
        if self.field_definition not in ("sarscope", "tripos52_se"):
            raise ValueError("Unknown field definition.")
        if self.field_definition == "tripos52_se" and (
            not self.field_subset or not set(self.field_subset) <= {"steric", "electrostatic"}
        ):
            raise ValueError(
                "The public Tripos-5.2 profile requires an explicit steric/electrostatic subset. "
                "SYBYL hydrophobic tables and directional H-bond sites are not substituted."
            )
        if self.dielectric_model not in ("constant", "distance"):
            raise ValueError("Choose constant or distance-dependent dielectric.")
        if self.missing_field_policy not in ("refuse", "training_mean"):
            raise ValueError("Unknown missing-field policy.")
        if self.missing_field_policy == "training_mean" and self.method != "comfa":
            raise ValueError("Training-mean overlap handling applies to CoMFA only.")
        if self.electrostatic_cutoff is not None and (
            not math.isfinite(self.electrostatic_cutoff) or self.electrostatic_cutoff <= 0
        ):
            raise ValueError("Electrostatic cutoff must be positive or None.")
        if not math.isfinite(self.max_imputed_fraction) or not 0 <= self.max_imputed_fraction <= 1:
            raise ValueError("Maximum mean-filled fraction must be in [0, 1].")
        allowed = {"steric", "electrostatic"}
        if self.method == "comsia":
            allowed |= {"hydrophobic", "hbond_donor", "hbond_acceptor"}
        if not set(self.field_subset) <= allowed or len(set(self.field_subset)) != len(
            self.field_subset
        ):
            raise ValueError("Field subset must contain unique names supported by the method.")
        if self.field_scaling not in ("raw", "block_std", "column_std"):
            raise ValueError("Choose raw, block_std or column_std training-only field scaling.")
        if self.grid_convention not in ("enclosing", "rounded_extent"):
            raise ValueError("Choose enclosing or rounded_extent grid bounds.")
        if len(self.grid_phase) != 3 or any(
            not math.isfinite(value) or not 0 <= value < 1 for value in self.grid_phase
        ):
            raise ValueError("Grid phase needs three finite fractions in [0, 1).")
        if self.grid_convention == "rounded_extent" and any(self.grid_phase):
            raise ValueError("Grid phase is supported only for the native enclosing grid.")
        if self.component_selection not in ("relative_improvement", "max_q2"):
            raise ValueError("Unknown PLS component-selection rule.")
        if self.component_sample_ratio is not None and (
            type(self.component_sample_ratio) is not int or self.component_sample_ratio < 1
        ):
            raise ValueError("Component sample ratio must be positive or explicitly None.")
        if not math.isfinite(self.pls_tolerance) or self.pls_tolerance <= 0:
            raise ValueError("PLS tolerance must be finite and positive.")
        if self.fixed_components is not None and (
            type(self.fixed_components) is not int
            or not 1 <= self.fixed_components <= self.component_cap
        ):
            raise ValueError("Prespecified PLS components must be within the component cap.")
        if self.alignment not in ("prealigned", "rigid_mcs", "constrained"):
            raise ValueError("Unknown 3D alignment route.")
        if self.preserve_input_hydrogens and self.alignment != "prealigned":
            raise ValueError("Preserving input hydrogens requires a pre-aligned SDF.")
        if self.filter_mode not in ("absolute", "percentile"):
            raise ValueError("Unknown field-filter mode.")
        if self.field_screen not in ("per_field", "global_raw"):
            raise ValueError("Unknown field screen; use per_field or global_raw.")
        if self.method == "comsia" and self.filter_mode != "percentile":
            raise ValueError("CoMSIA requires a relative variance filter, not a kcal/mol cutoff.")
        for name in (
            "spacing",
            "margin",
            "alpha",
            "dielectric",
            "steric_cutoff",
            "minimum_distance",
            "probe_radius",
            "probe_epsilon",
            "max_alignment_rmsd",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        for name in (
            "component_cap",
            "pls_max_iter",
            "max_features",
            "max_mappings",
            "mcs_timeout",
            "conformers",
            "molecule_timeout",
            "alignment_timeout",
            "permutation_timeout",
            "max_molecules",
            "max_grid_points",
            "max_memory_mb",
            "chunk_points",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer.")
        if not 0 < self.keep_fraction <= 1 or not 0 < self.min_core_coverage <= 1:
            raise ValueError("Retained fraction and core coverage must be in (0, 1].")
        if not math.isfinite(self.minimum_sd) or self.minimum_sd < 0:
            raise ValueError("Minimum standard deviation must be finite and nonnegative.")
        if not 0 <= self.component_improvement < 1 or not 0 < self.permutation_alpha < 1:
            raise ValueError("Invalid selection or permutation threshold.")
        if not 50 <= self.contour_percentile < 100:
            raise ValueError("Contour percentile must be in [50, 100).")
        if type(self.permutations) is not int or not 0 <= self.permutations <= 999:
            raise ValueError("Use 0–999 permutations; zero is exploratory only.")
        if type(self.seed) is not int or not 1 <= self.seed <= 2147483647:
            raise ValueError("Use a positive 32-bit seed.")


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
    #: LOO is explicit and regression-only, primarily for literature protocol comparisons.
    regression_cv: Literal["default", "loo"] = "default"
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
    #: Used only by the explicit field-PLS workflow, never by ordinary 2D QSAR.
    field_qsar: FieldQsarParams | None = None


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

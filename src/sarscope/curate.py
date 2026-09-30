"""Raw activity records -> one standardised molecule, one potency, one class.

Three stages, each returning the step-by-step counts that become the curation
log in the report. Those counts are the point. Curation is usually described in
a sentence or two and then forgotten, yet on BRAF these settings alone move the
dataset between roughly 2,900 and 6,700 molecules - the single largest source of
variation in everything downstream. Every record that leaves the pipeline leaves
through a named step, and the count is kept.

1. ``filter_chembl_records`` - record-level filters driven by CurationParams,
   then conversion to -log10(M). ChEMBL-specific.
2. ``standardize_and_aggregate`` - source-agnostic. Standardise with Sorbent,
   merge by structure, aggregate replicates.
3. ``assign_classes`` - potency bins and Group 1/2.

``curate_chembl`` and ``curate_table`` chain them.

Identity is decided by structure, not by ID. ChEMBL gives salt forms their own
molecule IDs, so deduplicating on the ID alone counts a hydrochloride and its
free base as two compounds with two potencies. Here the parent ID is used
first and the standardised InChIKey second, so they collapse into one molecule
whose measurements are pooled.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import pandas as pd
from rdkit import Chem
from sorbent.chem.parse import parse_smiles, standardize, to_inchikey

from sarscope.params import ClassScheme, CurationParams, RunParams
from sarscope.sources.origins import source_id, source_name
from sarscope.units import MOLAR_OFFSETS, to_pactivity

#: One row per measurement. The contract between a source and curation.
#: ``record_id`` traces a row back to its origin (ChEMBL activity_id, or
#: file:line for a user table).
MEASUREMENT_COLUMNS: tuple[str, ...] = ("molecule_id", "smiles", "pactivity", "record_id")

#: One row per molecule. The contract between curation and every analysis.
CURATED_COLUMNS: tuple[str, ...] = (
    "molecule_id",  # representative: first merged ID in (len, str) order, CHEMBL9 < CHEMBL10
    "merged_ids",  # every source ID that collapsed into this row, ";"-joined
    "smiles",  # standardised canonical SMILES
    "inchikey",  # None only if RDKit's InChI layer refused the structure
    "pactivity",  # aggregated -log10(M)
    "n_measurements",
    "pactivity_range",  # max - min over replicates; 0.0 for a single value
    "document_year",  # earliest source year; missing for tables without a year column
    "activity_class",
    "group",  # 1 or 2
)


@dataclass(frozen=True)
class CurationStep:
    """One named filter and what it did. ``molecules_out`` counts distinct IDs."""

    name: str
    records_in: int
    records_out: int
    molecules_out: int
    detail: str = ""

    @property
    def removed(self) -> int:
        return self.records_in - self.records_out


@dataclass
class CurationResult:
    table: pd.DataFrame  # CURATED_COLUMNS
    steps: list[CurationStep] = field(default_factory=list)
    #: Structures that failed standardisation: molecule_id, smiles, reason.
    rejected: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=["molecule_id", "smiles", "reason"])
    )
    #: Retained measurements with assay, publication, endpoint and target context.
    evidence: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Pre-aggregation rows needed for measurement-level prospective validation.
    measurements: pd.DataFrame = field(default_factory=pd.DataFrame)


def id_order(identifier: str) -> tuple[int, str]:
    """Sort key that puts CHEMBL9 before CHEMBL10 and works for any string ID."""
    return (len(identifier), identifier)


def _parse_value(raw: Any) -> float | None:
    """ChEMBL sends numbers as strings. None for anything missing or unparseable."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _parent_id(record: dict[str, Any]) -> Any:
    return record.get("parent_molecule_chembl_id") or record.get("molecule_chembl_id")


def filter_chembl_records(
    records: Sequence[dict[str, Any]], params: CurationParams
) -> tuple[pd.DataFrame, list[CurationStep]]:
    """Apply the record-level filters and return MEASUREMENT_COLUMNS rows.

    Steps, in this order, each logged even when it removes nothing:

      standard_type      ``standard_type`` in params.standard_types
      relation           ``standard_relation`` in params.relations
      units              ``standard_units`` in params.units
      assay_type         ``assay_type`` in params.assay_types
      variant            params.variant: None -> ``assay_variant_mutation`` is
                         None; "any" -> no filter; otherwise equality
      potential_duplicate  drop ``potential_duplicate == 1`` if enabled
      validity           drop non-null ``data_validity_comment`` if enabled
      bao_format         ``bao_label`` in params.bao_formats, if set
      document_year      ``document_year <= params.max_document_year``, if set
                         (records with no year are dropped when the filter is on)
      structure          drop records with no ``canonical_smiles``
      value              drop missing or non-positive ``standard_value``

    ``molecule_id`` is ``parent_molecule_chembl_id`` (falling back to
    ``molecule_chembl_id``). ``record_id`` is ``str(activity_id)``.
    ``pactivity`` is ``units.to_pactivity(float(standard_value), standard_units)``.
    """
    unknown = [u for u in params.units if u not in MOLAR_OFFSETS]
    if unknown:
        raise ValueError(f"cannot convert units {unknown} to a molar scale")
    if params.min_confidence_score is not None and not 0 <= params.min_confidence_score <= 9:
        raise ValueError("min_confidence_score must be between 0 and 9")

    def variant_ok(r: dict[str, Any]) -> bool:
        if params.variant == "any":
            return True
        return r.get("assay_variant_mutation") == params.variant

    def year_ok(r: dict[str, Any]) -> bool:
        year = r.get("document_year")
        return params.max_document_year is None or (
            year is not None and year <= params.max_document_year
        )

    def value_ok(r: dict[str, Any]) -> bool:
        value = _parse_value(r.get("standard_value"))
        return value is not None and value > 0

    filters: list[tuple[str, Callable[[dict[str, Any]], bool]]] = [
        ("standard_type", lambda r: r.get("standard_type") in params.standard_types),
        ("relation", lambda r: r.get("standard_relation") in params.relations),
        ("units", lambda r: r.get("standard_units") in params.units),
        ("assay_type", lambda r: r.get("assay_type") in params.assay_types),
        (
            "source_origin",
            lambda r: params.source_ids is None or source_id(r.get("src_id")) in params.source_ids,
        ),
        ("variant", variant_ok),
        (
            "potential_duplicate",
            lambda r: not (params.drop_potential_duplicates and r.get("potential_duplicate") == 1),
        ),
        (
            "validity",
            lambda r: not (params.drop_flagged_validity and r.get("data_validity_comment")),
        ),
        (
            "bao_format",
            lambda r: params.bao_formats is None or r.get("bao_label") in params.bao_formats,
        ),
        (
            "assay_id",
            lambda r: params.assay_ids is None or r.get("assay_chembl_id") in params.assay_ids,
        ),
        (
            "target_confidence",
            lambda r: (
                params.min_confidence_score is None
                or (value := _parse_value(r.get("confidence_score"))) is not None
                and value >= params.min_confidence_score
            ),
        ),
        ("document_year", year_ok),
        ("structure", lambda r: bool(r.get("canonical_smiles"))),
        ("value", value_ok),
    ]

    kept = list(records)
    steps: list[CurationStep] = []
    for name, keep in filters:
        before = len(kept)
        kept = [r for r in kept if keep(r)]
        detail = ""
        if name == "standard_type":
            detail = (
                "pooled after conversion to -log10(molar): " + ", ".join(params.standard_types)
                if len(params.standard_types) > 1
                else f"kept {params.standard_types[0]}"
            )
        elif name == "units":
            detail = "converted concentration values to -log10(molar) before aggregation"
        steps.append(
            CurationStep(name, before, len(kept), len({_parent_id(r) for r in kept}), detail)
        )

    frame = pd.DataFrame(
        {
            "molecule_id": [str(_parent_id(r)) for r in kept],
            "smiles": [r["canonical_smiles"] for r in kept],
            "pactivity": [
                to_pactivity(float(r["standard_value"]), r["standard_units"]) for r in kept
            ],
            "record_id": [str(r.get("activity_id")) for r in kept],
        },
        columns=list(MEASUREMENT_COLUMNS),
    )
    # Keep the public measurement-table contract stable while carrying years
    # into molecule-level curation for optional chronological validation.
    frame.attrs["document_year_by_record"] = {
        str(r.get("activity_id")): r.get("document_year") for r in kept
    }
    frame.attrs["source_id_by_record"] = {
        str(r.get("activity_id")): source_id(r.get("src_id")) for r in kept
    }
    evidence_fields = (
        "assay_chembl_id",
        "assay_description",
        "document_chembl_id",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_type",
        "bao_label",
        "assay_variant_mutation",
        "confidence_score",
        "document_year",
        "src_id",
    )
    frame.attrs["context_by_record"] = {
        str(r.get("activity_id")): {
            **{field: r.get(field) for field in evidence_fields},
            "source_origin": source_name(r.get("src_id")),
        }
        for r in kept
    }
    return frame, steps


@lru_cache(maxsize=8192)
def _standardise_one(smiles: str, canonical_tautomer: bool) -> tuple[str, str | None] | str:
    """(standard SMILES, InChIKey), or an error message."""
    mol = parse_smiles(smiles)
    if mol is None:
        return "could not parse SMILES"
    try:
        mol = standardize(mol, canonical_tautomer=canonical_tautomer)
    except Exception as exc:  # noqa: BLE001 - one bad structure must not stop a run
        return f"standardisation failed: {type(exc).__name__}: {exc}"
    if mol is None or mol.GetNumAtoms() == 0:
        return "standardisation left no parent fragment"
    return Chem.MolToSmiles(mol), to_inchikey(mol)


def standardize_and_aggregate(
    measurements: pd.DataFrame,
    params: CurationParams,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[CurationStep]]:
    """Standardise, merge by structure, aggregate replicates.

    Returns (molecules, rejected, steps). ``molecules`` has every column of
    CURATED_COLUMNS except ``activity_class`` and ``group``.

    Each *distinct* SMILES is standardised once, with Sorbent's pieces called
    directly because ``process_record`` does not expose the tautomer switch.
    Merge key: InChIKey, or the standardised SMILES when the key is None.
    """
    unique_smiles = measurements["smiles"].unique()
    results = {}
    for index, smi in enumerate(unique_smiles, 1):
        results[smi] = _standardise_one(smi, params.canonical_tautomer)
        if progress is not None and (index % 100 == 0 or index == len(unique_smiles)):
            progress(index, len(unique_smiles))
    failed = {smi for smi, r in results.items() if isinstance(r, str)}
    ok = {smi: r for smi, r in results.items() if not isinstance(r, str)}

    bad = measurements[measurements["smiles"].isin(failed)]
    rejected = (
        bad[["molecule_id", "smiles"]]
        .drop_duplicates()
        .assign(reason=lambda f: [results[s] for s in f["smiles"]])
        .reset_index(drop=True)
    )

    good = measurements[~measurements["smiles"].isin(failed)].copy()
    years = measurements.attrs.get("document_year_by_record")
    if years is not None:
        good["_document_year"] = pd.to_numeric(good["record_id"].map(years), errors="coerce")
    good["std_smiles"] = [ok[s][0] for s in good["smiles"]]
    good["inchikey"] = [ok[s][1] for s in good["smiles"]]
    good["key"] = good["inchikey"].fillna(good["std_smiles"])
    steps = [
        CurationStep(
            "standardisation",
            len(measurements),
            len(good),
            good["molecule_id"].nunique(),
            f"{len(failed)} structures failed",
        )
    ]

    # A pandas group object per structure is costly on large target datasets.
    # Group row tuples once instead; the first measurement for the chosen
    # representative ID and all potency/year aggregation rules stay unchanged.
    positions = {name: index for index, name in enumerate(good.columns)}
    grouped: dict[str, list[tuple[Any, ...]]] = {}
    for record in good.itertuples(index=False, name=None):
        grouped.setdefault(record[positions["key"]], []).append(record)
    rows = []
    multi_id = 0
    for group in grouped.values():
        source_ids = sorted({record[positions["molecule_id"]] for record in group}, key=id_order)
        multi_id += len(source_ids) > 1
        first = next(
            record for record in group if record[positions["molecule_id"]] == source_ids[0]
        )
        values = [float(record[positions["pactivity"]]) for record in group]
        row = {
            "molecule_id": source_ids[0],
            "merged_ids": ";".join(source_ids),
            "smiles": first[positions["std_smiles"]],
            "inchikey": first[positions["inchikey"]],
            "pactivity": float(
                statistics.median(values)
                if params.aggregate == "median"
                else statistics.fmean(values)
            ),
            "n_measurements": len(values),
            "pactivity_range": float(max(values) - min(values)),
        }
        if "_document_year" in positions:
            known_years = [
                record[positions["_document_year"]]
                for record in group
                if pd.notna(record[positions["_document_year"]])
            ]
            # A compound becomes available at its first documented year.
            row["document_year"] = int(min(known_years)) if known_years else pd.NA
        rows.append(row)
    columns = [c for c in CURATED_COLUMNS if c not in ("activity_class", "group")]
    molecules = pd.DataFrame(rows, columns=columns)
    molecules = molecules.sort_values("molecule_id", key=lambda s: s.map(id_order)).reset_index(
        drop=True
    )
    steps.append(
        CurationStep(
            "aggregation",
            len(good),
            len(molecules),
            len(molecules),
            f"{multi_id} structures merged records from more than one source ID",
        )
    )

    if params.max_replicate_range is not None:
        before = len(molecules)
        molecules = molecules[molecules["pactivity_range"] <= params.max_replicate_range]
        molecules = molecules.reset_index(drop=True)
        steps.append(
            CurationStep(
                "replicate_range",
                before,
                len(molecules),
                len(molecules),
                f"replicates spanning more than {params.max_replicate_range} log units",
            )
        )
    return molecules, rejected, steps


def assign_classes(pactivity: pd.Series, scheme: ClassScheme) -> tuple[pd.Series, pd.Series]:
    """Return (activity_class, group) aligned to ``pactivity``'s index.

    A value takes the first label in ``scheme.bounds`` whose bound it *meets*
    (>=), so exactly 8.0 is "potent" under the default scheme; below every
    bound it gets ``scheme.floor_label``. Group is 1 if the label is in
    ``scheme.group1``, else 2. NaN is a ValueError.
    """
    if pactivity.isna().any():
        raise ValueError("potency is missing for some molecules; curation should have dropped them")

    def label(value: float) -> str:
        for name, bound in scheme.bounds:
            if value >= bound:
                return name
        return scheme.floor_label

    classes = pactivity.map(label)
    groups = classes.map(lambda c: 1 if c in scheme.group1 else 2)
    return classes, groups


def _classify(molecules: pd.DataFrame, scheme: ClassScheme) -> pd.DataFrame:
    classes, groups = assign_classes(molecules["pactivity"], scheme)
    out = molecules.assign(activity_class=classes, group=groups.astype(int))
    extras = [column for column in out.columns if column not in CURATED_COLUMNS]
    return out[[*CURATED_COLUMNS, *extras]]


def curate_chembl(
    records: Sequence[dict[str, Any]],
    params: RunParams,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> CurationResult:
    """filter_chembl_records -> standardize_and_aggregate -> assign_classes."""
    measurements, steps = filter_chembl_records(records, params.curation)
    if measurements.empty:
        raise ValueError(
            "no measurements remain after curation filters; widen endpoint, assay, "
            "or evidence-origin settings"
        )
    molecules, rejected, more = standardize_and_aggregate(
        measurements, params.curation, progress=progress
    )
    table = _classify(molecules, params.classes)
    return CurationResult(
        table, steps + more, rejected, _evidence_table(measurements, table), measurements
    )


def curate_table(measurements: pd.DataFrame, params: RunParams) -> CurationResult:
    """Same as curate_chembl but starting from MEASUREMENT_COLUMNS rows."""
    molecules, rejected, steps = standardize_and_aggregate(measurements, params.curation)
    table = _classify(molecules, params.classes)
    return CurationResult(
        table, steps, rejected, _evidence_table(measurements, table), measurements
    )


def _evidence_table(measurements: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Trace each curated molecule to the individual records that support it."""
    context = measurements.attrs.get("context_by_record", {})
    years = measurements.attrs.get("document_year_by_record", {})
    source_to_curated = {
        source_id: row.molecule_id
        for row in table.itertuples()
        for source_id in str(row.merged_ids).split(";")
    }
    rows = []
    for row in measurements.itertuples(index=False):
        curated_id = source_to_curated.get(str(row.molecule_id))
        if curated_id is None:
            continue
        record_id = str(row.record_id)
        record_context = context.get(record_id, {})
        assay_id = record_context.get("assay_chembl_id")
        document_id = record_context.get("document_chembl_id")
        rows.append(
            {
                "molecule_id": curated_id,
                "source_molecule_id": str(row.molecule_id),
                "record_id": record_id,
                "pactivity": float(row.pactivity),
                "smiles": str(row.smiles),
                "document_year": years.get(record_id),
                **record_context,
                "activity_url": (
                    f"https://www.ebi.ac.uk/chembl/api/data/activity/{record_id}.json"
                    if record_id.isdigit()
                    else None
                ),
                "assay_url": (
                    f"https://www.ebi.ac.uk/chembl/api/data/assay/{assay_id}.json"
                    if assay_id
                    else None
                ),
                "document_url": (
                    f"https://www.ebi.ac.uk/chembl/api/data/document/{document_id}.json"
                    if document_id
                    else None
                ),
            }
        )
    return pd.DataFrame(rows)


def time_safe_table(curation: CurationResult, params: RunParams) -> pd.DataFrame:
    """Partition measurements before aggregating potency for a future-data test.

    Previously published records define the training label. Later measurements
    for those structures cannot alter training labels or enter the held-out set.
    """
    measured = curation.measurements
    if measured.empty or "document_year_by_record" not in measured.attrs:
        raise ValueError("time split needs measurement-level document years")
    years = pd.to_numeric(
        measured["record_id"].map(measured.attrs["document_year_by_record"]), errors="coerce"
    )
    if years.isna().any():
        raise ValueError(
            f"time split needs a year for every retained measurement; {years.isna().sum()} missing"
        )
    train = measured[years <= params.model.time_cutoff].copy()
    later = measured[years > params.model.time_cutoff].copy()
    if train.empty or later.empty:
        raise ValueError("time split needs measurements on both sides of the cutoff")
    for part in (train, later):
        part.attrs = measured.attrs.copy()
    early_table = curate_table(train, params).table
    late_table = curate_table(later, params).table
    early_keys = set(early_table["inchikey"].fillna(early_table["smiles"]))
    late_keys = late_table["inchikey"].fillna(late_table["smiles"])
    novel = late_table[~late_keys.isin(early_keys)]
    if novel.empty:
        raise ValueError("no newly measured structures remain after the cutoff")
    return pd.concat([early_table, novel], ignore_index=True)


def source_safe_table(curation: CurationResult, params: RunParams) -> pd.DataFrame:
    """Hold out a ChEMBL origin without sharing structures or labels with training.

    Measurements are partitioned before aggregation. A molecule measured in
    both origins remains in training, but cannot be counted as external test.
    """
    source = params.model.source_test_id
    if source is None:
        raise ValueError("source split needs a held-out ChEMBL source ID")
    measured = curation.measurements
    if measured.empty or "source_id_by_record" not in measured.attrs:
        raise ValueError("source split needs measurement-level ChEMBL origin IDs")
    origins = measured["record_id"].map(measured.attrs["source_id_by_record"])
    if origins.isna().any():
        raise ValueError(
            f"source split needs an origin for every measurement; {origins.isna().sum()} missing"
        )
    train = measured[origins != source].copy()
    heldout = measured[origins == source].copy()
    if train.empty or heldout.empty:
        raise ValueError(
            f"source split for origin {source} leaves {len(train)} training and "
            f"{len(heldout)} held-out measurements"
        )
    for part in (train, heldout):
        part.attrs = measured.attrs.copy()
    train_table = curate_table(train, params).table
    test_table = curate_table(heldout, params).table
    train_keys = set(train_table["inchikey"].fillna(train_table["smiles"]))
    test_keys = test_table["inchikey"].fillna(test_table["smiles"])
    novel = test_table[~test_keys.isin(train_keys)].copy()
    if novel.empty:
        raise ValueError(f"source {source} has no compounds absent from the other origins")
    train_table["source_test"] = False
    novel["source_test"] = True
    return pd.concat([train_table, novel], ignore_index=True)

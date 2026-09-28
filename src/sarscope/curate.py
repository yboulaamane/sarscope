"""Raw activity records -> one standardised molecule, one potency, one class.

Three stages, each returning the step-by-step counts that become the curation
log in the report. Those counts are the point: the reference paper describes
curation in two sentences, and on BRAF the unstated choices alone move the
dataset between roughly 3,400 and 6,650 molecules. Every record that leaves the
pipeline leaves through a named step.

1. ``filter_chembl_records`` - record-level filters driven by CurationParams,
   then conversion to -log10(M). ChEMBL-specific.
2. ``standardize_and_aggregate`` - source-agnostic. Standardise with Sorbent,
   merge by structure, aggregate replicates.
3. ``assign_classes`` - potency bins and Group 1/2.

``curate_chembl`` and ``curate_table`` chain them.

Identity is decided by structure, not by ID. ChEMBL gives salt forms separate
molecule IDs (33 BRAF records sit under a salt whose parent is a different ID),
and the paper deduplicated on molecule ID, which keeps those as separate
compounds. Here the parent ID is used first and the standardised InChIKey
second, so a hydrochloride and its free base are one molecule.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from rdkit import Chem
from sorbent.chem.parse import parse_smiles, standardize, to_inchikey

from sarscope.params import ClassScheme, CurationParams, RunParams
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
        ("document_year", year_ok),
        ("structure", lambda r: bool(r.get("canonical_smiles"))),
        ("value", value_ok),
    ]

    kept = list(records)
    steps: list[CurationStep] = []
    for name, keep in filters:
        before = len(kept)
        kept = [r for r in kept if keep(r)]
        steps.append(CurationStep(name, before, len(kept), len({_parent_id(r) for r in kept})))

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
    return frame, steps


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
    measurements: pd.DataFrame, params: CurationParams
) -> tuple[pd.DataFrame, pd.DataFrame, list[CurationStep]]:
    """Standardise, merge by structure, aggregate replicates.

    Returns (molecules, rejected, steps). ``molecules`` has every column of
    CURATED_COLUMNS except ``activity_class`` and ``group``.

    Each *distinct* SMILES is standardised once, with Sorbent's pieces called
    directly because ``process_record`` does not expose the tautomer switch.
    Merge key: InChIKey, or the standardised SMILES when the key is None.
    """
    results = {
        smi: _standardise_one(smi, params.canonical_tautomer)
        for smi in measurements["smiles"].unique()
    }
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

    rows = []
    multi_id = 0
    for _, grp in good.groupby("key", sort=False):
        source_ids = sorted(set(grp["molecule_id"]), key=id_order)
        multi_id += len(source_ids) > 1
        first = grp[grp["molecule_id"] == source_ids[0]].iloc[0]
        values = grp["pactivity"]
        rows.append(
            {
                "molecule_id": source_ids[0],
                "merged_ids": ";".join(source_ids),
                "smiles": first["std_smiles"],
                "inchikey": first["inchikey"],
                "pactivity": float(
                    values.median() if params.aggregate == "median" else values.mean()
                ),
                "n_measurements": len(values),
                "pactivity_range": float(values.max() - values.min()),
            }
        )
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
    return out[list(CURATED_COLUMNS)]


def curate_chembl(records: Sequence[dict[str, Any]], params: RunParams) -> CurationResult:
    """filter_chembl_records -> standardize_and_aggregate -> assign_classes."""
    measurements, steps = filter_chembl_records(records, params.curation)
    molecules, rejected, more = standardize_and_aggregate(measurements, params.curation)
    return CurationResult(_classify(molecules, params.classes), steps + more, rejected)


def curate_table(measurements: pd.DataFrame, params: RunParams) -> CurationResult:
    """Same as curate_chembl but starting from MEASUREMENT_COLUMNS rows."""
    molecules, rejected, steps = standardize_and_aggregate(measurements, params.curation)
    return CurationResult(_classify(molecules, params.classes), steps, rejected)

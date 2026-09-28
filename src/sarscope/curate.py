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

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from sarscope.params import ClassScheme, CurationParams, RunParams

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

    Field types as the API returns them: ``standard_value`` and
    ``pchembl_value`` are *strings*; ``potential_duplicate`` and
    ``document_year`` are ints; absent values are None.

    ``molecule_id`` is ``parent_molecule_chembl_id`` (falling back to
    ``molecule_chembl_id``). ``record_id`` is ``str(activity_id)``.
    ``pactivity`` is ``units.to_pactivity(float(standard_value), standard_units)``.
    """
    raise NotImplementedError


def standardize_and_aggregate(
    measurements: pd.DataFrame, params: CurationParams
) -> tuple[pd.DataFrame, pd.DataFrame, list[CurationStep]]:
    """Standardise, merge by structure, aggregate replicates.

    Returns (molecules, rejected, steps). ``molecules`` has every column of
    CURATED_COLUMNS except ``activity_class`` and ``group``.

    Standardise each *distinct* SMILES once (not once per measurement) using
    Sorbent's pieces directly, because ``sorbent.chem.parse.process_record``
    does not expose the tautomer switch:

        mol = parse_smiles(smiles)                      # None -> rejected
        mol = standardize(mol, canonical_tautomer=params.canonical_tautomer)
        smiles = Chem.MolToSmiles(mol); key = to_inchikey(mol)

    Merge key: InChIKey, or the standardised SMILES when the key is None.
    Aggregate ``pactivity`` with params.aggregate. Drop molecules whose
    ``pactivity_range`` exceeds params.max_replicate_range, if set.

    Steps logged: "standardisation" (records whose structure failed) and
    "aggregation" (records -> molecules; ``detail`` states how many merges
    joined *different* source IDs), plus "replicate_range" when enabled.
    """
    raise NotImplementedError


def assign_classes(pactivity: pd.Series, scheme: ClassScheme) -> tuple[pd.Series, pd.Series]:
    """Return (activity_class, group) aligned to ``pactivity``'s index.

    A value takes the first label in ``scheme.bounds`` whose bound it *meets*
    (>=), so exactly 8.0 is "potent" under the default scheme; below every
    bound it gets ``scheme.floor_label``. Group is 1 if the label is in
    ``scheme.group1``, else 2. NaN is a ValueError: a molecule without a
    potency should never have reached this point.
    """
    raise NotImplementedError


def curate_chembl(records: Sequence[dict[str, Any]], params: RunParams) -> CurationResult:
    """filter_chembl_records -> standardize_and_aggregate -> assign_classes."""
    raise NotImplementedError


def curate_table(measurements: pd.DataFrame, params: RunParams) -> CurationResult:
    """Same as curate_chembl but starting from MEASUREMENT_COLUMNS rows."""
    raise NotImplementedError

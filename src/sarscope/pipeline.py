"""Source -> curated table -> every analysis -> RunResults. No I/O except the source.

``analyse`` is source-agnostic and is where the order of operations lives:

    1. add_descriptors, add_scaffolds
    2. describe_groups, property_pca                       (Tables 2, 3)
    3. diversity_table, enrichment_table                   (Table 4, EF)
    4. per fingerprint in params.landscape.fingerprints:
       bit_vectors -> sas_map; then consensus              (Figs. 8, 9)
    5. fingerprint_matrix(params.model.features.fingerprint) -> evaluate
                                                            (Table 6)
    6. pca_bounding_box on the leak-free train/test split  (Fig. 10)
       using the feature filter fitted on the training rows

A dataset with a single class, or fewer than ``cv_folds`` molecules in the
smallest class, cannot be modelled: skip step 5-6, record why in
``RunResults.skipped``, and still return everything else. Losing a whole
report because one step is impossible is worse than a report with a hole in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sarscope.analysis.domain import DomainResult
from sarscope.analysis.landscape import SasResult
from sarscope.analysis.model import ModelResult
from sarscope.analysis.profile import GroupProfile, PcaResult
from sarscope.curate import CurationResult
from sarscope.params import RunParams
from sarscope.sources.chembl import ChemblClient


@dataclass
class RunResults:
    params: RunParams
    curation: CurationResult
    #: Curated table plus descriptor and scaffold columns.
    table: pd.DataFrame
    profile: GroupProfile
    pca: PcaResult
    diversity: pd.DataFrame
    enrichment: pd.DataFrame
    landscapes: dict[str, SasResult]
    consensus_cliffs: pd.DataFrame
    consensus_generators: list[str]
    models: ModelResult | None
    domain: DomainResult | None
    #: step name -> reason, for anything that could not run.
    skipped: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def analyse(curation: CurationResult, params: RunParams) -> RunResults:
    raise NotImplementedError


def run_chembl(target_id: str, params: RunParams, client: ChemblClient) -> RunResults:
    """Fetch (via the client's cache), curate_chembl, analyse, attach provenance."""
    raise NotImplementedError


def run_table(path: Path, params: RunParams, **read_kwargs: Any) -> RunResults:
    """read_activity_table, curate_table, analyse, attach provenance."""
    raise NotImplementedError

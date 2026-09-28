"""Source -> curated table -> every analysis -> RunResults. No I/O except the source.

``analyse`` is source-agnostic and is where the order of operations lives:

    1. add_descriptors, add_scaffolds
    2. describe_groups, property_pca
    3. diversity_table, enrichment_table
    4. decompose_top_scaffolds
    5. per fingerprint in params.landscape.fingerprints:
       bit_vectors -> sas_map; then consensus
    6. fingerprint_matrix(params.model.features.fingerprint) -> evaluate
    7. pca_bounding_box on the leak-free train/test split, using the feature
       filter fitted on the training rows

A dataset with a single class, or fewer than ``cv_folds`` molecules in the
smallest class, cannot be modelled: skip steps 6-7, record why in
``RunResults.skipped``, and still return everything else. Losing a whole
report because one step is impossible is worse than a report with a hole in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from sarscope import provenance
from sarscope.analysis.descriptors import add_descriptors
from sarscope.analysis.domain import DomainResult, pca_bounding_box
from sarscope.analysis.features import VarianceCorrelationFilter, bit_vectors, fingerprint_matrix
from sarscope.analysis.landscape import SasResult, consensus, sas_map
from sarscope.analysis.model import ModelResult, evaluate
from sarscope.analysis.profile import GroupProfile, PcaResult, describe_groups, property_pca
from sarscope.analysis.rgroups import ScaffoldSar, decompose_top_scaffolds
from sarscope.analysis.scaffolds import add_scaffolds, diversity_table, enrichment_table
from sarscope.curate import CurationResult, curate_chembl, curate_table
from sarscope.params import RunParams
from sarscope.sources.chembl import ChemblClient
from sarscope.sources.table import read_activity_table


@dataclass
class RunResults:
    params: RunParams
    curation: CurationResult
    #: Curated table plus descriptor and scaffold columns.
    table: pd.DataFrame
    profile: GroupProfile | None
    pca: PcaResult
    diversity: pd.DataFrame
    enrichment: pd.DataFrame
    landscapes: dict[str, SasResult]
    consensus_cliffs: pd.DataFrame
    consensus_generators: list[str]
    models: ModelResult | None
    domain: DomainResult | None
    #: Per-scaffold R-group decomposition, largest series first.
    rgroups: list[ScaffoldSar] = field(default_factory=list)
    #: step name -> reason, for anything that could not run.
    skipped: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def analyse(curation: CurationResult, params: RunParams) -> RunResults:
    table = add_scaffolds(add_descriptors(curation.table))
    skipped: dict[str, str] = {}

    # A set that is all potent, or all inactive, has no two groups to compare.
    # That is a real dataset shape (a focused in-house series often is), so the
    # comparison is skipped and everything that does not need it still runs.
    groups = sorted(table["group"].unique())
    profile: GroupProfile | None = None
    if len(groups) == 2:
        profile = describe_groups(table)
    else:
        skipped["profile"] = (
            f"every molecule is in Group {groups[0]}, so there are no two groups to "
            "compare. Descriptor values are still in the curated dataset."
        )
    pca = property_pca(table)
    diversity = diversity_table(table)
    enrichment = enrichment_table(table)
    rgroups = decompose_top_scaffolds(table, enrichment)

    landscapes: dict[str, SasResult] = {}
    for name in params.landscape.fingerprints:
        fps = bit_vectors(table["smiles"].tolist(), name, ecfp_bits=params.model.features.ecfp_bits)
        landscapes[name] = sas_map(
            fps,
            table["pactivity"].tolist(),
            table["molecule_id"].tolist(),
            params.landscape,
            fingerprint_name=name,
        )
    consensus_cliffs, consensus_generators = consensus(landscapes, params.landscape.generator_sd)

    models: ModelResult | None = None
    domain: DomainResult | None = None
    reason = _modelling_blocked(table, params)
    if reason:
        skipped["model"] = reason
        skipped["domain"] = reason
    else:
        X = fingerprint_matrix(
            table["smiles"].tolist(),
            params.model.features.fingerprint,
            ecfp_bits=params.model.features.ecfp_bits,
        )
        models = evaluate(
            X, table["activity_class"].tolist(), table["murcko"].tolist(), params.model
        )
        feats = params.model.features
        filt = VarianceCorrelationFilter(feats.variance_threshold, feats.correlation_threshold).fit(
            X[models.train_index]
        )
        domain = pca_bounding_box(
            filt.transform(X[models.train_index]), filt.transform(X[models.test_index])
        )

    return RunResults(
        params=params,
        curation=curation,
        table=table,
        profile=profile,
        pca=pca,
        diversity=diversity,
        enrichment=enrichment,
        landscapes=landscapes,
        consensus_cliffs=consensus_cliffs,
        consensus_generators=consensus_generators,
        models=models,
        domain=domain,
        rgroups=rgroups,
        skipped=skipped,
    )


def _modelling_blocked(table: pd.DataFrame, params: RunParams) -> str:
    """Why the model step cannot run, or "" when it can.

    Checked before any fitting so a small dataset produces a report with a
    stated gap, rather than a traceback three minutes in. Both limits bite on
    real data: a focused in-house set often has few scaffolds, and a scaffold
    split cannot make more folds than there are scaffolds.
    """
    model = params.model
    counts = table["activity_class"].value_counts()
    if len(counts) < 2:
        return (
            f"every molecule falls in one activity class ({counts.index[0]}), "
            "so there is nothing to classify. Widen the class bounds, or check "
            "that the potency column is on the scale you think it is."
        )
    if counts.min() < model.cv_folds:
        return (
            f"the smallest class has {counts.min()} molecules, fewer than the "
            f"{model.cv_folds} cross-validation folds. Lower cv_folds, or widen "
            "the class bounds so the rarest class is better populated."
        )
    if model.split == "scaffold":
        # Acyclic molecules are each their own group, so they only ever help.
        groups = int(table["murcko"].nunique() + table["murcko"].isna().sum())
        outer = max(2, round(1 / model.test_fraction))
        # The outer split takes its share of scaffolds first, so cross-validation
        # runs on roughly (1 - test_fraction) of them and needs cv_folds of those.
        needed = max(outer, math.ceil(model.cv_folds / (1 - model.test_fraction)))
        if groups < needed:
            return (
                f"a scaffold split with {model.cv_folds} folds needs about {needed} distinct "
                f"scaffolds and this dataset has {groups}. Lower cv_folds, or pass "
                "--split random (which will flatter the model, because one series can "
                "then sit on both sides of the split)."
            )
    return ""


def run_chembl(target_id: str, params: RunParams, client: ChemblClient) -> RunResults:
    """Fetch (via the client's cache), curate_chembl, analyse, attach provenance."""
    target = client.target(target_id)
    records = client.activities(target_id, params.curation.standard_types)
    results = analyse(curate_chembl(records, params), params)
    results.provenance = provenance.collect(
        params, provenance.chembl_source(target, client.release, len(records))
    )
    return results


def run_table(path: Path, params: RunParams, **read_kwargs: Any) -> RunResults:
    """read_activity_table, curate_table, analyse, attach provenance."""
    measurements = read_activity_table(path, **read_kwargs)
    results = analyse(curate_table(measurements, params), params)
    results.provenance = provenance.collect(params, provenance.table_source(path))
    return results

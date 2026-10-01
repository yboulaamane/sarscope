"""SARscope in the browser. Run locally with ``streamlit run streamlit_app.py``.

Analyses run in the browser process, so the heavy steps are opt-in and
pairwise landscape analysis has a dataset cap. A free hosting tier has one
shared core and a memory ceiling. The CLI (``sarscope run``) has no such cap
and writes the full report folder.
"""

from __future__ import annotations

import collections
import hashlib
import io
import json
import sys
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, cast

# Streamlit Community Cloud clones the repo and runs this file from the root;
# it installs requirements.txt but not this project, and the package lives
# under src/. Put src/ on the path so the import below works there as it does
# in a local editable install, where this line is a harmless no-op.
_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from sarscope import __version__, provenance  # noqa: E402
from sarscope.__main__ import FETCH_SUMMARY_FIELDS, default_cache_dir  # noqa: E402
from sarscope.analysis.chemical_space import StructuralSpace, ecfp4_umap  # noqa: E402
from sarscope.analysis.descriptors import add_descriptors  # noqa: E402
from sarscope.analysis.diagnostics import (  # noqa: E402
    bootstrap_regression,
    novelty_summary,
    regression_summary,
)
from sarscope.analysis.domain import DomainResult, pca_bounding_box  # noqa: E402
from sarscope.analysis.explain import (  # noqa: E402
    DescriptorExplanation,
    explain_descriptor_model,
)
from sarscope.analysis.features import (  # noqa: E402
    bit_vectors,
    feature_filter,
    fingerprint_matrix,
)
from sarscope.analysis.landscape import (  # noqa: E402
    SasResult,
    cliff_generators,
    consensus,
    sas_map,
)
from sarscope.analysis.mmp import matched_molecular_pairs, summarise_transformations  # noqa: E402
from sarscope.analysis.model import ModelResult, evaluate  # noqa: E402
from sarscope.analysis.profile import (  # noqa: E402
    GroupProfile,
    PcaResult,
    describe_groups,
    property_pca,
)
from sarscope.analysis.regression import (  # noqa: E402
    RegressionResult,
    evaluate_regression,
    fit_deployment_model,
)
from sarscope.analysis.representations import (  # noqa: E402
    ALL_2D,
    DESCRIPTOR_PRESETS,
    model_feature_names,
    model_matrix,
    resolved_features,
)
from sarscope.analysis.rgroups import ScaffoldSar, decompose_top_scaffolds  # noqa: E402
from sarscope.analysis.scaffolds import (  # noqa: E402
    ENRICHMENT_COLUMNS,
    add_scaffolds,
    diversity_table,
    enrichment_table,
)
from sarscope.compare import ComparisonResult, run_compare  # noqa: E402
from sarscope.curate import (  # noqa: E402
    CurationResult,
    curate_chembl,
    source_safe_table,
    time_safe_table,
)
from sarscope.depict import to_svg, unavailable_reason  # noqa: E402
from sarscope.ml_selection import select_ml_table  # noqa: E402
from sarscope.params import (  # noqa: E402
    ClassScheme,
    CurationParams,
    FeatureParams,
    FingerprintName,
    LandscapeParams,
    ModelParams,
    RunParams,
    SplitStrategy,
)
from sarscope.pipeline import (  # noqa: E402
    RunResults,
    _cliff_model_errors,
    _modelling_blocked,
    regression_error_profile,
)
from sarscope.predict import PredictionBundle, similarity_domain_threshold  # noqa: E402
from sarscope.prioritise import diverse_shortlist, predict_candidates  # noqa: E402
from sarscope.report import write_report  # noqa: E402
from sarscope.screen import run_screen_snapshot  # noqa: E402
from sarscope.sources.chembl import (  # noqa: E402
    ChemblClient,
    ChemblError,
    normalise_target_id,
)
from sarscope.sources.opentargets import (  # noqa: E402
    OpenTargetsError,
    disease_targets,
    search_diseases,
)
from sarscope.sources.origins import (  # noqa: E402
    SOURCE_NAMES,
    source_disagreement,
    source_summary,
)
from sarscope.sources.pubchem import PubChemClient, PubChemError  # noqa: E402

#: Categorical slots of the reference palette. Group 1 / Group 2 keep these
#: hues everywhere in the app, so colour follows the entity, never the rank.
TEAL, PURPLE, ORANGE, RED = "#52dfb6", "#c59bff", "#ffca76", "#ff8194"
GROUP_COLORS = [TEAL, ORANGE]
GROUP_LABELS = {1: "Potent + active", 2: "Intermediate + inactive"}
CLASS_COLORS = {
    "potent": TEAL,
    "active": PURPLE,
    "intermediate": ORANGE,
    "inactive": RED,
}

#: Above this many molecules, the all-pairs landscape gets slow in a shared
#: process. The CLI has no cap.
LANDSCAPE_WARN = 4000
COMPARE_RECORD_LIMIT = 20_000

ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]

#: What each summarised field means, and the curation decision it drives.
FIELD_NOTES: dict[str, tuple[str, str]] = {
    "standard_relation": (
        "Relation",
        '"=" is a measured value. "<" and ">" are bounds (censored), dropped by default.',
    ),
    "standard_units": ("Units", "Only molar units convert to the -log10(M) scale."),
    "assay_type": ("Assay type", "B binding, F functional, A ADMET. Binding kept by default."),
    "src_id": (
        "Original data source",
        "1 literature, 7 PubChem BioAssay, 37 BindingDB; these are ChEMBL integrations.",
    ),
    "assay_variant_mutation": (
        "Protein variant",
        "(none) means no mutation annotation, not confirmed wild-type protein. "
        "Only unannotated records are kept by default; annotated variants can be selected "
        "using their exact annotation in the sidebar.",
    ),
    "potential_duplicate": (
        "ChEMBL duplicate flag",
        "1 marks a likely re-report of an earlier measurement; dropped by default.",
    ),
    "data_validity_comment": (
        "Validity flag",
        "ChEMBL's warnings about suspect values; flagged records dropped by default.",
    ),
    "bao_label": (
        "Assay format",
        "Cell-based and enzyme IC50s are different measurements. Reported, not filtered.",
    ),
}

FAST_ALGORITHMS = [
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "nearest_neighbors",
    "neural_net",
]


@dataclass
class CurationStage:
    params: RunParams
    curation: CurationResult
    table: pd.DataFrame
    provenance: dict[str, Any]
    skipped: dict[str, str] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class PropertyStage:
    table: pd.DataFrame
    profile: GroupProfile | None
    pca: PcaResult
    skipped: dict[str, str] = field(default_factory=dict)


@dataclass
class ScaffoldStage:
    table: pd.DataFrame
    diversity: pd.DataFrame
    enrichment: pd.DataFrame
    rgroups: list[ScaffoldSar] = field(default_factory=list)
    matched_pairs: pd.DataFrame = field(default_factory=pd.DataFrame)
    transformation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class LandscapeStage:
    params: RunParams
    table: pd.DataFrame
    landscapes: dict[str, SasResult]
    consensus_cliffs: pd.DataFrame
    consensus_generators: list[str]


@dataclass
class MlStage:
    params: RunParams
    table: pd.DataFrame
    models: ModelResult | None
    regression: RegressionResult | None
    domain: DomainResult | None
    model_test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    cliff_model_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    prediction_bundle: PredictionBundle | None = None
    skipped: dict[str, str] = field(default_factory=dict)
    regression_test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    drop_intermediate: bool = False


# -- data ---------------------------------------------------------------------


def client() -> ChemblClient:
    return ChemblClient(cache_dir=default_cache_dir())


@st.cache_data(ttl=86_400, show_spinner=False)
def lookup(target_id: str, types: tuple[str, ...]) -> tuple[dict[str, Any], str, int]:
    with client() as c:
        return c.target(target_id), c.release, c.count_activities(target_id, types)


@st.cache_data(ttl=86_400, show_spinner=False)
def _fetch_cached(target_id: str, types: tuple[str, ...]) -> list[dict[str, Any]]:
    with client() as c:
        return c.activities(target_id, types)


def fetch(
    target_id: str,
    types: tuple[str, ...],
    _progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    if _progress is None:
        return _fetch_cached(target_id, types)
    # Progress updates refer to UI blocks outside this function. Streamlit
    # must not record/replay them from a cached call; ChEMBL's release-keyed
    # disk cache still avoids downloading records again during recuration.
    with client() as c:
        return c.activities(target_id, types, progress=_progress)


@st.cache_data(ttl=86_400, max_entries=16, show_spinner=False)
def assay_confidences(target_id: str) -> dict[str, int]:
    with client() as c:
        return c.assay_confidences(target_id)


def compare_targets(
    target_a: str, target_b: str, params: RunParams, second_assay_id: str = ""
) -> ComparisonResult:
    with client() as chembl:
        count = chembl.count_activities(target_b, params.curation.standard_types)
        if count > COMPARE_RECORD_LIMIT:
            raise ValueError(
                f"{target_b} has {count:,} activity records of the selected types. "
                f"The browser comparison limit is {COMPARE_RECORD_LIMIT:,}; "
                "use the CLI for this target pair."
            )
        return run_compare(
            target_a,
            target_b,
            params,
            chembl,
            secondary_assay_ids=(second_assay_id,) if second_assay_id else (),
        )


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def search_target_names(query: str, human_only: bool) -> tuple[list[dict[str, Any]], int]:
    with ChemblClient(timeout=20, retries=1) as c:
        return c.search_targets(query, organism="Homo sapiens" if human_only else None)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def search_disease_names(query: str) -> dict[str, Any]:
    return search_diseases(query)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def load_disease_targets(disease_id: str) -> dict[str, Any]:
    return disease_targets(disease_id)


@st.cache_data(ttl=3600, max_entries=128, show_spinner=False)
def map_disease_proteins(accessions: tuple[str, ...]) -> tuple[list[dict[str, Any]], int]:
    with ChemblClient(timeout=20, retries=1) as c:
        return c.targets_for_accessions(accessions)


def target_search_controls() -> tuple[str | None, bool, dict[str, Any]]:
    """Resolve an explicit user choice; network calls only follow button clicks."""
    mode = st.radio("Find a target by", ["ChEMBL ID", "Gene / protein", "Disease"], horizontal=True)
    context: dict[str, Any] = {"mode": mode}
    if mode == "ChEMBL ID":
        left, right = st.columns([3, 1], vertical_alignment="bottom")
        raw_id = left.text_input(
            "ChEMBL target ID", value="CHEMBL5145", help="e.g. CHEMBL5145 (BRAF) or just 5145"
        )
        check = right.button("Check target", type="primary", width="stretch")
        try:
            return normalise_target_id(raw_id), check, context
        except ValueError as exc:
            if raw_id.strip():
                st.error(str(exc))
            return None, False, context

    human_only = True
    if mode == "Gene / protein":
        human_only = st.checkbox("Human targets only", value=True)
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    query = left.text_input(
        "Gene or protein name" if mode == "Gene / protein" else "Disease or phenotype",
        placeholder="BRAF, EGFR, acetylcholinesterase" if mode == "Gene / protein" else "melanoma",
        key=f"search_query_{mode}",
        max_chars=200,
    ).strip()
    search_key = (mode, query, human_only)
    if right.button("Search", type="primary", width="stretch", disabled=not query):
        st.session_state.pop("discovery_results", None)
        st.session_state.pop("target_lookup", None)
        clear_workflow()
        try:
            with st.spinner(
                "Searching ChEMBL…" if mode == "Gene / protein" else "Searching diseases…"
            ):
                results = (
                    search_target_names(query, human_only)
                    if mode == "Gene / protein"
                    else search_disease_names(query)
                )
            st.session_state["discovery_results"] = (search_key, results)
        except (ChemblError, OpenTargetsError, ValueError) as exc:
            st.error(str(exc))
    found = st.session_state.get("discovery_results")
    if found is None or found[0] != search_key:
        st.info("Search, review the matches, then choose a target to check.")
        return None, False, context
    context.update(query=query, human_only=human_only)
    if mode == "Gene / protein":
        candidates, total = found[1]
    else:
        diseases = found[1]
        hits = {hit["id"]: hit for hit in diseases["hits"]}
        st.caption(
            f"Showing {len(hits)} of {diseases['total']:,} disease matches. Refine broad searches."
        )
        if not hits:
            st.info("No disease matches. Try another name or a more specific term.")
            return None, False, context
        disease_id = st.selectbox(
            "Choose the disease term",
            list(hits),
            index=None,
            format_func=lambda value: f"{hits[value]['name']} · {value}",
            key=f"disease_choice_{query}",
        )
        if disease_id is None:
            return None, False, context
        if st.button("Find associated genes"):
            st.session_state.pop("disease_associations", None)
            try:
                with st.spinner("Loading direct disease associations from Open Targets…"):
                    st.session_state["disease_associations"] = load_disease_targets(disease_id)
            except (OpenTargetsError, ValueError) as exc:
                st.error(str(exc))
        associations = st.session_state.get("disease_associations")
        if associations is None or associations["disease_id"] != disease_id:
            return None, False, context
        st.caption(
            f"Open Targets · {associations['disease_name']} · {disease_id}. "
            f"Showing {len(associations['rows'])} of {associations['total']:,} "
            "direct associations; "
            "evidence from descendant disease terms is excluded. Scores summarize evidence, "
            "not the probability of treatment success. ChEMBL assay coverage is checked separately."
        )
        st.markdown(
            f"[Inspect disease evidence in Open Targets](https://platform.opentargets.org/disease/{disease_id}/associations)"
        )
        genes = {row["gene_id"]: row for row in associations["rows"]}
        if not genes:
            st.info("No direct associations for this term. Try a more specific disease.")
            return None, False, context
        st.dataframe(
            pd.DataFrame(associations["rows"])[["symbol", "name", "gene_id", "score"]],
            hide_index=True,
            width="stretch",
        )
        gene_id = st.selectbox(
            "Choose a gene to resolve in ChEMBL",
            list(genes),
            index=None,
            format_func=lambda value: (
                f"{genes[value]['symbol']} · {genes[value]['name']} · {value}"
            ),
            key=f"gene_choice_{disease_id}",
        )
        if gene_id is None:
            return None, False, context
        gene = genes[gene_id]
        if not gene["accessions"]:
            st.info(
                "No reviewed UniProt accession is available for this gene. Try gene/protein search."
            )
            return None, False, context
        mapping_key = (disease_id, gene_id, tuple(gene["accessions"]))
        if st.button("Find ChEMBL targets for this gene"):
            st.session_state.pop("disease_mapping", None)
            try:
                with st.spinner("Matching reviewed UniProt accessions to human single proteins…"):
                    mapped = map_disease_proteins(tuple(gene["accessions"]))
                st.session_state["disease_mapping"] = (mapping_key, mapped)
            except (ChemblError, ValueError) as exc:
                st.error(str(exc))
        mapping = st.session_state.get("disease_mapping")
        if mapping is None or mapping[0] != mapping_key:
            return None, False, context
        candidates, total = mapping[1]
        context.update(
            {key: value for key, value in associations.items() if key != "rows"},
            gene=gene,
        )
    st.caption(
        f"Showing {len(candidates)} of {total:,} ChEMBL targets. Confirm species and target type."
    )
    if not candidates:
        st.info("No matching ChEMBL target. Try a different name, gene, or species filter.")
        return None, False, context
    targets = {row["target_chembl_id"]: row for row in candidates}
    target_id = st.selectbox(
        "Choose a ChEMBL target",
        list(targets),
        index=None,
        format_func=lambda value: (
            f"{targets[value]['pref_name']} · {value} · {targets[value]['organism']} · "
            f"{targets[value]['target_type']}"
        ),
        key=f"target_choice_{mode}_{query}_{human_only}_{tuple(targets)}",
    )
    check = st.button("Check target", type="primary", disabled=target_id is None)
    return target_id, check, context


def run_curation_stage(
    target_id: str,
    types: tuple[str, ...],
    params: RunParams,
    release: str,
    target_name: str,
    organism: str,
    *,
    on_status: Callable[[str], None] | None = None,
    on_download_progress: Callable[[int, int], None] | None = None,
    on_standardize_progress: Callable[[int, int], None] | None = None,
) -> CurationStage:
    """Download and curate only; every later analysis is explicitly opt-in."""
    started = perf_counter()
    if on_status is not None:
        on_status("Loading activity records from ChEMBL or the local cache…")
    records = fetch(target_id, types, _progress=on_download_progress)
    load_seconds = perf_counter() - started
    if on_status is not None:
        on_status(f"Loaded {len(records):,} raw records in {load_seconds:.1f}s.")
    assay_seconds = 0.0
    if params.curation.min_confidence_score is not None:
        if on_status is not None:
            on_status("Loading assay-target confidence scores…")
        assay_started = perf_counter()
        scores = assay_confidences(target_id)
        records = [
            {**record, "confidence_score": scores.get(str(record.get("assay_chembl_id")))}
            for record in records
        ]
        assay_seconds = perf_counter() - assay_started
    if on_status is not None:
        on_status("Filtering measurements and standardizing distinct structures…")
    if on_standardize_progress is not None:
        on_standardize_progress(0, 0)
    curation_started = perf_counter()
    curation = curate_chembl(records, params, progress=on_standardize_progress)
    curation_seconds = perf_counter() - curation_started
    if on_status is not None:
        on_status(f"Curated {len(curation.table):,} compounds in {curation_seconds:.1f}s.")
    target = {
        "target_chembl_id": target_id,
        "pref_name": target_name,
        "organism": organism,
    }
    record = provenance.collect(params, provenance.chembl_source(target, release, len(records)))
    return CurationStage(
        params,
        curation,
        curation.table,
        record,
        timings={
            "load_records": load_seconds,
            "assay_confidence": assay_seconds,
            "curation": curation_seconds,
        },
    )


def run_property_stage(curation: CurationResult) -> PropertyStage:
    table = add_descriptors(curation.table)
    skipped: dict[str, str] = {}
    profile: GroupProfile | None = None
    groups = sorted(table["group"].unique())
    if len(groups) == 2:
        profile = describe_groups(table)
    else:
        skipped["profile"] = "Only one activity group is present, so no group comparison ran."
    return PropertyStage(table, profile, property_pca(table), skipped)


@st.cache_data(max_entries=4, show_spinner=False)
def run_structural_space(table: pd.DataFrame, n_neighbors: int, min_dist: float) -> StructuralSpace:
    return ecfp4_umap(table, n_neighbors=n_neighbors, min_dist=min_dist)


def descriptor_controls(prefix: str) -> tuple[str, ...]:
    preset = st.selectbox(
        "Descriptor preset", list(DESCRIPTOR_PRESETS), index=1, key=f"{prefix}_preset"
    )
    chosen = st.multiselect(
        "Descriptors to include",
        list(ALL_2D),
        default=list(DESCRIPTOR_PRESETS[preset]),
        key=f"{prefix}_descriptors_{preset}",
        help="Choose named 2D descriptors. No 3D conformers are generated.",
    )
    st.caption(
        f"{len(chosen)} selected / {len(ALL_2D)} available. Missing values are median-imputed "
        "using training rows only. Constants and correlated columns are filtered inside ML folds. "
        "Selecting descriptors after inspecting test performance would bias that test."
    )
    return tuple(chosen)


def run_scaffold_stage(curation: CurationResult) -> ScaffoldStage:
    table = add_scaffolds(curation.table)
    diversity = diversity_table(table)
    try:
        enrichment = enrichment_table(table)
    except ValueError:
        enrichment = pd.DataFrame(columns=list(ENRICHMENT_COLUMNS))
    return ScaffoldStage(table, diversity, enrichment)


def run_landscape_stage(curation: CurationResult, params: RunParams) -> LandscapeStage:
    table = curation.table
    landscapes: dict[str, SasResult] = {}
    for name in params.landscape.fingerprints:
        vectors = bit_vectors(
            table["smiles"].tolist(), name, ecfp_bits=params.model.features.ecfp_bits
        )
        landscapes[name] = sas_map(
            vectors,
            table["pactivity"].tolist(),
            table["molecule_id"].tolist(),
            params.landscape,
            fingerprint_name=name,
        )
    pairs, generators = consensus(landscapes, params.landscape.generator_sd)
    return LandscapeStage(params, table, landscapes, pairs, generators)


def run_ml_stage(
    scaffold: ScaffoldStage,
    curation: CurationResult,
    params: RunParams,
    *,
    classification: bool,
    regression: bool,
    landscape: LandscapeStage | None,
    drop_intermediate: bool = False,
) -> MlStage:
    table = scaffold.table
    split_error = ""
    if params.model.split in ("time", "source"):
        try:
            safe = (
                time_safe_table(curation, params)
                if params.model.split == "time"
                else source_safe_table(curation, params)
            )
            table = add_scaffolds(safe)
        except ValueError as exc:
            split_error = str(exc)
    try:
        table = select_ml_table(table, drop_intermediate=drop_intermediate)
    except ValueError as exc:
        split_error = str(exc)
        table = table.iloc[0:0].copy()
    reason = split_error or _modelling_blocked(table, params)
    if reason:
        return MlStage(
            params,
            table,
            None,
            None,
            None,
            skipped={"model": reason, "domain": reason},
            drop_intermediate=drop_intermediate,
        )
    features = params.model.features
    X = model_matrix(table["smiles"].tolist(), features)
    domain_X = fingerprint_matrix(
        table["smiles"].tolist(), features.fingerprint, ecfp_bits=features.ecfp_bits
    )
    years = table["document_year"].tolist()
    source_test = table["source_test"].tolist() if "source_test" in table else None
    models = (
        evaluate(
            X,
            table["activity_class"].tolist(),
            table["murcko"].tolist(),
            params.model,
            years=years,
            source_test=source_test,
        )
        if classification
        else None
    )
    regression_result = (
        evaluate_regression(
            X,
            table["pactivity"].tolist(),
            table["murcko"].tolist(),
            table["activity_class"].tolist(),
            params.model,
            years=years,
            source_test=source_test,
        )
        if regression
        else None
    )
    selected = regression_result or models
    assert selected is not None
    filt = feature_filter(features).fit(X[selected.train_index])
    domain = pca_bounding_box(
        filt.transform(X[selected.train_index]), filt.transform(X[selected.test_index])
    )

    test_predictions = pd.DataFrame()
    cliff_performance = pd.DataFrame()
    regression_test_predictions = (
        regression_error_profile(table, domain_X, regression_result)
        if regression_result is not None
        else pd.DataFrame()
    )
    if regression_result is not None and landscape is not None:
        test_predictions, cliff_performance = _cliff_model_errors(
            table, landscape.landscapes, models, regression_result
        )

    bundle: PredictionBundle | None = None
    if regression_result is not None:
        deploy_filter, deploy_model = fit_deployment_model(
            X, table["pactivity"].tolist(), regression_result.best_algorithm, params.model
        )
        bundle = PredictionBundle(
            algorithm=regression_result.best_algorithm,
            features=resolved_features(features),
            feature_filter=deploy_filter,
            estimator=deploy_model,
            train_fingerprints=domain_X,
            similarity_threshold=similarity_domain_threshold(domain_X),
            train_ids=table["molecule_id"].astype(str).tolist(),
            train_smiles=table["smiles"].astype(str).tolist(),
            train_pactivity=table["pactivity"].astype(float).tolist(),
            canonical_tautomer=params.curation.canonical_tautomer,
            empirical_half_width=regression_result.empirical_half_width,
            feature_names=model_feature_names(features),
        )
    return MlStage(
        params,
        table,
        models,
        regression_result,
        domain,
        test_predictions,
        cliff_performance,
        bundle,
        regression_test_predictions=regression_test_predictions,
        drop_intermediate=drop_intermediate,
    )


def build_params(settings: dict[str, Any]) -> RunParams:
    variant = settings["variant"]
    if variant in ("No mutation annotation", "Wild-type only"):
        variant = None
    elif variant == "Specific mutation":
        variant = settings.get("mutation", "").strip()
        if not variant:
            raise ValueError(
                "Enter a mutation annotation or choose another protein-variant filter."
            )
    if variant == "All (pooled)":
        variant = "any"
    return RunParams(
        curation=CurationParams(
            standard_types=tuple(settings["types"]),
            source_ids=settings["source_ids"],
            relations=("=", "<", ">", "<=", ">=") if settings["censored"] else ("=",),
            variant=variant,
            max_document_year=settings["max_year"],
            assay_ids=(settings["assay_id"],) if settings["assay_id"] else None,
            min_confidence_score=settings["min_confidence"] or None,
        ),
        classes=ClassScheme(
            bounds=(
                ("potent", settings.get("potent_cutoff", 8.0)),
                ("active", settings.get("active_cutoff", 7.0)),
                ("intermediate", settings.get("intermediate_cutoff", 6.0)),
            )
        ),
    )


# -- charts -------------------------------------------------------------------


def bar_chart(frame: pd.DataFrame, value: str, label: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(color=PURPLE, cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X(f"{value}:Q", title=label),
            y=alt.Y("value:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip(f"{value}:Q", title=label, format=","),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=26 * len(frame) + 30)
    )


def show_structure(smiles: str, caption: str = "", size: tuple[int, int] = (300, 210)) -> None:
    """Draw one molecule, or fall back to its SMILES if RDKit cannot."""
    svg = to_svg(smiles, size)
    if svg is None:
        st.code(smiles, language="text")
        return
    st.image(svg, caption=caption or None)


def structure_grid(
    items: list[tuple[str, str]], columns: int = 4, size: tuple[int, int] = (260, 190)
) -> None:
    """A grid of (smiles, caption) panels."""
    for row in range(0, len(items), columns):
        chunk = items[row : row + columns]
        for col, (smiles, caption) in zip(st.columns(columns), chunk, strict=False):
            with col:
                show_structure(smiles, caption, size)


def breakdown(records: list[dict[str, Any]], field: str) -> pd.DataFrame:
    counts = collections.Counter(
        "(none)" if r.get(field) is None else str(r.get(field)) for r in records
    )
    frame = pd.DataFrame(counts.most_common(), columns=["value", "records"])
    frame["share"] = frame["records"] / max(len(records), 1)
    return frame


def histogram(table: pd.DataFrame, column: str) -> alt.Chart:
    return (
        alt.Chart(table)
        .mark_bar(opacity=0.62)
        .encode(
            x=alt.X(f"{column}:Q", bin=alt.Bin(maxbins=40), title=column),
            y=alt.Y("count()", title="Molecules", stack=None),
            color=alt.Color(
                "group_label:N",
                title="Group",
                scale=alt.Scale(domain=list(GROUP_LABELS.values()), range=GROUP_COLORS),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[alt.Tooltip("count()", title="Molecules"), "group_label:N"],
        )
        .properties(height=190)
    )


def pca_chart(results: RunResults | PropertyStage) -> alt.Chart:
    frame = results.pca.scores.copy()
    frame["Activity class"] = results.table["activity_class"].to_numpy()
    frame["molecule"] = results.table["molecule_id"].to_numpy()
    frame["pactivity"] = results.table["pactivity"].to_numpy()
    var = results.pca.explained
    return (
        alt.Chart(frame)
        .mark_circle(size=26, opacity=0.5)
        .encode(
            x=alt.X("PC1:Q", title=f"PC1 ({var.iloc[0]:.1%} of variance)"),
            y=alt.Y("PC2:Q", title=f"PC2 ({var.iloc[1]:.1%} of variance)"),
            color=alt.Color(
                "Activity class:N",
                scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("molecule:N", title="Molecule"),
                alt.Tooltip("pactivity:Q", title="Potency", format=".2f"),
                "Activity class:N",
            ],
        )
        .properties(height=420)
        .interactive()
    )


def sas_chart(cliffs: pd.DataFrame, params: LandscapeParams) -> alt.Chart:
    points = (
        alt.Chart(cliffs)
        .mark_circle(size=22, opacity=0.55, color=RED)
        .encode(
            x=alt.X(
                "similarity:Q",
                title="Tanimoto similarity",
                scale=alt.Scale(domain=[params.similarity_threshold, 1.0]),
            ),
            y=alt.Y("delta:Q", title="Potency difference (log units)"),
            tooltip=[
                alt.Tooltip("id_a:N", title="Molecule A"),
                alt.Tooltip("id_b:N", title="Molecule B"),
                alt.Tooltip("similarity:Q", title="Similarity", format=".3f"),
                alt.Tooltip("delta:Q", title="Difference", format=".2f"),
                alt.Tooltip("sali:Q", title="SALI", format=".1f"),
            ],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [params.activity_threshold]}))
        .mark_rule(color="#8a8a85", strokeDash=[4, 4])
        .encode(y="y:Q")
    )
    return (points + rule).properties(height=400).interactive()


# -- sections -----------------------------------------------------------------


def show_overview(results: RunResults) -> None:
    """The first screen after Analyse: what was found, with structures."""
    table = results.table
    source = results.provenance.get("source", {})
    target_id = source.get("target_id")
    target_name = source.get("target_name")
    if target_id:
        heading = f"{target_name} · `{target_id}`" if target_name else f"`{target_id}`"
        st.markdown(f"### {heading}")
    cols = st.columns(4)
    cols[0].metric("Molecules", f"{len(table):,}")
    cols[1].metric("Scaffolds", f"{int(table['murcko'].nunique()):,}")
    cliffs = sum(len(s.cliffs) for s in results.landscapes.values())
    cols[2].metric("Activity cliffs", f"{cliffs:,}")
    cols[3].metric(
        "Best model",
        results.models.best_algorithm.replace("_", " ") if results.models else "—",
    )

    st.markdown("**Most potent molecules**")
    st.caption("The top of the curated dataset, by measured potency.")
    best = table.nlargest(4, "pactivity")
    structure_grid(
        [
            (row.smiles, f"{row.molecule_id} · potency {row.pactivity:.2f} · {row.activity_class}")
            for row in best.itertuples()
        ]
    )

    ranked = results.enrichment[results.enrichment["n"] >= 5]
    if len(ranked):
        st.markdown("**Most enriched scaffolds**")
        st.caption(
            "Scaffolds whose molecules are disproportionately potent, ranked by the "
            "Wilson lower bound so a single lucky compound cannot top the list."
        )
        structure_grid(
            [
                (
                    r.scaffold,
                    f"{r.n} molecules · {r.frac_group1:.0%} potent + active · EF {r.ef:.2f}",
                )
                for r in ranked.head(4).itertuples()
            ]
        )

    if results.models is not None and "naive" in set(results.models.scores["protocol"]):
        wide = results.models.scores.pivot(
            index="algorithm", columns="protocol", values="test_accuracy"
        )
        gap = (wide["naive"] - wide["leak_free"]).max()
        st.info(
            f"**Leakage audit.** Oversampling before the train/test split would have "
            f"inflated test accuracy by up to {gap:.3f} on this dataset. See the Models tab."
        )


def show_curation(results: RunResults | CurationStage) -> None:
    table = results.table
    cols = st.columns(4)
    cols[0].metric("Molecules", f"{len(table):,}")
    cols[1].metric("Retained measurements", f"{len(results.curation.evidence):,}")
    cols[2].metric("Activity classes present", str(table["activity_class"].nunique()))
    cols[3].metric("Rejected structures", f"{len(results.curation.rejected):,}")
    scheme = results.params.classes
    bounds = dict(scheme.bounds)
    variant = results.params.curation.variant
    variant_label = (
        "All mutation annotations pooled"
        if variant == "any"
        else f"Exact mutation annotation: {variant}"
        if variant is not None
        else "No mutation annotation (not confirmation of wild-type protein)"
    )
    st.caption(f"Applied protein-variant filter: {variant_label}.")
    ranges = {
        "potent": f"pActivity ≥ {bounds['potent']:g}",
        "active": f"{bounds['active']:g} ≤ pActivity < {bounds['potent']:g}",
        "intermediate": f"{bounds['intermediate']:g} ≤ pActivity < {bounds['active']:g}",
        "inactive": f"pActivity < {bounds['intermediate']:g}",
    }
    counts = (
        table["activity_class"]
        .value_counts()
        .reindex(list(CLASS_COLORS), fill_value=0)
        .rename_axis("activity_class")
        .reset_index(name="molecules")
    )
    st.markdown("**Activity classes in this curated subset**")
    for col, row in zip(st.columns(4), counts.itertuples(index=False), strict=True):
        col.metric(row.activity_class.title(), f"{row.molecules:,}")
        col.caption(ranges[row.activity_class])
    st.caption(
        "pActivity = −log10(molar concentration); higher means more potent. "
        "These are user-defined potency bins, not confirmed biological activity calls. "
        "Change Activity class cutoffs in the sidebar and rerun curation. "
        "Classes with zero compounds are shown too."
    )
    counts["range"] = counts["activity_class"].map(ranges)
    st.altair_chart(
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=6)
        .encode(
            x=alt.X("molecules:Q", title="Compounds"),
            y=alt.Y(
                "activity_class:N",
                title=None,
                sort=list(CLASS_COLORS),
                axis=alt.Axis(labelAngle=0),
            ),
            color=alt.Color(
                "activity_class:N",
                scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
                legend=None,
            ),
            tooltip=["activity_class:N", "molecules:Q", "range:N"],
        )
        .properties(height=190),
        width="stretch",
    )
    with st.expander("What do enrichment groups 1 and 2 mean?"):
        st.caption(
            "Group 1 = potent + active; Group 2 = intermediate + inactive, not inactive alone. "
            "These pooled groups are used for enrichment comparisons; classification ML "
            "uses the individual activity classes and regression uses continuous pActivity."
        )
        group_cols = st.columns(2)
        for col, group in zip(group_cols, (1, 2), strict=True):
            col.metric(
                f"Group {group} · {GROUP_LABELS[group]}",
                f"{int((table['group'] == group).sum()):,}",
            )

    st.caption(
        "Every record that left the dataset, and the step that removed it. These choices "
        "change every number below, which is why they are recorded rather than assumed."
    )
    log = pd.DataFrame(
        [
            {
                "Step": s.name,
                "In": s.records_in,
                "Out": s.records_out,
                "Removed": s.removed,
                "Molecules left": s.molecules_out,
                "Note": s.detail,
            }
            for s in results.curation.steps
        ]
    )
    st.dataframe(log, hide_index=True, width="stretch")
    evidence = results.curation.evidence
    if not evidence.empty:
        origins = source_summary(evidence)
        if not origins.empty:
            st.markdown("**Where these measurements came from**")
            st.caption(
                "These are original ChEMBL source IDs. PubChem and BindingDB entries are "
                "already integrated into ChEMBL; shared molecules are flagged rather than "
                "counted as independent evidence. This does not cover every record in either "
                "upstream database."
            )
            st.dataframe(origins, hide_index=True, width="stretch")
            disagreements = source_disagreement(evidence)
            if not disagreements.empty:
                with st.expander("Compounds measured in more than one origin"):
                    st.caption(
                        "Largest differences between source-specific median potencies. "
                        "Different assay protocols or endpoints may explain the spread; "
                        "inspect measurements before pooling."
                    )
                    st.dataframe(disagreements.head(100), hide_index=True, width="stretch")
        st.markdown("**Assay evidence behind the pooled molecule values**")
        st.caption(
            "Each row is a retained measurement. Values from different endpoints or assay "
            "formats may share a logarithmic unit without measuring the same biology. "
            "Use an assay ID in the sidebar to rerun a coherent subset. Confidence scores "
            "are fetched from ChEMBL assays when a minimum is selected."
        )
        fields = ["assay_chembl_id", "standard_type", "bao_label", "confidence_score"]
        if all(field in evidence for field in fields):
            overview = (
                evidence.groupby(fields, dropna=False)
                .agg(records=("record_id", "size"), molecules=("molecule_id", "nunique"))
                .reset_index()
                .sort_values("records", ascending=False)
            )
            st.dataframe(overview.head(100), hide_index=True, width="stretch")
        picked_id = st.text_input("Inspect measurements for molecule ID", key="evidence_id")
        if picked_id.strip():
            selected = evidence[evidence["molecule_id"].eq(picked_id.strip())]
            if selected.empty:
                st.info("No retained measurements for that curated molecule ID.")
            else:
                st.dataframe(selected, hide_index=True, width="stretch")
                st.caption(
                    "Record and document IDs trace back to ChEMBL. The report archive "
                    "contains the complete retained-measurement table."
                )
    for step, reason in results.skipped.items():
        st.warning(f"**{step} skipped.** {reason}")


def show_properties(results: RunResults | PropertyStage) -> None:
    from sarscope.analysis.descriptors import CORE_DESCRIPTORS

    table = results.table.copy()
    table["group_label"] = table["group"].map(GROUP_LABELS)
    st.caption(
        "Distributions by activity group, then the same six properties reduced to two "
        "components. Hover any point for the molecule behind it."
    )
    for row in range(0, len(CORE_DESCRIPTORS), 3):
        for col, prop in zip(st.columns(3), CORE_DESCRIPTORS[row : row + 3], strict=False):
            col.altair_chart(histogram(table, prop), width="stretch")

    if results.profile is None:
        # One group only: the comparison is meaningless, the PCA below is not.
        st.info(results.skipped.get("profile", "No group comparison available."))
    else:
        with st.expander("Descriptor statistics and Mann-Whitney p-values"):
            st.caption("Kurtosis is Fisher excess kurtosis, so a normal distribution scores 0.")
            stats = results.profile.stats.reset_index()
            stats["group"] = stats["group"].map(GROUP_LABELS)
            st.dataframe(stats, hide_index=True, width="stretch")
            st.dataframe(
                results.profile.p_values.rename("p_value").rename_axis("property").reset_index(),
                hide_index=True,
                width="stretch",
            )

    st.markdown("**Chemical space (PCA on the six properties)**")
    st.altair_chart(pca_chart(results), width="stretch")
    with st.expander("PCA loadings"):
        st.caption(
            "Properties are standardised first, or molecular weight would dominate every "
            "component. Signs are fixed so two runs are comparable."
        )
        st.dataframe(results.pca.loadings.reset_index(names="property"), width="stretch")
        st.dataframe(
            pd.DataFrame(
                {
                    "component": results.pca.explained.index,
                    "explained": results.pca.explained.to_numpy(),
                    "cumulative": results.pca.cumulative.to_numpy(),
                }
            ),
            hide_index=True,
            width="stretch",
        )


def show_structural_space(space: StructuralSpace, neighbors: int, min_dist: float) -> None:
    settings = space.settings
    st.caption(
        f"Displayed run: {settings['plotted_molecules']:,} / {settings['total_molecules']:,} "
        f"compounds plotted (sampled {settings['sampled_molecules']:,}) · "
        f"neighbors {settings['n_neighbors']} · minimum distance "
        f"{settings['min_dist']} · seed {settings['seed']}."
    )
    expected_neighbors = min(neighbors, len(space.points) - 1)
    missing = int(settings["unembedded_molecules"])
    if missing:
        st.warning(
            f"{missing:,} sampled compounds have no finite UMAP position (for example, "
            "disconnected fingerprint neighborhoods). They are omitted from the plot, "
            "not placed at an arbitrary location; the coordinates CSV flags them as unembedded."
        )
    if settings["n_neighbors"] != expected_neighbors or settings["min_dist"] != min_dist:
        st.info("UMAP controls have changed. Rerun UMAP to update the displayed projection.")
    color_by = st.radio(
        "Color structural space by", ["Activity class", "Continuous potency"], horizontal=True
    )
    color = (
        alt.Color(
            "activity_class:N",
            title="Activity class",
            scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
        )
        if color_by == "Activity class"
        else alt.Color("pactivity:Q", title="pActivity", scale=alt.Scale(scheme="viridis"))
    )
    chart = (
        alt.Chart(space.points.loc[space.points["embedded"]])
        .mark_circle(size=35, opacity=0.7)
        .encode(
            x=alt.X("UMAP1:Q", title="UMAP 1 (arbitrary units)"),
            y=alt.Y("UMAP2:Q", title="UMAP 2 (arbitrary units)"),
            color=color,
            tooltip=["molecule_id:N", "activity_class:N", alt.Tooltip("pactivity:Q", format=".2f")],
        )
        .properties(height=440)
        .interactive()
    )
    st.altair_chart(chart, width="stretch")
    st.download_button(
        "Download UMAP coordinates",
        space.points.to_csv(index=False),
        file_name="ecfp4_umap.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download UMAP settings",
        json.dumps(settings, indent=2),
        file_name="ecfp4_umap_settings.json",
        mime="application/json",
    )


def show_scaffolds(results: RunResults | ScaffoldStage) -> None:
    st.caption(
        "Ns scaffolds, Nss of them carrying a single molecule, Ncsk cyclic skeletons. "
        'The skeleton columns use RDKit\'s generic scaffold; "cyclic skeleton" has no single '
        "definition, so skeleton counts from different tools are not comparable."
    )
    st.dataframe(results.diversity.reset_index(names="class"), width="stretch")

    st.markdown("**Scaffold enrichment**")
    st.caption(
        "EF is the potent + active fraction within a scaffold divided by that of the whole "
        "dataset. A single active molecule scores the maximum EF, so the table is sorted by "
        "ef_lower — the Wilson lower bound, which requires evidence."
    )
    ranked = results.enrichment[results.enrichment["n"] >= 5]
    if ranked.empty:
        ranked = results.enrichment
    st.markdown("**Most enriched scaffolds with at least 5 molecules**")
    structure_grid(
        [
            (
                row.scaffold,
                f"{row.n} molecules · {row.frac_group1:.0%} potent + active · EF {row.ef:.2f}",
            )
            for row in ranked.head(8).itertuples()
        ]
    )
    with st.expander("Full enrichment table"):
        st.dataframe(
            results.enrichment,
            hide_index=True,
            width="stretch",
            column_config={
                "scaffold": st.column_config.TextColumn("Scaffold (SMILES)", width="large"),
                "frac_group1": st.column_config.NumberColumn(
                    "Potent + active fraction", format="percent"
                ),
                "ef": st.column_config.NumberColumn("EF", format="%.3f"),
                "ef_lower": st.column_config.NumberColumn("EF lower bound", format="%.3f"),
            },
        )


def show_rgroups(results: RunResults | ScaffoldStage) -> None:
    if not results.rgroups:
        st.info("No series had enough members to decompose.")
        return
    st.caption(
        "The input to a medicinal-chemistry SAR read, not a substitute for one. **delta** "
        "compares a substituent's median potency against the molecules of the same series that "
        "differ at that position. Those molecules may differ elsewhere too, so read delta "
        "together with n and p_value."
    )
    labels = [
        f"Series {i}: {s.n_molecules} molecules, {len(s.positions)} positions"
        for i, s in enumerate(results.rgroups, start=1)
    ]
    choice = st.selectbox("Series", labels, label_visibility="collapsed")
    sar = results.rgroups[labels.index(choice)]

    core, members = st.columns([1, 1])
    with core:
        st.markdown("**Shared scaffold**")
        show_structure(sar.scaffold, size=(330, 240))
    with members:
        st.markdown("**Labelled core**")
        st.caption("Numbered attachment points are the positions in the table below.")
        show_structure(sar.core, size=(330, 240))

    order = sar.substituents["delta"].abs().sort_values(ascending=False).index
    st.dataframe(
        sar.substituents.reindex(order),
        hide_index=True,
        width="stretch",
        column_config={
            "substituent": st.column_config.TextColumn("Substituent", width="medium"),
            "delta": st.column_config.NumberColumn("Delta (log units)", format="%.2f"),
            "p_value": st.column_config.NumberColumn("p", format="%.2e"),
        },
    )
    strongest = sar.substituents.reindex(order).head(6)
    if len(strongest):
        st.markdown("**Substituents with the largest effect**")
        structure_grid(
            [
                (
                    row.substituent,
                    f"{row.position} · n={row.n} · delta {row.delta:+.2f} · p={row.p_value:.1e}",
                )
                for row in strongest.itertuples()
            ],
            columns=3,
            size=(200, 150),
        )

    with st.expander("Molecules in this series"):
        st.dataframe(sar.members, hide_index=True, width="stretch")
        picked = st.selectbox(
            "Draw a molecule", sar.members["molecule_id"].tolist(), key=f"draw_{choice}"
        )
        row = results.table[results.table["molecule_id"] == picked]
        if len(row):
            show_structure(
                str(row.iloc[0]["smiles"]),
                f"{picked} · potency {row.iloc[0]['pactivity']:.2f}",
                size=(360, 260),
            )


def show_landscape(results: RunResults | LandscapeStage) -> None:
    st.caption(
        "Every pair of molecules, placed by structural similarity and potency difference. "
        "Cliffs are similar pairs with very different potency: they can challenge models "
        "and reveal sensitive SAR. Inspect assay context before interpreting a cliff."
    )
    params = results.params.landscape
    st.caption(
        f"Applied thresholds: Tanimoto > {params.similarity_threshold:g} and "
        f"|ΔpActivity| > {params.activity_threshold:g} "
        f"(>{10**params.activity_threshold:g}-fold potency difference). "
        "These are the settings used for this result; slider changes require rerunning analysis."
    )
    tabs = st.tabs(list(results.landscapes))
    for tab, sas in zip(tabs, results.landscapes.values(), strict=True):
        with tab:
            cols = st.columns(5)
            for col, (region, count) in zip(cols, sas.region_counts.items(), strict=False):
                col.metric(region.replace("_", " ").title(), f"{count:,}")
            cols[4].metric("Identical pairs", f"{len(sas.identical_pairs):,}")

            if sas.cliffs.empty:
                st.info("No pairs cross both thresholds with this fingerprint.")
                continue
            st.altair_chart(sas_chart(sas.cliffs, results.params.landscape), width="stretch")

            st.markdown("**Inspect a cliff pair**")
            st.caption(
                "The two molecules below are structurally similar but differ sharply in "
                "potency. What changed between them is the SAR."
            )
            query = st.text_input(
                "Filter pairs by molecule ID", key=f"cliff_search_{sas.fingerprint}"
            ).strip()
            sort_order = st.selectbox(
                "Sort cliff pairs",
                ["Largest potency difference", "Highest similarity"],
                key=f"cliff_sort_{sas.fingerprint}",
            )
            filtered = sas.cliffs
            if query:
                matches = filtered["id_a"].astype(str).str.contains(
                    query, case=False, regex=False
                ) | filtered["id_b"].astype(str).str.contains(query, case=False, regex=False)
                filtered = filtered.loc[matches]
            filtered = filtered.sort_values(
                "delta" if sort_order == "Largest potency difference" else "similarity",
                ascending=False,
                kind="stable",
            )
            if filtered.empty:
                st.info("No cliff pairs match that molecule ID. Clear the filter to see all pairs.")
                continue
            pages = (len(filtered) + 24) // 25
            page_key = f"cliff_page_{sas.fingerprint}"
            filter_key = f"cliff_filter_{sas.fingerprint}"
            if st.session_state.get(filter_key) != (query, sort_order, len(filtered)):
                st.session_state[page_key] = 1
                st.session_state[filter_key] = (query, sort_order, len(filtered))
            page = int(
                st.number_input(
                    "Cliff-pair page",
                    min_value=1,
                    max_value=pages,
                    value=1,
                    step=1,
                    key=page_key,
                )
            )
            start = (page - 1) * 25
            top = filtered.iloc[start : start + 25]
            st.caption(
                f"Showing pairs {start + 1}–{start + len(top)} of {len(filtered):,} matches "
                f"({len(sas.cliffs):,} total cliffs). Every pair is accessible; "
                "choose a page, then a pair to draw its two structures."
            )
            options = [
                f"{r.id_a} / {r.id_b} — similarity {r.similarity:.2f}, "
                f"delta {r.delta:.2f} log units"
                for r in top.itertuples()
            ]
            picked = st.selectbox("Cliff pair", options, key=f"cliff_{sas.fingerprint}")
            pair = top.iloc[options.index(picked)]
            st.dataframe(top, hide_index=True, width="stretch")
            smiles = results.table.set_index("molecule_id")["smiles"]
            potency = results.table.set_index("molecule_id")["pactivity"]
            left, right = st.columns(2)
            for col, molecule_id in zip((left, right), (pair["id_a"], pair["id_b"]), strict=True):
                with col:
                    if molecule_id in smiles.index:
                        show_structure(
                            str(smiles[molecule_id]),
                            f"{molecule_id} · potency {potency[molecule_id]:.2f}",
                            size=(340, 250),
                        )

            if all(molecule_id in smiles.index for molecule_id in (pair["id_a"], pair["id_b"])):
                from sarscope.docking_ui import show_cliff_docking

                show_cliff_docking(
                    [
                        {
                            "molecule_id": str(m),
                            "smiles": str(smiles[m]),
                            "pactivity": float(potency[m]),
                        }
                        for m in (pair["id_a"], pair["id_b"])
                    ],
                    key=f"cliff_docking_{sas.fingerprint}",
                )

            gens = cliff_generators(sas.cliffs, results.params.landscape.generator_sd)
            generators = gens[gens["is_generator"]]
            st.markdown(
                f"**{len(generators)} cliff generators** "
                f"(more than {gens.attrs['threshold']:.1f} cliffs each)"
            )
            st.caption(
                "Molecules forming many cliffs at once. They are hard for a model and "
                "informative for a chemist: a small change here moves potency a lot."
            )
            if len(generators):
                structure_grid(
                    [
                        (str(smiles[m]), f"{m} · {n} cliffs")
                        for m, n in zip(
                            generators["molecule_id"], generators["n_cliffs"], strict=True
                        )
                        if m in smiles.index
                    ][:8],
                    size=(240, 180),
                )
            st.dataframe(generators, hide_index=True, width="stretch")

    if results.consensus_generators:
        st.success(
            "Cliff generators found under every fingerprint: "
            + ", ".join(results.consensus_generators)
        )


def show_models(results: RunResults | MlStage) -> None:
    if results.models is None and results.regression is None:
        st.info(results.skipped.get("model", "Modelling did not run."))
        return
    if results.params.model.split == "time":
        model = results.regression if results.regression is not None else results.models
        assert model is not None
        model_table = results.model_table if isinstance(results, RunResults) else results.table
        years = pd.to_numeric(model_table["document_year"], errors="coerce")
        train_years = years.iloc[model.train_index]
        test_years = years.iloc[model.test_index]
        st.info(
            f"**Time-based validation:** {len(train_years)} compounds first documented "
            f"through {results.params.model.time_cutoff} train the model; "
            f"{len(test_years)} newly documented later compounds test it "
            f"({int(test_years.min())}–{int(test_years.max())}). "
            "Model selection uses expanding, earlier-to-later year folds within training."
        )
    if results.params.model.split == "source":
        model = results.regression if results.regression is not None else results.models
        assert model is not None
        source_id = results.params.model.source_test_id
        assert source_id is not None
        st.info(
            f"**Origin-held-out validation:** trained on {len(model.train_index)} compounds "
            f"from other ChEMBL origins; tested on {len(model.test_index)} compounds "
            f"unique to source {source_id} "
            f"({SOURCE_NAMES.get(source_id, 'other ChEMBL source')}). "
            "Compounds measured in both origins are excluded from the test set."
        )
    if results.models is not None:
        st.caption(
            "**leak_free** selects features and resamples inside training folds only, after "
            "the split. **naive** is the optional audit ordering—select and oversample on "
            "everything, then split—which leaks held-out information."
        )
        scores = results.models.scores
        show = scores[
            [
                "algorithm",
                "protocol",
                "test_accuracy",
                "test_mcc",
                "cv_roc_auc_ovr",
                "test_roc_auc_ovr",
                "cv_pr_auc_ovr",
                "test_pr_auc_ovr",
                "test_auc_classes",
            ]
        ]
        st.dataframe(
            show,
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(c.replace("_", " ").title(), format="%.3f")
                for c in [
                    "test_accuracy",
                    "test_mcc",
                    "cv_roc_auc_ovr",
                    "test_roc_auc_ovr",
                    "cv_pr_auc_ovr",
                    "test_pr_auc_ovr",
                ]
            },
        )
        st.markdown(f"Best classifier by CV MCC: **{results.models.best_algorithm}**")
        st.caption(
            "ROC AUC and PR AUC are macro one-vs-rest summaries over classes with both "
            "positives and negatives in the held-out set. PR AUC here is average precision "
            "(AP), not trapezoidal area; compare it with each class's prevalence. "
            "AUCs are undefined when no class is evaluable."
            " Ranking scores from margin-based models are not calibrated probabilities."
        )
        st.markdown("**Held-out performance by activity class**")
        st.dataframe(results.models.test_class_metrics, hide_index=True, width="stretch")
        with st.expander("Confusion matrix and ROC / PR curves"):
            st.caption("Rows are measured classes; columns are predicted classes.")
            st.dataframe(results.models.test_confusion, width="stretch")
            class_scores = results.models.test_class_scores
            truth = results.models.test_truth
            if class_scores is not None and truth is not None:
                evaluable = results.models.test_class_metrics.dropna(subset=["roc_auc_ovr"])
                choices = evaluable["activity_class"].tolist()
                if choices:
                    chosen = st.selectbox("One-vs-rest curve for", choices)
                    curve_color = CLASS_COLORS.get(chosen, PURPLE)
                    column = results.models.test_score_classes.index(chosen)
                    positive = np.asarray(truth == chosen, dtype=bool)
                    values = class_scores[:, column]
                    fpr, tpr, _ = roc_curve(positive, values)
                    precision, recall, _ = precision_recall_curve(positive, values)
                    curve_cols = st.columns(2)
                    roc_points = pd.DataFrame(
                        {"false_positive_rate": fpr, "true_positive_rate": tpr}
                    )
                    pr_points = pd.DataFrame({"recall": recall, "precision": precision})
                    curve_cols[0].altair_chart(
                        alt.Chart(roc_points)
                        .mark_line(color=curve_color)
                        .encode(
                            x=alt.X("false_positive_rate:Q", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("true_positive_rate:Q", scale=alt.Scale(domain=[0, 1])),
                        )
                        .properties(title=f"ROC · {chosen}"),
                        width="stretch",
                    )
                    curve_cols[1].altair_chart(
                        alt.Chart(pr_points)
                        .mark_line(color=curve_color)
                        .encode(
                            x=alt.X("recall:Q", scale=alt.Scale(domain=[0, 1])),
                            y=alt.Y("precision:Q", scale=alt.Scale(domain=[0, 1])),
                        )
                        .properties(title=f"Precision–recall · {chosen}"),
                        width="stretch",
                    )
                    st.caption(f"PR baseline for {chosen}: {positive.mean():.1%} prevalence.")
                else:
                    st.info("ROC / PR curves need both positives and negatives in the test set.")

        if "naive" in set(scores["protocol"]):
            wide = scores.pivot(index="algorithm", columns="protocol", values="test_accuracy")
            gap = (wide["naive"] - wide["leak_free"]).sort_values(ascending=False)
            st.markdown("**Inflation from the leaky order, in test accuracy**")
            st.dataframe(
                gap.rename("inflation").reset_index(),
                hide_index=True,
                width="stretch",
                column_config={"inflation": st.column_config.NumberColumn(format="%.3f")},
            )

    if results.regression is not None:
        st.markdown("**Continuous pActivity regression**")
        st.caption(
            "Fits the measured potency directly instead of discarding information at class "
            "boundaries. The winner is selected by cross-validated RMSE."
        )
        st.dataframe(results.regression.scores, hide_index=True, width="stretch")
        st.caption(
            "The mean_baseline row predicts the training-fold mean and is a reference, not "
            "an additional tuned model. Requested regressors are selected by training CV only."
        )
        st.markdown(
            f"Best regressor by cross-validated RMSE: **{results.regression.best_algorithm}**"
        )
        band = results.regression.empirical_half_width
        if np.isfinite(band):
            metrics = st.columns(2)
            metrics[0].metric("Empirical 90% half-width", f"±{band:.2f} log units")
            metrics[1].metric(
                "Observed test coverage", f"{results.regression.empirical_test_coverage:.1%}"
            )
            st.caption(
                "This band is the 90th percentile of training-fold absolute errors for the "
                "CV-selected model. It is an empirical guide, not a formal coverage guarantee; "
                "the observed test coverage above matters most for new scaffolds and later data."
            )
        if not results.regression_test_predictions.empty:
            profile = results.regression_test_predictions
            show_regression_diagnostics(results.regression, profile)
            st.markdown("**Error versus nearest training analogue**")
            chart = (
                alt.Chart(profile)
                .mark_circle(size=55, opacity=0.65, color=PURPLE)
                .encode(
                    x=alt.X("max_training_similarity:Q", title="Nearest training similarity"),
                    y=alt.Y("absolute_error:Q", title="Absolute potency error"),
                    tooltip=["molecule_id:N", "pactivity:Q", "predicted_pactivity:Q"],
                )
            )
            st.altair_chart(chart, width="stretch")
            st.dataframe(novelty_summary(profile), hide_index=True, width="stretch")
            with st.expander("Held-out predictions and novelty"):
                st.dataframe(profile, hide_index=True, width="stretch")
            st.download_button(
                "Download held-out regression predictions",
                profile.to_csv(index=False),
                file_name="regression_test_predictions.csv",
                mime="text/csv",
            )
    if not results.cliff_model_performance.empty:
        st.markdown("**Held-out error on cliff compounds versus the rest**")
        st.dataframe(results.cliff_model_performance, hide_index=True, width="stretch")

    if results.domain is not None:
        show_applicability_domain(results)


def applicability_domain_chart(
    domain: DomainResult, train_ids: list[str], test_ids: list[str]
) -> tuple[alt.LayerChart, pd.DataFrame]:
    """Plot the fitted criterion, not a new embedding or a test-fitted boundary."""
    dimensions = domain.train_scores.shape[1]
    train = pd.DataFrame(domain.train_scores[:, :2], columns=["PC1", "PC2"][:dimensions])
    test = pd.DataFrame(domain.query_scores[:, :2], columns=["PC1", "PC2"][:dimensions])
    train["molecule_id"] = train_ids
    test["molecule_id"] = test_ids
    train["status"] = "Training compound"
    train["inside_pca_box"] = True
    test["status"] = np.where(domain.in_domain, "Test · inside PCA box", "Test · outside PCA box")
    test["inside_pca_box"] = domain.in_domain
    points = pd.concat([train, test], ignore_index=True)
    # Boundaries and downloads always use the full data. Cap only rendered points.
    rendered = pd.concat(
        [part.sample(min(2000, len(part)), random_state=42) for part in (train, test)],
        ignore_index=True,
    )
    bounds = pd.DataFrame({"x_min": [train.PC1.min()], "x_max": [train.PC1.max()]})
    rectangle = (
        alt.Chart(bounds)
        .mark_rect(color=PURPLE, opacity=0.13, stroke=PURPLE, strokeWidth=2)
        .encode(x="x_min:Q", x2="x_max:Q")
    )
    if dimensions >= 2:
        bounds["y_min"], bounds["y_max"] = train.PC2.min(), train.PC2.max()
        rectangle = rectangle.encode(y="y_min:Q", y2="y_max:Q")
        y = alt.Y("PC2:Q", title="PC2 · fitted on training features", scale=alt.Scale(zero=False))
        tooltip = ["molecule_id:N", "status:N", "PC1:Q", "PC2:Q"]
        height = 350
    else:
        # A single retained feature gives a genuine 1D interval, not a fabricated PC2.
        rectangle = rectangle.encode(y=alt.value(0), y2=alt.value(140))
        y = alt.value(70)
        tooltip = ["molecule_id:N", "status:N", "PC1:Q"]
        height = 140
    palette = alt.Scale(
        domain=["Training compound", "Test · inside PCA box", "Test · outside PCA box"],
        range=[TEAL, PURPLE, RED],
    )
    dots = (
        alt.Chart(rendered)
        .mark_circle(size=48, opacity=0.6)
        .encode(
            x=alt.X(
                "PC1:Q", title="PC1 · fitted on training features", scale=alt.Scale(zero=False)
            ),
            y=y,
            color=alt.Color("status:N", title=None, scale=palette, legend=alt.Legend(orient="top")),
            tooltip=tooltip,
            order=alt.Order("status:N", sort="descending"),
        )
    )
    chart = (
        (rectangle + dots)
        .properties(height=height, title="Training feature envelope and held-out compounds")
        .interactive()
    )
    return chart, points


def show_applicability_domain(results: RunResults | MlStage) -> None:
    domain = results.domain
    selected = results.regression or results.models
    if domain is None or selected is None:
        return
    if not len(domain.in_domain):
        st.info("Applicability-domain coverage is undefined: there are no held-out compounds.")
        return
    table = results.table
    if isinstance(results, RunResults) and not results.model_table.empty:
        table = results.model_table
    train_ids = table.iloc[selected.train_index]["molecule_id"].astype(str).tolist()
    test_ids = table.iloc[selected.test_index]["molecule_id"].astype(str).tolist()
    inside = int(domain.in_domain.sum())
    total = len(domain.in_domain)
    dimensions = domain.train_scores.shape[1]
    component_label = "both plotted components" if dimensions > 1 else "the plotted component"
    st.markdown("### Applicability domain · how familiar is the test chemistry?")
    metrics = st.columns(3)
    metrics[0].metric("Test compounds inside the applicability domain", f"{domain.coverage:.1%}")
    metrics[1].metric("Inside PCA box", f"{inside:,} / {total:,}")
    metrics[2].metric("Outside PCA box", f"{total - inside:,} / {total:,}")
    st.markdown(
        f"**What this means:** {inside:,} of {total:,} held-out compounds fall within the "
        f"training set's ranges on {component_label}. "
        "The shaded region is that boundary; mint points are training compounds, purple "
        "points are inside-test compounds and coral points are outside-test compounds. "
        "Hover for molecule IDs; zoom to inspect overlapping points."
    )
    chart, points = applicability_domain_chart(domain, train_ids, test_ids)
    st.altair_chart(chart, width="stretch")
    if len(train_ids) > 2000 or total > 2000:
        st.caption(
            "Rendering up to 2,000 training and 2,000 test points (seed 42). "
            "Boundary, counts and downloadable coordinates use all compounds."
        )
    st.warning(
        "100% inside is not 100% prediction accuracy. This is a permissive feature-space "
        "screen, not a probability of reliability: a point can lie inside the box but "
        "far from training examples, in a gap between chemical series, or on an activity cliff. "
        "A two-component view may also omit variation in other feature dimensions."
        if dimensions > 1
        else "Inside this interval is not a prediction-accuracy guarantee: a compound can "
        "still lack close training analogues or lie on an activity cliff."
    )
    with st.expander("How the boundary is calculated"):
        features = results.params.model.features
        st.markdown(
            f"Representation: **{features.representation}**; validation split: "
            f"**{results.params.model.split}**; training compounds: **{len(train_ids):,}**. "
            "Preprocessing and PCA are fitted only on the outer training set. "
            f"For each of the {dimensions} retained PCA components, the boundary is the "
            "training minimum and maximum; a test point is inside only if every component "
            "lies in its range (with numerical tolerance). The percentage is a property "
            "of this split and representation, not an intrinsic property of the target. "
            "The boundary is not refitted to the held-out compounds."
        )
        if dimensions == 1:
            st.caption(
                "Only one component is available, so this plot is a 1D interval, not a 2D map."
            )
        if results.regression is not None and results.models is not None:
            st.caption(
                "Both prediction tasks were selected; this panel uses the regressor's "
                "outer training/test partition."
            )
    profile = results.regression_test_predictions
    if not profile.empty and profile["molecule_id"].astype(str).tolist() == test_ids:
        queries = points[points["status"] != "Training compound"].reset_index(drop=True)
        for name in [
            "max_training_similarity",
            "training_similarity_cutoff",
            "in_training_domain",
            "pactivity",
            "predicted_pactivity",
            "absolute_error",
        ]:
            if name in profile:
                queries[name] = profile[name].to_numpy()
        st.markdown("**A separate check: similarity to the nearest training analogue**")
        st.caption(
            f"Binary {results.params.model.features.fingerprint.upper()} Tanimoto similarity "
            "ranges from 0 (no shared bits) to 1 (identical fingerprints, not necessarily "
            "identical molecules). Unlike the PCA box, this checks proximity to a real "
            "training compound. It uses the original binary fingerprint even for descriptor models."
        )
        histogram = queries[["max_training_similarity", "status"]].copy()
        histogram["bin_left"] = (
            np.minimum(np.floor(histogram["max_training_similarity"] * 20), 19) / 20
        )
        histogram = histogram.groupby(["bin_left", "status"], as_index=False).size()
        histogram["bin_right"] = histogram["bin_left"] + 0.05
        bars = (
            alt.Chart(histogram)
            .mark_bar(opacity=0.8)
            .encode(
                x=alt.X(
                    "bin_left:Q",
                    title="Nearest training Tanimoto similarity",
                    scale=alt.Scale(domain=[0, 1]),
                ),
                x2="bin_right:Q",
                y=alt.Y("size:Q", title="Test compounds", stack="zero"),
                color=alt.Color(
                    "status:N",
                    scale=alt.Scale(
                        domain=["Test · inside PCA box", "Test · outside PCA box"],
                        range=[PURPLE, RED],
                    ),
                    title="PCA status",
                    legend=alt.Legend(orient="top"),
                ),
                tooltip=[
                    alt.Tooltip("size:Q", title="Compounds"),
                    "status:N",
                    alt.Tooltip("bin_left:Q", title="Similarity from", format=".2f"),
                    alt.Tooltip("bin_right:Q", title="Similarity to", format=".2f"),
                ],
            )
        )
        if "training_similarity_cutoff" in queries:
            cutoff = float(queries["training_similarity_cutoff"].iloc[0])
            line = (
                alt.Chart(pd.DataFrame({"cutoff": [cutoff]}))
                .mark_rule(color=ORANGE, strokeDash=[6, 4], strokeWidth=2)
                .encode(x="cutoff:Q", tooltip=[alt.Tooltip("cutoff:Q", format=".3f")])
            )
            bars = bars + line
            structural_inside = int(queries["in_training_domain"].sum())
            st.markdown(
                f"**Structural similarity coverage: {structural_inside:,} / {total:,} "
                f"({structural_inside / total:.1%}).** The amber line at **{cutoff:.3f}** "
                "is the 5th percentile of training compounds' nearest non-self similarities, "
                "computed without test compounds. This is a separate heuristic, not the "
                "criterion behind the PCA percentage above. Neither rule guarantees accuracy."
            )
        st.altair_chart(bars.properties(height=220), width="stretch")
        errors = []
        for label in ("Test · inside PCA box", "Test · outside PCA box"):
            part = queries[queries["status"] == label]
            error = part["absolute_error"].to_numpy()
            errors.append(
                {
                    "PCA domain status": label,
                    "Test compounds": len(part),
                    "MAE (log units)": float(error.mean()) if len(error) else np.nan,
                    "RMSE (log units)": float(np.sqrt(np.mean(error**2))) if len(error) else np.nan,
                }
            )
        st.markdown("**Observed regression error, not just domain membership**")
        st.dataframe(pd.DataFrame(errors), hide_index=True, width="stretch")
        if inside == total:
            st.caption(
                "There are no outside-PCA-box test compounds in this split, so their "
                "prediction error cannot be assessed or compared with inside compounds."
            )
        points = pd.concat(
            [points[points["status"] == "Training compound"], queries], ignore_index=True
        )
    st.download_button(
        "Download applicability-domain coordinates and flags",
        points.to_csv(index=False),
        file_name="applicability_domain.csv",
        mime="text/csv",
    )


def report_zip(results: RunResults) -> bytes:
    """The same folder the CLI writes, as a zip for the download button."""
    with TemporaryDirectory() as tmp:
        out = Path(tmp) / "report"
        write_report(results, out)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out))
        return buffer.getvalue()


WORKFLOW_KEYS = (
    "curation_stage",
    "property_stage",
    "structural_space",
    "scaffold_stage",
    "landscape_stage",
    "ml_stage",
    "regression_bootstrap",
    "explanation_stage",
    "report_zip",
    "comparison_stage",
    "uploaded_predictions",
    "shortlist_stage",
)


def clear_workflow(*, keep_curation: bool = False) -> None:
    for key in WORKFLOW_KEYS:
        if keep_curation and key == "curation_stage":
            continue
        st.session_state.pop(key, None)
    for key in list(st.session_state):
        if key.startswith("cliff_docking_") and key.endswith("_result"):
            st.session_state.pop(key, None)


def show_regression_diagnostics(result: RegressionResult, profile: pd.DataFrame) -> None:
    summary = regression_summary(result)
    values = summary.set_index("metric")["value"]
    metrics = st.columns(4)
    metrics[0].metric("Held-out MAE", f"{values['Test MAE']:.2f} log units")
    metrics[1].metric("Within 3-fold", f"{values['Within 3-fold']:.1%}")
    metrics[2].metric("Within 10-fold", f"{values['Within 10-fold']:.1%}")
    metrics[3].metric("Typical fold error", f"{values['Typical fold error']:.1f}×")
    if values["RMSE gain versus baseline"] <= 0:
        st.warning(
            "This model did not improve held-out RMSE over predicting the training mean. "
            "Treat its predictions as exploratory; do not use test results to retune and "
            "then claim the same test as independent validation."
        )
    plotted = profile
    if len(profile) > 5000:
        plotted = profile.sample(5000, random_state=42)
        st.caption(
            "Plots sample 5,000 test compounds; metrics and downloads use the full test set."
        )
    low = float(min(profile["pactivity"].min(), profile["predicted_pactivity"].min())) - 0.2
    high = float(max(profile["pactivity"].max(), profile["predicted_pactivity"].max())) + 0.2
    points = (
        alt.Chart(plotted)
        .mark_circle(size=40, opacity=0.65)
        .encode(
            x=alt.X("pactivity:Q", title="Measured pActivity", scale=alt.Scale(domain=[low, high])),
            y=alt.Y(
                "predicted_pactivity:Q",
                title="Predicted pActivity",
                scale=alt.Scale(domain=[low, high]),
            ),
            color=alt.Color(
                "activity_class:N",
                scale=alt.Scale(domain=list(CLASS_COLORS), range=list(CLASS_COLORS.values())),
            ),
            tooltip=["molecule_id:N", "pactivity:Q", "predicted_pactivity:Q", "absolute_error:Q"],
        )
    )
    guides = [
        alt.Chart(pd.DataFrame({"x": [low, high], "y": [low + offset, high + offset]}))
        .mark_line(color=color, strokeDash=dashes)
        .encode(x="x:Q", y="y:Q")
        for offset, color, dashes in [
            (0, TEAL, [1, 0]),
            (np.log10(3), ORANGE, [5, 4]),
            (-np.log10(3), ORANGE, [5, 4]),
        ]
    ]
    st.markdown("**Measured versus predicted · held-out compounds**")
    st.caption(
        "Mint is perfect prediction; dashed amber lines mark ±3-fold error, not confidence bands."
    )
    st.altair_chart(alt.layer(*guides, points).properties(height=420), width="stretch")
    left, right = st.columns(2)
    residual = (
        alt.Chart(plotted)
        .mark_circle(size=30, opacity=0.6, color=PURPLE)
        .encode(
            x=alt.X("predicted_pactivity:Q", title="Predicted pActivity"),
            y=alt.Y("residual:Q", title="Measured − predicted (log units)"),
            tooltip=["molecule_id:N", "residual:Q"],
        )
    )
    zero = alt.Chart(pd.DataFrame({"residual": [0.0]})).mark_rule(color=TEAL).encode(y="residual:Q")
    left.altair_chart((residual + zero).properties(height=260), width="stretch")
    right.altair_chart(
        alt.Chart(plotted)
        .mark_bar(color=ORANGE)
        .encode(
            x=alt.X("residual:Q", bin=alt.Bin(maxbins=35), title="Measured − predicted"),
            y=alt.Y("count()", title="Test compounds"),
        )
        .properties(height=260),
        width="stretch",
    )
    with st.expander("Regression metrics and interpretation"):
        st.dataframe(summary, hide_index=True, width="stretch")
        st.download_button(
            "Download regression diagnostic metrics",
            summary.to_csv(index=False),
            file_name="regression_diagnostics.csv",
            mime="text/csv",
        )
    st.markdown("**Optional uncertainty on held-out metrics**")
    st.caption(
        "Scaffold-block bootstrap resamples whole held-out chemical series and compares "
        "model and baseline on the same draws. These approximate 95% intervals describe "
        "this fitted model/test population; they are not prediction intervals or protection "
        "against distribution shift. Few scaffold blocks make them unstable."
    )
    bootstrap_key = hashlib.sha256(
        result.test_predictions.tobytes()
        + result.test_truth.tobytes()
        + (
            result.baseline_test_predictions.tobytes()
            if result.baseline_test_predictions is not None
            else b""
        )
        + profile["murcko"].fillna("").astype(str).str.cat(sep="|").encode()
    ).hexdigest()
    if st.button("Estimate metric confidence intervals", key="run_regression_bootstrap"):
        try:
            with st.spinner("Resampling held-out scaffold blocks (500 repeats)…"):
                st.session_state["regression_bootstrap"] = (
                    bootstrap_key,
                    bootstrap_regression(result, profile["murcko"].tolist()),
                )
        except ValueError as exc:
            st.info(str(exc))
    cached = st.session_state.get("regression_bootstrap")
    if cached is not None and cached[0] == bootstrap_key:
        intervals = cached[1]
        if int(intervals.iloc[0]["scaffold_blocks"]) < 10:
            st.warning(
                "Fewer than 10 held-out scaffold blocks: these intervals can be very unstable."
            )
        st.dataframe(intervals, hide_index=True, width="stretch")
        st.download_button(
            "Download metric confidence intervals",
            intervals.to_csv(index=False),
            file_name="regression_bootstrap.csv",
            mime="text/csv",
        )


def importance_chart(frame: pd.DataFrame, title: str) -> alt.Chart:
    shown = frame.head(12).sort_values("importance")
    return (
        alt.Chart(shown)
        .mark_bar(color=ORANGE, cornerRadiusEnd=3)
        .encode(
            x=alt.X("importance:Q", title="Importance"),
            y=alt.Y("descriptor:N", sort=None, title=None),
            tooltip=["descriptor:N", alt.Tooltip("importance:Q", format=".4f")],
        )
        .properties(height=28 * len(shown) + 35, title=title)
    )


def show_explanation(explanation: DescriptorExplanation) -> None:
    metric = explanation.metrics.iloc[0]
    cols = st.columns(3)
    cols[0].metric("Held-out R²", f"{metric['test_r2']:.3f}")
    cols[1].metric("Held-out RMSE", f"{metric['test_rmse']:.3f}")
    cols[2].metric("Held-out Spearman", f"{metric['test_spearman']:.3f}")
    st.caption(
        "Permutation importance is measured only on held-out compounds. Positive values mean "
        "shuffling that descriptor worsened RMSE; correlated descriptors can share importance."
    )
    st.altair_chart(
        importance_chart(explanation.permutation, "Held-out permutation importance"),
        width="stretch",
    )
    st.dataframe(explanation.permutation, hide_index=True, width="stretch")
    if not explanation.intrinsic.empty:
        st.markdown("**Random Forest impurity importance**")
        st.caption(
            "This is the tree model's built-in reduction-in-variance importance. It is not "
            "Gini importance—the target is continuous—and it can favour high-variance or "
            "correlated descriptors, so read it beside permutation importance."
        )
        st.altair_chart(
            importance_chart(explanation.intrinsic, "RF impurity importance"),
            width="stretch",
        )
    st.markdown("**Held-out predictions and residuals**")
    st.dataframe(explanation.predictions, hide_index=True, width="stretch")
    st.markdown("**SHAP**")
    st.caption(explanation.shap_status)
    if not explanation.shap_global.empty:
        st.altair_chart(
            importance_chart(explanation.shap_global, "Mean absolute SHAP value"),
            width="stretch",
        )
        molecule_id = st.selectbox(
            "Local SHAP explanation",
            explanation.shap_local["molecule_id"].tolist(),
            key="shap_molecule",
        )
        local = explanation.shap_local[explanation.shap_local["molecule_id"] == molecule_id].iloc[0]
        contributions = (
            local.drop(labels="molecule_id")
            .rename("contribution")
            .rename_axis("descriptor")
            .reset_index()
        )
        contributions["absolute"] = contributions["contribution"].abs()
        st.dataframe(
            contributions.sort_values("absolute", ascending=False).drop(columns="absolute"),
            hide_index=True,
            width="stretch",
        )


def assemble_report_results(
    curation: CurationStage,
    properties: PropertyStage,
    scaffolds: ScaffoldStage,
    landscape: LandscapeStage,
    ml: MlStage | None,
) -> RunResults:
    table = properties.table.copy()
    table["murcko"] = scaffolds.table["murcko"]
    table["skeleton"] = scaffolds.table["skeleton"]
    return RunResults(
        params=ml.params if ml is not None else landscape.params,
        curation=curation.curation,
        table=table,
        profile=properties.profile,
        pca=properties.pca,
        diversity=scaffolds.diversity,
        enrichment=scaffolds.enrichment,
        landscapes=landscape.landscapes,
        consensus_cliffs=landscape.consensus_cliffs,
        consensus_generators=landscape.consensus_generators,
        models=ml.models if ml is not None else None,
        regression=ml.regression if ml is not None else None,
        domain=ml.domain if ml is not None else None,
        matched_pairs=scaffolds.matched_pairs,
        transformation_summary=scaffolds.transformation_summary,
        model_test_predictions=(ml.model_test_predictions if ml is not None else pd.DataFrame()),
        cliff_model_performance=(ml.cliff_model_performance if ml is not None else pd.DataFrame()),
        prediction_bundle=ml.prediction_bundle if ml is not None else None,
        model_table=ml.table if ml is not None else pd.DataFrame(),
        regression_test_predictions=(
            ml.regression_test_predictions if ml is not None else pd.DataFrame()
        ),
        rgroups=scaffolds.rgroups,
        skipped={**curation.skipped, **properties.skipped, **(ml.skipped if ml else {})},
        provenance={
            **curation.provenance,
            "ml_selection": {
                "drop_intermediate": ml.drop_intermediate if ml is not None else False,
                "model_compounds": len(ml.table) if ml is not None else None,
            },
        },
    )


# -- page ---------------------------------------------------------------------


def sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Settings")
        st.caption("Fast browser defaults; the CLI remains available for exhaustive runs.")

        with st.expander("Curation", expanded=True):
            types = st.multiselect("Activity types", ACTIVITY_TYPES, default=["IC50"])
            variant = st.selectbox(
                "Protein variant",
                ["No mutation annotation", "Specific mutation", "All (pooled)"],
                index=0,
            )
            mutation = ""
            if variant == "Specific mutation":
                mutation = st.text_input(
                    "Exact mutation annotation",
                    help="Matches ChEMBL assay_variant_mutation exactly for any target. "
                    "Inspect raw fields to see the available annotations.",
                ).strip()
            st.caption(
                "No mutation annotation means ChEMBL has not annotated a mutation; "
                "it does not establish wild-type protein. Pooling annotations can mix variants."
            )
            censored = st.checkbox("Keep censored values (>, <)", value=False)
            max_year = st.number_input(
                "Only documents up to year", min_value=1990, max_value=2030, value=2030
            )
            assay_id = (
                st.text_input(
                    "Restrict to ChEMBL assay ID (optional)",
                    help=(
                        "Inspect the assay table after curation, then enter one assay ID "
                        "here and curate again."
                    ),
                )
                .strip()
                .upper()
            )
            min_confidence = st.slider(
                "Minimum assay-target confidence",
                0,
                9,
                0,
                help="0 keeps all assays; 9 requires a direct single-protein assignment.",
            )
            origin_labels = {
                f"{name} · {identifier}": identifier for identifier, name in SOURCE_NAMES.items()
            }
            chosen_origins = st.multiselect(
                "ChEMBL evidence origins (optional)",
                list(origin_labels),
                help="Empty keeps all origins. Source 7 is integrated PubChem BioAssay; "
                "source 37 is integrated BindingDB. This avoids downloading the same "
                "measurements again from those databases.",
            )

        with st.expander("Activity class cutoffs"):
            st.caption(
                "User-defined bins on the pActivity = −log10(M) scale; higher is more potent. "
                "Defaults 8 / 7 / 6 correspond to 10 / 100 / 1000 nM. "
                "These are conventions, not target-specific biological activity criteria."
            )
            potent_cutoff = st.number_input(
                "Potent minimum pActivity", min_value=0.0, max_value=15.0, value=8.0, step=0.1
            )
            active_cutoff = st.number_input(
                "Active minimum pActivity", min_value=0.0, max_value=15.0, value=7.0, step=0.1
            )
            intermediate_cutoff = st.number_input(
                "Intermediate minimum pActivity",
                min_value=0.0,
                max_value=15.0,
                value=6.0,
                step=0.1,
            )
            st.caption(
                "Below the intermediate minimum = inactive. "
                "Enrichment Group 1 = potent + active; Group 2 = intermediate + inactive. "
                "ML uses the four class labels, not these two enrichment groups. "
                "Rerun curation after changing cutoffs; regression still uses continuous values."
            )

        st.caption(f"SARscope {__version__} · [source](https://github.com/yboulaamane/sarscope)")

    return {
        "types": types,
        "variant": variant,
        "mutation": mutation,
        "censored": censored,
        "max_year": None if max_year >= 2030 else int(max_year),
        "assay_id": assay_id,
        "min_confidence": min_confidence,
        "source_ids": tuple(origin_labels[name] for name in chosen_origins) or None,
        "potent_cutoff": float(potent_cutoff),
        "active_cutoff": float(active_cutoff),
        "intermediate_cutoff": float(intermediate_cutoff),
    }


def apply_visual_theme() -> None:
    """Warm, multicolour accents without adding network assets or startup work."""
    st.markdown(
        """
        <style>
        .stApp {
            background: radial-gradient(circle at 88% 4%, rgba(197, 155, 255, .15),
                transparent 30%), radial-gradient(circle at 5% 32%,
                rgba(255, 129, 148, .08), transparent 28%), radial-gradient(circle at 80% 90%,
                rgba(82, 223, 182, .06), transparent 35%), #14131d;
        }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #251e31, #1b1a28 72%);
            border-right: 1px solid rgba(197, 155, 255, .17);
        }
        [data-testid="stMetric"] {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(197, 155, 255, .22);
            border-radius: 12px;
            background: rgba(40, 34, 52, .78);
            padding: .7rem 1rem;
        }
        [data-testid="stMetric"]::before {
            content: "";
            position: absolute;
            inset: 0 0 auto;
            height: 3px;
            background: linear-gradient(90deg, #52dfb6, #c59bff, #ffca76, #ff8194);
        }
        div.stButton > button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(105deg, #ffca76, #ff9e8e) !important;
            color: #24151e !important;
            border: 1px solid #ffd492 !important;
            font-weight: 700;
            box-shadow: 0 4px 18px rgba(255, 158, 142, .22);
        }
        div.stButton > button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(105deg, #ffdc9e, #ffb7a6) !important;
            color: #24151e !important;
        }
        div.stButton > button[kind="primary"]:focus-visible,
        button[data-testid="stBaseButton-primary"]:focus-visible {
            outline: 2px solid #52dfb6;
            outline-offset: 2px;
        }
        div.stButton > button[kind="secondary"],
        div.stDownloadButton > button {
            border-color: rgba(197, 155, 255, .48);
            background: rgba(197, 155, 255, .07);
        }
        div.stButton > button[kind="secondary"]:hover,
        div.stDownloadButton > button:hover {
            border-color: #52dfb6;
            color: #f5f1f9;
            background: rgba(82, 223, 182, .11);
        }
        h1 { color: #f5f1f9; letter-spacing: -.035em; }
        h1::after {
            content: "";
            display: block;
            width: 5.5rem;
            height: 4px;
            margin-top: .3rem;
            border-radius: 4px;
            background: linear-gradient(90deg, #52dfb6, #c59bff, #ffca76);
        }
        h2, h3 { color: #ebd9ff; }
        div[data-baseweb="tab-list"] { border-bottom: 1px solid #51445f; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #52dfb6; }
        div[data-baseweb="tab-highlight"] {
            background: linear-gradient(90deg, #52dfb6, #c59bff);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_pubchem_screen() -> None:
    st.title("SARscope · PubChem qualitative screen")
    st.markdown(
        "Classify **Active vs Inactive** outcomes from one PubChem BioAssay AID. "
        "This is separate from pIC50 QSAR: screening calls are never converted "
        "to potency values."
    )
    aid = int(st.number_input("PubChem BioAssay AID", min_value=1, value=1000, step=1))
    st.caption(
        "The browser workflow is limited to 750 assay calls and runs only when you press "
        "the button. Use the CLI for larger eligible assays; neither route silently "
        "truncates a screen."
    )
    if st.button("Run qualitative screen", type="primary"):
        try:
            with st.spinner("Retrieving assay calls and evaluating the scaffold holdout…"):
                with PubChemClient() as pubchem:
                    rows, meta = pubchem.concise_assay(aid)
                    if len(rows) > 750:
                        raise ValueError(
                            f"AID {aid} has {len(rows):,} calls; use the CLI for this screen"
                        )
                    cids = [
                        int(row["CID"])
                        for row in rows
                        if row.get("CID", "").isdigit()
                        and row.get("Activity Outcome", "").lower() in ("active", "inactive")
                    ]
                    smiles = pubchem.smiles_for_cids(cids)
                result = run_screen_snapshot(aid, rows, smiles, meta)
            st.session_state["pubchem_screen"] = (aid, result)
        except (PubChemError, ValueError) as exc:
            st.error(f"Screen stopped: {exc}")
            return
    saved = st.session_state.get("pubchem_screen")
    if saved is None or saved[0] != aid:
        st.info(
            "Enter an AID and run the screen to inspect its curated calls and held-out metrics."
        )
        return
    result = saved[1]
    st.subheader(f"{result.metadata['assay_name']} · AID {aid}")
    st.caption(
        f"{result.metadata['assay_type']} assay · target accession "
        f"{result.metadata['target_accession'] or 'not provided'}"
    )
    metrics = result.scores
    cols = st.columns(4)
    cols[0].metric("Curated compounds", f"{len(result.compounds):,}")
    cols[1].metric("Held-out ROC AUC", f"{metrics['test_roc_auc']:.3f}")
    cols[2].metric("Held-out PR AUC", f"{metrics['test_pr_auc']:.3f}")
    cols[3].metric("Test prevalence", f"{metrics['test_prevalence']:.3f}")
    st.caption(
        "PR AUC baseline equals test prevalence. This single scaffold split is exploratory, "
        "not a prospective or multi-assay performance claim."
    )
    st.json(result.audit)
    st.dataframe(result.predictions, hide_index=True, width="stretch")
    snapshot = {
        "assay": result.metadata,
        "rows": result.source_rows,
        "smiles_by_cid": result.source_smiles,
    }
    st.download_button(
        "Download source snapshot",
        json.dumps(snapshot, indent=2) + "\n",
        file_name=f"pubchem_aid{aid}_source_snapshot.json",
        mime="application/json",
    )
    st.download_button(
        "Download held-out predictions",
        result.predictions.to_csv(index=False),
        file_name=f"pubchem_aid{aid}_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="SARscope", layout="wide", page_icon="🔬")
    apply_visual_theme()
    with st.sidebar:
        workflow = st.radio("Workflow", ["ChEMBL potency & SAR", "PubChem qualitative screen"])
    if workflow == "PubChem qualitative screen":
        show_pubchem_screen()
        return
    settings = sidebar()
    if settings["variant"] == "Specific mutation" and not settings["mutation"]:
        st.info("Enter an exact mutation annotation in the sidebar, or choose another filter.")
        return
    if not settings["potent_cutoff"] > settings["active_cutoff"] > settings["intermediate_cutoff"]:
        st.error(
            "Class cutoffs must be ordered: potent > active > intermediate. Adjust the sidebar."
        )
        return

    st.title("SARscope")
    st.markdown(
        "Start from a target ID, gene, protein, or disease and build an auditable "
        "medicinal-chemistry analysis. Choose the stages you need; nothing runs automatically."
    )
    st.image(
        str(Path(__file__).parent / "assets" / "sarscope-hero.svg"),
        use_container_width=True,
    )
    st.caption(
        "Find target → Curate evidence → Explore SAR → Validate QSAR. "
        "Illustration only; PubChem qualitative screening is separate."
    )
    st.caption(
        "QSAR predictions support compound prioritisation; they do not by themselves "
        "establish binding, selectivity, safety, or experimental activity."
    )

    reason = unavailable_reason()
    if reason:
        # Everything still works; structures fall back to SMILES text.
        st.warning(f"Structures cannot be drawn here. {reason}")

    if not settings["types"]:
        st.info("Choose at least one activity type in the sidebar.")
        return
    target_id, check_target, discovery = target_search_controls()
    if target_id is None:
        return

    lookup_key = (target_id, tuple(settings["types"]), repr(discovery))
    if check_target:
        try:
            with st.spinner("Checking this target in ChEMBL…"):
                target, release, n_records = lookup(target_id, lookup_key[1])
        except ChemblError as exc:
            st.error(str(exc))
            return
        st.session_state["target_lookup"] = {
            "key": lookup_key,
            "target": target,
            "release": release,
            "n_records": n_records,
        }
        clear_workflow()

    target_lookup = st.session_state.get("target_lookup")
    if target_lookup is None or target_lookup.get("key") != lookup_key:
        st.info("Press **Check target** to confirm its identity and available activity records.")
        return
    target = target_lookup["target"]
    release = str(target_lookup["release"])
    n_records = int(target_lookup["n_records"])

    resolved_id = str(target["target_chembl_id"])
    st.subheader(f"{target['pref_name']} · {resolved_id}")
    cols = st.columns(4)
    cols[0].metric("ChEMBL target ID", resolved_id)
    cols[1].metric("Organism", target["organism"])
    cols[2].metric("Type", str(target["target_type"]).title())
    cols[3].metric(f"{'/'.join(settings['types'])} records", f"{n_records:,}")
    st.caption(
        f"Data: {release}. Check the name above — a wrong ID does not fail, it quietly "
        "fetches a different protein, and every number below is then about that protein."
    )

    if n_records == 0:
        st.warning("No records of these types for this target.")
        return

    base_params = build_params(settings)
    curation_key = (
        target_id,
        tuple(settings["types"]),
        repr(base_params.curation),
        repr(base_params.classes),
        release,
        repr(discovery),
    )
    action_cols = st.columns(2)
    run_curation = action_cols[0].button(
        f"1 · Curate {resolved_id}", type="primary", width="stretch"
    )
    load_raw = action_cols[1].button("Inspect raw fields", width="stretch")
    if n_records >= 5_000:
        st.caption(
            f"{n_records:,} raw records match these activity types. The first load may take "
            "time; curation progress and separate phase timings appear when you start. "
            "Narrower curation settings reduce processing, but currently do not reduce "
            "the raw download."
        )
    if load_raw:
        with st.spinner("Downloading raw activity records…"):
            records = fetch(target_id, tuple(settings["types"]))
        for row in range(0, len(FETCH_SUMMARY_FIELDS), 2):
            for col, field in zip(st.columns(2), FETCH_SUMMARY_FIELDS[row : row + 2], strict=False):
                title, note = FIELD_NOTES[field]
                with col:
                    st.markdown(f"**{title}**")
                    st.caption(note)
                    st.altair_chart(
                        bar_chart(breakdown(records, field), "records", "Records"),
                        width="stretch",
                    )
    if run_curation:
        with st.status(f"Preparing {resolved_id}…", expanded=True) as status:
            progress_bar = st.progress(0, text="Loading activity records…")

            def update_download(done: int, total: int) -> None:
                progress_bar.progress(
                    min(done / total, 1.0) if total else 1.0,
                    text=f"Loading activity records: {done:,} / {total:,}",
                )

            def update_standardization(done: int, total: int) -> None:
                progress_bar.progress(
                    done / total if total else 0.0,
                    text=(
                        f"Standardizing structures: {done:,} / {total:,}"
                        if total
                        else "Filtering measurements…"
                    ),
                )

            try:
                stage = run_curation_stage(
                    target_id,
                    tuple(settings["types"]),
                    base_params,
                    release,
                    str(target["pref_name"]),
                    str(target["organism"]),
                    on_status=status.write,
                    on_download_progress=update_download,
                    on_standardize_progress=update_standardization,
                )
                stage.provenance["target_discovery"] = discovery
            except (ChemblError, ValueError) as exc:
                status.update(label="Curation stopped", state="error", expanded=True)
                st.error(f"Curation stopped: {exc}")
                return
            progress_bar.empty()
            status.update(label=f"{resolved_id} curated", state="complete", expanded=False)
        clear_workflow()
        st.session_state["curation_stage"] = (curation_key, stage)

    curation_entry = st.session_state.get("curation_stage")
    if curation_entry is None or curation_entry[0] != curation_key:
        st.info(
            "Start with curation. Later chemistry and ML steps remain idle until you choose "
            "them, and changing curation settings requires this step to run again."
        )
        return
    curation: CurationStage = curation_entry[1]
    if curation.timings:
        timings = curation.timings
        st.caption(
            f"Activity load: {timings['load_records']:.1f}s · "
            f"Curation: {timings['curation']:.1f}s"
            + (
                f" · Assay confidence: {timings['assay_confidence']:.1f}s"
                if timings["assay_confidence"]
                else ""
            )
            + " · Later filter changes reuse standardized structures while this app process lives."
        )

    tabs = st.tabs(
        [
            "1 · Curation",
            "2 · Chemical space",
            "3 · Scaffolds & SAR",
            "4 · Activity cliffs",
            "5 · ML",
            "6 · Explain",
            "7 · Report",
            "8 · Selectivity",
            "9 · Prioritize",
        ]
    )
    with tabs[0]:
        show_curation(curation)
        with st.expander("Curated molecules"):
            st.dataframe(curation.table, hide_index=True, width="stretch")

    with tabs[1]:
        st.caption(
            "Compute named RDKit physicochemical descriptors, compare activity groups and "
            "project the compounds into standardized descriptor PCA space. "
            "For structural neighborhoods, run ECFP4/Jaccard UMAP separately below."
        )
        if st.button("Run chemical-space analysis", key="run_properties"):
            with st.spinner("Calculating descriptors and PCA…"):
                st.session_state["property_stage"] = run_property_stage(curation.curation)
            st.session_state.pop("report_zip", None)
        properties: PropertyStage | None = st.session_state.get("property_stage")
        if properties is None:
            st.info("This step has not run.")
        else:
            show_properties(properties)
        st.markdown("**Structural chemical space · ECFP4 UMAP**")
        st.caption(
            "Binary Morgan radius-2 fingerprints (2048 bits), Jaccard distance = 1 − Tanimoto. "
            "No activity labels enter the embedding. Up to 2,000 compounds are uniformly "
            "sampled with seed 42; larger curated datasets remain intact for other analyses. "
            "UMAP is exploratory: 2D gaps and clusters are not potency or applicability evidence. "
            "The first run may take longer while numerical kernels compile."
        )
        controls = st.columns(2)
        neighbors = controls[0].slider("UMAP neighbors", 2, 50, 15)
        min_dist = controls[1].slider("UMAP minimum distance", 0.0, 1.0, 0.1, 0.05)
        umap_available = find_spec("umap") is not None
        if not umap_available:
            st.info(
                "UMAP needs umap-learn installed on the host. Descriptor PCA remains available."
            )
        if st.button(
            "Run ECFP4 UMAP",
            key="run_umap",
            disabled=not umap_available or len(curation.table) < 3,
        ):
            try:
                with st.spinner("Embedding ECFP4 fingerprints with Jaccard distance…"):
                    st.session_state["structural_space"] = run_structural_space(
                        curation.table, neighbors, min_dist
                    )
            except (ImportError, ValueError) as exc:
                st.error(f"UMAP stopped: {exc}")
        space: StructuralSpace | None = st.session_state.get("structural_space")
        if space is not None:
            show_structural_space(space, neighbors, min_dist)

    with tabs[2]:
        st.caption(
            "Murcko diversity and enrichment are the core scaffold step. R-group decomposition "
            "and matched molecular pairs are separate opt-in calculations."
        )
        if st.button("Run scaffold analysis", key="run_scaffolds"):
            with st.spinner("Calculating Murcko scaffolds and enrichment…"):
                st.session_state["scaffold_stage"] = run_scaffold_stage(curation.curation)
            for key in ("ml_stage", "explanation_stage", "report_zip"):
                st.session_state.pop(key, None)
        scaffolds: ScaffoldStage | None = st.session_state.get("scaffold_stage")
        if scaffolds is None:
            st.info("Run the scaffold step before series-level SAR or ML.")
        else:
            show_scaffolds(scaffolds)
            sar_cols = st.columns(2)
            if sar_cols[0].button("Run R-group decomposition", key="run_rgroups"):
                if scaffolds.enrichment.empty:
                    st.warning("R-group analysis needs at least one enriched scaffold.")
                else:
                    with st.spinner("Decomposing the largest scaffold series…"):
                        scaffolds.rgroups = decompose_top_scaffolds(
                            scaffolds.table, scaffolds.enrichment
                        )
                        st.session_state["scaffold_stage"] = scaffolds
                        st.session_state.pop("report_zip", None)
            if sar_cols[1].button("Run matched molecular pairs", key="run_mmp"):
                with st.spinner("Enumerating single-cut transformations…"):
                    scaffolds.matched_pairs = matched_molecular_pairs(scaffolds.table)
                    scaffolds.transformation_summary = summarise_transformations(
                        scaffolds.matched_pairs, curation.curation.evidence
                    )
                    st.session_state["scaffold_stage"] = scaffolds
                    st.session_state.pop("report_zip", None)
            if scaffolds.rgroups:
                show_rgroups(scaffolds)
            if not scaffolds.matched_pairs.empty:
                st.markdown("**Transformation evidence**")
                st.caption(
                    "Positive median delta favours the fragment in the 'to' column. "
                    "Check pair counts, independent contexts, opposing examples, and "
                    "whether both compounds were measured in a shared assay."
                )
                st.dataframe(scaffolds.transformation_summary, hide_index=True, width="stretch")
                with st.expander("Individual matched molecular pairs"):
                    st.dataframe(scaffolds.matched_pairs, hide_index=True, width="stretch")

    with tabs[3]:
        st.caption(
            "Activity cliffs are an all-pairs calculation. Choose the structural definitions "
            "and thresholds explicitly before running it."
        )
        fingerprints = st.multiselect(
            "Landscape fingerprints", ["ecfp4", "maccs"], default=["ecfp4"], key="landscape_fp"
        )
        threshold_cols = st.columns(2)
        similarity_threshold = threshold_cols[0].slider(
            "Similarity threshold", 0.5, 0.99, 0.9, 0.01
        )
        activity_threshold = threshold_cols[1].slider(
            "Potency difference (log units)", 0.5, 4.0, 2.0, 0.1
        )
        pair_count = len(curation.table) * (len(curation.table) - 1) // 2
        st.caption(f"This run will examine {pair_count:,} unordered molecular pairs.")
        if len(curation.table) > LANDSCAPE_WARN:
            st.warning(
                f"The browser limit for all-pairs cliffs is {LANDSCAPE_WARN:,} molecules. "
                "Use the CLI for the full dataset or curate a coherent assay subset."
            )
        if st.button(
            "Run activity-cliff analysis",
            key="run_landscape",
            disabled=len(curation.table) > LANDSCAPE_WARN,
        ):
            if not fingerprints:
                st.error("Choose at least one fingerprint.")
            else:
                landscape_params = LandscapeParams(
                    fingerprints=cast(tuple[FingerprintName, ...], tuple(fingerprints)),
                    similarity_threshold=similarity_threshold,
                    activity_threshold=activity_threshold,
                )
                run_params = RunParams(
                    curation=base_params.curation,
                    classes=base_params.classes,
                    landscape=landscape_params,
                )
                with st.spinner("Building the selected SAS maps…"):
                    st.session_state["landscape_stage"] = run_landscape_stage(
                        curation.curation, run_params
                    )
                for key in ("ml_stage", "explanation_stage", "report_zip"):
                    st.session_state.pop(key, None)
        landscape: LandscapeStage | None = st.session_state.get("landscape_stage")
        if landscape is None:
            st.info("This step has not run.")
        else:
            show_landscape(landscape)

    with tabs[4]:
        st.caption(
            "Choose the prediction task, fingerprint, validation split and algorithms. Nothing "
            "in this section runs when a chemistry setting changes."
        )
        scaffolds = st.session_state.get("scaffold_stage")
        if scaffolds is None:
            st.info("Run the scaffold step first; scaffold-aware validation needs its groups.")
        else:
            tasks = st.multiselect(
                "Prediction tasks",
                ["Continuous pActivity regression", "Activity-class classification"],
                default=["Continuous pActivity regression"],
            )
            intermediate_count = int((scaffolds.table["activity_class"] == "intermediate").sum())
            drop_intermediate = st.checkbox(
                f"Exclude intermediate class from ML ({intermediate_count:,} compounds)",
                value=False,
                disabled=intermediate_count == 0,
                help="Applies only to the selected ML tasks and their held-out validation. "
                "Curation, chemical space, scaffolds and activity cliffs keep the full dataset. "
                "Potent and active remain separate classifier labels; this does not "
                "silently turn the task into binary classification. "
                "For continuous regression, retaining intermediate compounds usually preserves "
                "useful potency information.",
            )
            retained = len(scaffolds.table) - (intermediate_count if drop_intermediate else 0)
            st.caption(
                f"Pre-split ML selection: {retained:,} of {len(scaffolds.table):,} "
                "curated compounds. Time and source splits can change this count. "
                "Upstream SAR analyses are unchanged."
            )
            if drop_intermediate:
                st.info(
                    "Intermediate compounds will be excluded before ML splitting and CV. "
                    "Potent, active and inactive remain distinct classes; continuous regression "
                    "also uses this reduced set if selected."
                )
            algorithms = (
                st.multiselect(
                    "Classification algorithms", FAST_ALGORITHMS, default=["random_forest"]
                )
                if "Activity-class classification" in tasks
                else []
            )
            regression_algorithms = (
                st.multiselect(
                    "Regression algorithms",
                    [
                        "random_forest",
                        "extra_trees",
                        "gradient_boosting",
                        "nearest_neighbors",
                        "svr",
                        "ridge",
                        "neural_net",
                    ],
                    default=["random_forest", "ridge"],
                )
                if "Continuous pActivity regression" in tasks
                else []
            )
            representations = {
                "Fingerprints": "fingerprint",
                "RDKit 2D descriptors": "rdkit2d",
                "Fingerprints + RDKit 2D": "hybrid",
                "Molfeat RDKit 2D (optional)": "molfeat2d",
            }
            representation_label = st.selectbox("Molecular representation", list(representations))
            representation = representations[representation_label]
            descriptor_names = descriptor_controls("ml") if representation != "fingerprint" else ()
            molfeat_missing = representation == "molfeat2d" and find_spec("molfeat") is None
            if molfeat_missing:
                st.info(
                    "Molfeat is an optional local/server extra: pip install 'sarscope[molfeat]'. "
                    "It requires PyTorch, so it is not installed by default on Community Cloud. "
                    "Native RDKit 2D gives you named descriptors without that dependency."
                )
            if representation != "fingerprint":
                st.caption(
                    "Descriptor/hybrid ML uses training-fold median imputation, standardization, "
                    "constant removal and correlation filtering. Structural similarity and "
                    "analogue-domain flags still use binary fingerprints, not descriptor Tanimoto."
                )
            ml_cols = st.columns(3)
            model_fp = ml_cols[0].selectbox("Fingerprint", ["ecfp4", "maccs"])
            split = ml_cols[1].selectbox(
                "Validation split", ["scaffold", "time", "source", "random"]
            )
            cv_folds = ml_cols[2].slider("CV folds", 3, 10, 5)
            time_cutoff = 2019
            if split == "time":
                time_cutoff = int(
                    st.number_input(
                        "Last training document year",
                        min_value=1950,
                        max_value=2029,
                        value=2019,
                        help="Train on compounds first documented by this year; test only "
                        "on new compounds documented later. Cross-validation within training "
                        "also moves forward by document year.",
                    )
                )
                st.caption(
                    "Needs document years and at least CV folds + 1 distinct years through "
                    "the cutoff. Later repeat measurements do not update earlier training "
                    "labels. For compound tables, use the first documented year."
                )
            source_test_id = None
            if split == "source":
                evidence = curation.curation.evidence
                available = sorted(
                    {int(value) for value in evidence.get("src_id", []) if pd.notna(value)}
                )
                if available:
                    source_test_id = st.selectbox(
                        "Held-out ChEMBL evidence origin",
                        available,
                        index=available.index(37) if 37 in available else 0,
                        format_func=lambda value: (
                            f"{SOURCE_NAMES.get(value, f'ChEMBL source {value}')} · {value}"
                        ),
                        help="Only compounds absent from all other origins enter the test.",
                    )
                if source_test_id is None:
                    st.warning("No ChEMBL origin IDs were retained; choose another split.")
                st.caption(
                    "Training uses other origins and scaffold CV. Shared structures are "
                    "excluded from the held-out origin, even if their records disagree."
                )
            audit = st.checkbox(
                "Run classification leakage audit",
                value=False,
                disabled=split in ("time", "source"),
                help="The audit uses a random split, so its score gap is not comparable "
                "with a time- or origin-based holdout.",
            )
            if st.button("Run selected ML", key="run_ml", type="primary", disabled=molfeat_missing):
                if (
                    not tasks
                    or ("Activity-class classification" in tasks and not algorithms)
                    or ("Continuous pActivity regression" in tasks and not regression_algorithms)
                    or (split == "source" and source_test_id is None)
                    or (representation != "fingerprint" and not descriptor_names)
                ):
                    st.error("Choose at least one task and one algorithm.")
                else:
                    model_params = ModelParams(
                        features=FeatureParams(
                            fingerprint=cast(FingerprintName, model_fp),
                            representation=cast(Any, representation),
                            descriptor_names=descriptor_names,
                        ),
                        algorithms=tuple(algorithms),
                        regression_algorithms=tuple(regression_algorithms),
                        split=cast(SplitStrategy, split),
                        time_cutoff=int(time_cutoff),
                        source_test_id=source_test_id,
                        cv_folds=cv_folds,
                        leakage_audit=audit and split not in ("time", "source"),
                    )
                    landscape_for_params = (
                        landscape.params.landscape if landscape is not None else LandscapeParams()
                    )
                    run_params = RunParams(
                        curation=base_params.curation,
                        classes=base_params.classes,
                        landscape=landscape_for_params,
                        model=model_params,
                    )
                    try:
                        with st.spinner("Running only the selected models and validation…"):
                            st.session_state["ml_stage"] = run_ml_stage(
                                scaffolds,
                                curation.curation,
                                run_params,
                                classification="Activity-class classification" in tasks,
                                regression="Continuous pActivity regression" in tasks,
                                landscape=landscape,
                                drop_intermediate=drop_intermediate,
                            )
                    except (ValueError, ImportError) as exc:
                        st.session_state.pop("ml_stage", None)
                        st.error(f"ML stopped: {exc}")
                    st.session_state.pop("explanation_stage", None)
                    st.session_state.pop("report_zip", None)
                    st.session_state.pop("uploaded_predictions", None)
                    st.session_state.pop("shortlist_stage", None)
            ml: MlStage | None = st.session_state.get("ml_stage")
            if ml is None:
                st.info("No ML run yet.")
            else:
                if (
                    ml.params.model.features.representation != representation
                    or ml.params.model.features.descriptor_names != descriptor_names
                    or (
                        "Continuous pActivity regression" in tasks
                        and ml.params.model.regression_algorithms != tuple(regression_algorithms)
                    )
                ):
                    st.warning(
                        "ML feature/model choices changed. Rerun ML to update these results."
                    )
                if ml.drop_intermediate != drop_intermediate:
                    st.warning(
                        "ML selection changed. Press Run selected ML to update these results."
                    )
                st.caption(
                    f"Model dataset: {len(ml.table):,} compounds · "
                    + (
                        "intermediate class excluded"
                        if ml.drop_intermediate
                        else "all curated activity classes retained"
                    )
                )
                show_models(ml)

    with tabs[5]:
        st.caption(
            "Fit a separate low-dimensional model to named RDKit descriptors on the same "
            "held-out split. This is for interpretation, not a claim that descriptors are the "
            "best predictive representation."
        )
        ml = st.session_state.get("ml_stage")
        if ml is None or (ml.regression is None and ml.models is None):
            st.info("Run ML first to establish the untouched train/test partition.")
        else:
            explanation_names = descriptor_controls("explain")
            if len(explanation_names) > 50:
                st.info(
                    "Browser explanations are limited to 50 selected descriptors; choose a subset."
                )
            explain_model = st.radio(
                "Descriptor explanation model",
                ["random_forest", "mlp"],
                format_func=lambda value: "Random Forest" if value == "random_forest" else "MLP",
                horizontal=True,
            )
            repeats = st.slider("Permutation repeats", 3, 15, 5)
            shap_available = find_spec("shap") is not None
            use_shap = st.checkbox(
                "Compute TreeSHAP (Random Forest only)",
                value=False,
                disabled=not shap_available or explain_model != "random_forest",
            )
            if not shap_available:
                st.caption(
                    "SHAP works in Streamlit, but the shap Python package is missing on this "
                    "deployment. Install requirements.txt and redeploy to enable TreeSHAP. "
                    "Permutation importance remains available."
                )
            else:
                st.caption(
                    "TreeSHAP attributes the descriptor Random Forest's predictions to individual "
                    "properties. It runs only when selected, on at most 100 held-out compounds; "
                    "it explains model behavior, not biological causation. MLP uses permutation "
                    "importance instead."
                )
            if st.button(
                "Run descriptor explanation",
                key="run_explanation",
                disabled=not explanation_names or len(explanation_names) > 50,
            ):
                selected = ml.regression or ml.models
                assert selected is not None
                with st.spinner("Fitting and explaining the descriptor model…"):
                    st.session_state["explanation_stage"] = explain_descriptor_model(
                        ml.table,
                        np.asarray(selected.train_index),
                        np.asarray(selected.test_index),
                        algorithm=explain_model,
                        permutation_repeats=repeats,
                        compute_shap=use_shap,
                        descriptor_names=explanation_names,
                    )
            explanation: DescriptorExplanation | None = st.session_state.get("explanation_stage")
            if explanation is None:
                st.info("No explanation run yet.")
            else:
                show_explanation(explanation)

    with tabs[6]:
        properties = st.session_state.get("property_stage")
        scaffolds = st.session_state.get("scaffold_stage")
        landscape = st.session_state.get("landscape_stage")
        ml = st.session_state.get("ml_stage")
        if properties is None or scaffolds is None or landscape is None:
            st.info(
                "Run chemical space, scaffolds and activity cliffs before assembling the full "
                "report. ML, R-groups and matched pairs remain optional."
            )
        else:
            results = assemble_report_results(curation, properties, scaffolds, landscape, ml)
            report_key = (
                target_id,
                repr(results.params.to_dict()),
                len(scaffolds.rgroups),
                len(scaffolds.matched_pairs),
            )
            if st.button("Prepare report archive", key="prepare_report", type="primary"):
                with st.spinner("Building the report archive…"):
                    st.session_state["report_zip"] = (report_key, report_zip(results))
            prepared = st.session_state.get("report_zip")
            if prepared is not None and prepared[0] == report_key:
                st.download_button(
                    "Download the report folder (.zip)",
                    data=prepared[1],
                    file_name=f"sarscope_{target_id}.zip",
                    mime="application/zip",
                )
            st.json(curation.provenance, expanded=False)

    with tabs[7]:
        st.caption(
            "Compare this curated target with another ChEMBL target. Shared structures get "
            "a potency ratio and assay-context label; unmeasured target–compound pairs "
            "stay unknown. This can download a second target, so run it deliberately."
        )
        other_raw = st.text_input("Second ChEMBL target ID", placeholder="CHEMBL279")
        second_assay_id = (
            st.text_input(
                "Second-target assay ID (optional)",
                help="The primary assay ID in the sidebar does not apply to the second target.",
            )
            .strip()
            .upper()
        )
        if st.button("Compare targets", type="primary", key="compare_targets"):
            try:
                other_id = normalise_target_id(other_raw)
                if other_id == target_id:
                    raise ValueError("Choose a different target for selectivity comparison.")
                with st.spinner("Curating and matching compounds across both targets…"):
                    comparison = compare_targets(target_id, other_id, base_params, second_assay_id)
                st.session_state["comparison_stage"] = (
                    target_id,
                    other_id,
                    repr(base_params.curation),
                    second_assay_id,
                    comparison,
                )
            except (ChemblError, ValueError) as exc:
                st.session_state.pop("comparison_stage", None)
                st.error(str(exc))
        comparison_entry = st.session_state.get("comparison_stage")
        if comparison_entry is not None:
            a_id, b_id, curation_settings, assay_b, comparison = comparison_entry
            other_key = other_raw.strip().upper()
            if other_key.isdigit():
                other_key = f"CHEMBL{other_key}"
            if (a_id, b_id, curation_settings, assay_b) == (
                target_id,
                other_key,
                repr(base_params.curation),
                second_assay_id,
            ):
                st.subheader(
                    f"{comparison.target_a['pref_name']} · {a_id} vs "
                    f"{comparison.target_b['pref_name']} · {b_id}"
                )
                st.metric("Shared measured structures", len(comparison.compounds))
                st.markdown("**Compound–target coverage**")
                st.caption("A blank potency is an unmeasured target, not inactivity.")
                st.dataframe(comparison.activity_matrix, hide_index=True, width="stretch")
                st.download_button(
                    "Download compound–target matrix",
                    comparison.activity_matrix.to_csv(index=False),
                    file_name=f"sarscope_{a_id}_{b_id}_matrix.csv",
                    mime="text/csv",
                )
                st.markdown("**Shared compounds and assay context**")
                st.caption(
                    "A potency ratio across different endpoints or assay formats is only a "
                    "cross-assay observation. Review the underlying assays before "
                    "calling it selectivity."
                )
                st.dataframe(comparison.compounds, hide_index=True, width="stretch")
                st.markdown("**Shared-scaffold comparison**")
                st.dataframe(comparison.scaffolds, hide_index=True, width="stretch")

    with tabs[8]:
        st.caption(
            "Upload up to 500 SMILES to score with the selected regression model. "
            "Structures are standardised with the training curation setting. The output "
            "includes invalid-row reasons, nearest measured analogues, applicability flags, "
            "and a property-aware diverse shortlist."
        )
        ml = st.session_state.get("ml_stage")
        if ml is None or ml.prediction_bundle is None:
            st.info("Run continuous pActivity regression in the ML step first.")
        else:
            upload = st.file_uploader("Candidate CSV or TSV", type=["csv", "tsv", "tab"])
            st.caption("Required column: smiles. Optional column: molecule_id.")
            if upload is not None:
                data = upload.getvalue()
                upload_key = (
                    target_id,
                    id(ml.prediction_bundle),
                    hashlib.sha256(data).hexdigest(),
                )
                if st.button("Predict uploaded compounds", type="primary"):
                    try:
                        if len(data) > 2_000_000:
                            raise ValueError("Upload a file smaller than 2 MB.")
                        separator = "\t" if upload.name.lower().endswith((".tsv", ".tab")) else ","
                        frame = pd.read_csv(io.BytesIO(data), sep=separator)
                        with st.spinner("Standardising molecules and predicting potency…"):
                            predicted = predict_candidates(ml.prediction_bundle, frame)
                        st.session_state["uploaded_predictions"] = (upload_key, predicted)
                        st.session_state.pop("shortlist_stage", None)
                    except (ValueError, pd.errors.ParserError, UnicodeError) as exc:
                        st.session_state.pop("uploaded_predictions", None)
                        st.error(str(exc))
                prediction_entry = st.session_state.get("uploaded_predictions")
                if prediction_entry is not None and prediction_entry[0] == upload_key:
                    predicted = prediction_entry[1]
                    valid = predicted[predicted["status"] == "predicted"]
                    counts = st.columns(3)
                    counts[0].metric("Predicted", len(valid))
                    counts[1].metric("Invalid rows", len(predicted) - len(valid))
                    counts[2].metric(
                        "Inside domain",
                        int(valid["in_applicability_domain"].eq(True).sum()) if len(valid) else 0,
                    )
                    st.caption(
                        "Empirical potency bands come from training-fold residuals. "
                        "They are approximate and may undercover novel scaffolds "
                        "or later chemistry."
                    )
                    st.dataframe(predicted, hide_index=True, width="stretch")
                    st.download_button(
                        "Download all predictions",
                        predicted.to_csv(index=False),
                        file_name=f"sarscope_{target_id}_predictions.csv",
                        mime="text/csv",
                    )
                    if len(valid):
                        st.markdown("**Choose shortlist constraints**")
                        controls = st.columns(3)
                        max_mw = controls[0].slider("Maximum MW", 200, 900, 500, 25)
                        max_logp = controls[1].slider("Maximum logP", -1.0, 10.0, 5.0, 0.5)
                        max_tpsa = controls[2].slider("Maximum TPSA", 20, 250, 140, 10)
                        other = st.columns(3)
                        n_shortlist = other[0].slider("Shortlist size", 1, 30, 15)
                        diversity_weight = other[1].slider("Diversity weight", 0.0, 2.0, 1.0, 0.1)
                        domain_only = other[2].checkbox("Inside domain only", value=True)
                        criteria = (
                            max_mw,
                            max_logp,
                            max_tpsa,
                            n_shortlist,
                            diversity_weight,
                            domain_only,
                        )
                        if st.button("Build diverse shortlist"):
                            shortlist = diverse_shortlist(
                                predicted,
                                ml.prediction_bundle,
                                n=n_shortlist,
                                max_mw=max_mw,
                                max_logp=max_logp,
                                max_tpsa=max_tpsa,
                                in_domain_only=domain_only,
                                diversity_weight=diversity_weight,
                            )
                            st.session_state["shortlist_stage"] = (upload_key, criteria, shortlist)
                        chosen = st.session_state.get("shortlist_stage")
                        if chosen is not None and chosen[:2] == (upload_key, criteria):
                            shortlist = chosen[2]
                            if shortlist.empty:
                                st.info("No uploaded compounds meet these constraints.")
                            else:
                                st.caption(
                                    "Ranking balances predicted pActivity with fingerprint "
                                    "diversity. Inspect analogue evidence and domain flags "
                                    "before choosing experiments."
                                )
                                st.dataframe(shortlist, hide_index=True, width="stretch")
                                st.download_button(
                                    "Download shortlist",
                                    shortlist.to_csv(index=False),
                                    file_name=f"sarscope_{target_id}_shortlist.csv",
                                    mime="text/csv",
                                )


main()

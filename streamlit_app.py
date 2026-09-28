"""SARscope in the browser. Run locally with ``streamlit run streamlit_app.py``.

Analyses run in the browser process, so the heavy steps are opt-in: the
sidebar caps the dataset and the model bake-off, because a free hosting tier
has one shared core and a memory ceiling. The CLI (``sarscope run``) has no
such caps and writes the full report folder.
"""

from __future__ import annotations

import collections
import io
import sys
import zipfile
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

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

from sarscope import __version__, provenance  # noqa: E402
from sarscope.__main__ import FETCH_SUMMARY_FIELDS, default_cache_dir  # noqa: E402
from sarscope.analysis.descriptors import add_descriptors  # noqa: E402
from sarscope.analysis.domain import DomainResult, pca_bounding_box  # noqa: E402
from sarscope.analysis.explain import (  # noqa: E402
    DescriptorExplanation,
    explain_descriptor_model,
)
from sarscope.analysis.features import (  # noqa: E402
    VarianceCorrelationFilter,
    bit_vectors,
    fingerprint_matrix,
)
from sarscope.analysis.landscape import (  # noqa: E402
    SasResult,
    cliff_generators,
    consensus,
    sas_map,
)
from sarscope.analysis.mmp import matched_molecular_pairs  # noqa: E402
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
from sarscope.analysis.rgroups import ScaffoldSar, decompose_top_scaffolds  # noqa: E402
from sarscope.analysis.scaffolds import (  # noqa: E402
    ENRICHMENT_COLUMNS,
    add_scaffolds,
    diversity_table,
    enrichment_table,
)
from sarscope.curate import CurationResult, curate_chembl  # noqa: E402
from sarscope.depict import to_svg, unavailable_reason  # noqa: E402
from sarscope.params import (  # noqa: E402
    ClassScheme,
    CurationParams,
    FeatureParams,
    LandscapeParams,
    ModelParams,
    RunParams,
)
from sarscope.pipeline import (  # noqa: E402
    RunResults,
    _cliff_model_errors,
    _modelling_blocked,
)
from sarscope.predict import PredictionBundle, similarity_domain_threshold  # noqa: E402
from sarscope.report import write_report  # noqa: E402
from sarscope.sources.chembl import (  # noqa: E402
    ChemblClient,
    ChemblError,
    normalise_target_id,
)

#: Categorical slots of the reference palette. Group 1 / Group 2 keep these
#: hues everywhere in the app, so colour follows the entity, never the rank.
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
GROUP_COLORS = [BLUE, ORANGE]

#: Above this many molecules, the all-pairs landscape gets slow in a shared
#: process. The CLI has no cap.
LANDSCAPE_WARN = 4000

ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]

#: What each summarised field means, and the curation decision it drives.
FIELD_NOTES: dict[str, tuple[str, str]] = {
    "standard_relation": (
        "Relation",
        '"=" is a measured value. "<" and ">" are bounds (censored), dropped by default.',
    ),
    "standard_units": ("Units", "Only molar units convert to the -log10(M) scale."),
    "assay_type": ("Assay type", "B binding, F functional, A ADMET. Binding kept by default."),
    "assay_variant_mutation": (
        "Protein variant",
        "(none) is wild-type. Mutant assays measure a different protein; "
        "wild-type only by default.",
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


# -- data ---------------------------------------------------------------------


def client() -> ChemblClient:
    return ChemblClient(cache_dir=default_cache_dir())


@st.cache_data(ttl=86_400, show_spinner=False)
def lookup(target_id: str, types: tuple[str, ...]) -> tuple[dict[str, Any], str, int]:
    with client() as c:
        return c.target(target_id), c.release, c.count_activities(target_id, types)


@st.cache_data(ttl=86_400, show_spinner=False)
def fetch(target_id: str, types: tuple[str, ...]) -> list[dict[str, Any]]:
    with client() as c:
        return c.activities(target_id, types)


def run_curation_stage(
    target_id: str,
    types: tuple[str, ...],
    params: RunParams,
    release: str,
    target_name: str,
    organism: str,
) -> CurationStage:
    """Download and curate only; every later analysis is explicitly opt-in."""
    records = fetch(target_id, types)
    curation = curate_chembl(records, params)
    target = {
        "target_chembl_id": target_id,
        "pref_name": target_name,
        "organism": organism,
    }
    record = provenance.collect(params, provenance.chembl_source(target, release, len(records)))
    return CurationStage(params, curation, curation.table, record)


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
    params: RunParams,
    *,
    classification: bool,
    regression: bool,
    landscape: LandscapeStage | None,
) -> MlStage:
    table = scaffold.table
    reason = _modelling_blocked(table, params)
    if reason:
        return MlStage(
            params,
            table,
            None,
            None,
            None,
            skipped={"model": reason, "domain": reason},
        )
    features = params.model.features
    X = fingerprint_matrix(
        table["smiles"].tolist(), features.fingerprint, ecfp_bits=features.ecfp_bits
    )
    years = table["document_year"].tolist()
    models = (
        evaluate(
            X,
            table["activity_class"].tolist(),
            table["murcko"].tolist(),
            params.model,
            years=years,
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
        )
        if regression
        else None
    )
    selected = regression_result or models
    assert selected is not None
    filt = VarianceCorrelationFilter(
        features.variance_threshold, features.correlation_threshold
    ).fit(X[selected.train_index])
    domain = pca_bounding_box(
        filt.transform(X[selected.train_index]), filt.transform(X[selected.test_index])
    )

    test_predictions = pd.DataFrame()
    cliff_performance = pd.DataFrame()
    if models is not None and regression_result is not None and landscape is not None:
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
            features=features,
            feature_filter=deploy_filter,
            estimator=deploy_model,
            train_fingerprints=X,
            similarity_threshold=similarity_domain_threshold(X),
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
    )


def build_params(settings: dict[str, Any]) -> RunParams:
    variant = None if settings["variant"] == "Wild-type only" else settings["variant"]
    if variant == "All (pooled)":
        variant = "any"
    return RunParams(
        curation=CurationParams(
            standard_types=tuple(settings["types"]),
            relations=("=", "<", ">", "<=", ">=") if settings["censored"] else ("=",),
            variant=variant,
            max_document_year=settings["max_year"],
        ),
        classes=ClassScheme(),
    )


# -- charts -------------------------------------------------------------------


def bar_chart(frame: pd.DataFrame, value: str, label: str) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(color=BLUE, cornerRadiusEnd=4, size=14)
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
                scale=alt.Scale(range=GROUP_COLORS),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[alt.Tooltip("count()", title="Molecules"), "group_label:N"],
        )
        .properties(height=190)
    )


def pca_chart(results: RunResults) -> alt.Chart:
    frame = results.pca.scores.copy()
    frame["Group"] = ["Group 1" if g == 1 else "Group 2" for g in results.table["group"]]
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
                "Group:N",
                scale=alt.Scale(domain=["Group 1", "Group 2"], range=GROUP_COLORS),
                legend=alt.Legend(orient="top"),
            ),
            tooltip=[
                alt.Tooltip("molecule:N", title="Molecule"),
                alt.Tooltip("pactivity:Q", title="Potency", format=".2f"),
                "Group:N",
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
                (r.scaffold, f"{r.n} molecules · {r.frac_group1:.0%} Group 1 · EF {r.ef:.2f}")
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


def show_curation(results: RunResults) -> None:
    table = results.table
    cols = st.columns(4)
    cols[0].metric("Molecules", f"{len(table):,}")
    cols[1].metric("Group 1 (potent + active)", f"{int((table['group'] == 1).sum()):,}")
    cols[2].metric("Group 2", f"{int((table['group'] == 2).sum()):,}")
    cols[3].metric("Rejected structures", f"{len(results.curation.rejected):,}")

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
    for step, reason in results.skipped.items():
        st.warning(f"**{step} skipped.** {reason}")


def show_properties(results: RunResults) -> None:
    from sarscope.analysis.descriptors import CORE_DESCRIPTORS

    table = results.table.copy()
    table["group_label"] = ["Group 1" if g == 1 else "Group 2" for g in table["group"]]
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
            st.dataframe(results.profile.stats.reset_index(), hide_index=True, width="stretch")
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


def show_scaffolds(results: RunResults) -> None:
    st.caption(
        "Ns scaffolds, Nss of them carrying a single molecule, Ncsk cyclic skeletons. "
        'The skeleton columns use RDKit\'s generic scaffold; "cyclic skeleton" has no single '
        "definition, so skeleton counts from different tools are not comparable."
    )
    st.dataframe(results.diversity.reset_index(names="class"), width="stretch")

    st.markdown("**Scaffold enrichment**")
    st.caption(
        "EF is the Group 1 fraction within a scaffold over the Group 1 fraction of the whole "
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
                f"{row.n} molecules · {row.frac_group1:.0%} Group 1 · EF {row.ef:.2f}",
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
                "frac_group1": st.column_config.NumberColumn("Group 1 fraction", format="percent"),
                "ef": st.column_config.NumberColumn("EF", format="%.3f"),
                "ef_lower": st.column_config.NumberColumn("EF lower bound", format="%.3f"),
            },
        )


def show_rgroups(results: RunResults) -> None:
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


def show_landscape(results: RunResults) -> None:
    st.caption(
        "Every pair of molecules, placed by structural similarity and potency difference. "
        "Cliffs are similar pairs with very different potency: the pairs a model gets wrong "
        "and a chemist learns from."
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
            top = sas.cliffs.head(30)
            options = [
                f"{r.id_a} / {r.id_b} — similarity {r.similarity:.2f}, "
                f"delta {r.delta:.2f} log units"
                for r in top.itertuples()
            ]
            picked = st.selectbox("Cliff pair", options, key=f"cliff_{sas.fingerprint}")
            pair = top.iloc[options.index(picked)]
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
                "train_accuracy",
                "cv_accuracy",
                "test_accuracy",
                "test_mcc",
            ]
        ]
        st.dataframe(
            show,
            hide_index=True,
            width="stretch",
            column_config={
                c: st.column_config.NumberColumn(c.replace("_", " ").title(), format="%.3f")
                for c in ["train_accuracy", "cv_accuracy", "test_accuracy", "test_mcc"]
            },
        )
        st.markdown(f"Best classifier by CV MCC: **{results.models.best_algorithm}**")

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
        st.markdown(
            f"Best regressor by cross-validated RMSE: **{results.regression.best_algorithm}**"
        )
    if not results.cliff_model_performance.empty:
        st.markdown("**Held-out error on cliff compounds versus the rest**")
        st.dataframe(results.cliff_model_performance, hide_index=True, width="stretch")

    if results.domain is not None:
        st.metric(
            "Test compounds inside the applicability domain", f"{results.domain.coverage:.1%}"
        )
        st.caption(
            "A PCA bounding box in two components is a generous criterion: a molecule can sit "
            "inside it and still be far from every training compound."
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
    "scaffold_stage",
    "landscape_stage",
    "ml_stage",
    "explanation_stage",
    "report_zip",
)


def clear_workflow(*, keep_curation: bool = False) -> None:
    for key in WORKFLOW_KEYS:
        if keep_curation and key == "curation_stage":
            continue
        st.session_state.pop(key, None)


def importance_chart(frame: pd.DataFrame, title: str) -> alt.Chart:
    shown = frame.head(12).sort_values("importance")
    return (
        alt.Chart(shown)
        .mark_bar(color=BLUE, cornerRadiusEnd=3)
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
        model_test_predictions=(ml.model_test_predictions if ml is not None else pd.DataFrame()),
        cliff_model_performance=(ml.cliff_model_performance if ml is not None else pd.DataFrame()),
        prediction_bundle=ml.prediction_bundle if ml is not None else None,
        rgroups=scaffolds.rgroups,
        skipped={**curation.skipped, **properties.skipped, **(ml.skipped if ml else {})},
        provenance=curation.provenance,
    )


# -- page ---------------------------------------------------------------------


def sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Settings")
        st.caption("Fast browser defaults; the CLI remains available for exhaustive runs.")

        with st.expander("Curation", expanded=True):
            types = st.multiselect("Activity types", ACTIVITY_TYPES, default=["IC50"])
            variant = st.selectbox(
                "Protein variant", ["Wild-type only", "V600E", "All (pooled)"], index=0
            )
            censored = st.checkbox("Keep censored values (>, <)", value=False)
            max_year = st.number_input(
                "Only documents up to year", min_value=1990, max_value=2030, value=2030
            )

        st.caption(f"SARscope {__version__} · [source](https://github.com/yboulaamane/sarscope)")

    return {
        "types": types,
        "variant": variant,
        "censored": censored,
        "max_year": None if max_year >= 2030 else int(max_year),
    }


def main() -> None:
    st.set_page_config(page_title="SARscope", layout="wide", page_icon="🔬")
    settings = sidebar()

    st.title("SARscope")
    st.markdown(
        "Turn a ChEMBL target into an auditable, stepwise medicinal-chemistry analysis: "
        "curate assay records, inspect chemical space and SAR, then build and explain "
        "predictive models only when the data support them."
    )
    overview = st.columns(3)
    overview[0].caption(
        "**1 · Curate**  Confirm the target and activity types, then standardise and "
        "filter experimental records."
    )
    overview[1].caption(
        "**2 · Understand SAR**  Explore physicochemical space, scaffolds, matched pairs, "
        "R-groups, and activity cliffs."
    )
    overview[2].caption(
        "**3 · Model carefully**  Compare validation strategies, inspect errors and "
        "feature effects, and check applicability."
    )
    st.caption(
        "QSAR predictions support compound prioritisation; they do not by themselves "
        "establish binding, selectivity, safety, or experimental activity."
    )

    reason = unavailable_reason()
    if reason:
        # Everything still works; structures fall back to SMILES text.
        st.warning(f"Structures cannot be drawn here. {reason}")

    left, right = st.columns([3, 1], vertical_alignment="bottom")
    raw_id = left.text_input(
        "ChEMBL target ID", value="CHEMBL5145", help="e.g. CHEMBL5145 (BRAF) or just 5145"
    )
    check_target = right.button("Check target", type="primary", width="stretch")

    if not settings["types"]:
        st.info("Choose at least one activity type in the sidebar.")
        return
    try:
        target_id = normalise_target_id(raw_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    lookup_key = (target_id, tuple(settings["types"]))
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
        st.info("Enter a ChEMBL ID and press **Check target**. No network request runs at startup.")
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
        release,
    )
    action_cols = st.columns(2)
    run_curation = action_cols[0].button(
        f"1 · Curate {resolved_id}", type="primary", width="stretch"
    )
    load_raw = action_cols[1].button("Inspect raw fields", width="stretch")
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
        try:
            with st.spinner("Downloading and curating activity records…"):
                stage = run_curation_stage(
                    target_id,
                    tuple(settings["types"]),
                    base_params,
                    release,
                    str(target["pref_name"]),
                    str(target["organism"]),
                )
        except (ChemblError, ValueError) as exc:
            st.error(f"Curation stopped: {exc}")
            return
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

    tabs = st.tabs(
        [
            "1 · Curation",
            "2 · Chemical space",
            "3 · Scaffolds & SAR",
            "4 · Activity cliffs",
            "5 · ML",
            "6 · Explain",
            "7 · Report",
        ]
    )
    with tabs[0]:
        show_curation(curation)
        with st.expander("Curated molecules"):
            st.dataframe(curation.table, hide_index=True, width="stretch")

    with tabs[1]:
        st.caption(
            "Compute named RDKit physicochemical descriptors, compare activity groups and "
            "project the compounds into standardized descriptor PCA space."
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
                    st.session_state["scaffold_stage"] = scaffolds
                    st.session_state.pop("report_zip", None)
            if scaffolds.rgroups:
                show_rgroups(scaffolds)
            if not scaffolds.matched_pairs.empty:
                st.markdown("**Matched molecular pairs**")
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
                "This dataset is large for a hosted all-pairs calculation; use one fingerprint."
            )
        if st.button("Run activity-cliff analysis", key="run_landscape"):
            if not fingerprints:
                st.error("Choose at least one fingerprint.")
            else:
                landscape_params = LandscapeParams(
                    fingerprints=tuple(fingerprints),
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
            algorithms = st.multiselect("Algorithms", FAST_ALGORITHMS, default=["random_forest"])
            ml_cols = st.columns(3)
            model_fp = ml_cols[0].selectbox("Fingerprint", ["ecfp4", "maccs"])
            split = ml_cols[1].selectbox("Validation split", ["scaffold", "time", "random"])
            cv_folds = ml_cols[2].slider("CV folds", 3, 10, 5)
            time_cutoff = st.number_input(
                "Training cutoff year (time split only)", 1950, 2029, 2019
            )
            audit = st.checkbox("Run classification leakage audit", value=False)
            if st.button("Run selected ML", key="run_ml", type="primary"):
                if not tasks or not algorithms:
                    st.error("Choose at least one task and one algorithm.")
                else:
                    model_params = ModelParams(
                        features=FeatureParams(fingerprint=model_fp),
                        algorithms=tuple(algorithms),
                        regression_algorithms=tuple(algorithms),
                        split=split,
                        time_cutoff=int(time_cutoff),
                        cv_folds=cv_folds,
                        leakage_audit=audit,
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
                    with st.spinner("Running only the selected models and validation…"):
                        st.session_state["ml_stage"] = run_ml_stage(
                            scaffolds,
                            run_params,
                            classification="Activity-class classification" in tasks,
                            regression="Continuous pActivity regression" in tasks,
                            landscape=landscape,
                        )
                    st.session_state.pop("explanation_stage", None)
                    st.session_state.pop("report_zip", None)
            ml: MlStage | None = st.session_state.get("ml_stage")
            if ml is None:
                st.info("No ML run yet.")
            else:
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
                    "Optional SHAP is not installed on this host; permutation importance "
                    "remains available."
                )
            if st.button("Run descriptor explanation", key="run_explanation"):
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


main()

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
from importlib import resources
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
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sarscope import __version__, provenance  # noqa: E402
from sarscope.__main__ import FETCH_SUMMARY_FIELDS, default_cache_dir  # noqa: E402
from sarscope.analysis.landscape import cliff_generators  # noqa: E402
from sarscope.curate import curate_chembl  # noqa: E402
from sarscope.depict import to_svg, unavailable_reason  # noqa: E402
from sarscope.params import (  # noqa: E402
    ClassScheme,
    CurationParams,
    FeatureParams,
    LandscapeParams,
    ModelParams,
    RunParams,
)
from sarscope.pipeline import RunResults, analyse  # noqa: E402
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

FAST_ALGORITHMS = ["extra_trees", "random_forest", "gradient_boosting", "nearest_neighbors"]


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


@st.cache_resource(ttl=3600, show_spinner=False, max_entries=3)
def run_analysis(
    target_id: str, types: tuple[str, ...], params: RunParams, release: str
) -> RunResults:
    """Cached on the settings, so changing a slider re-runs but a redraw does not.

    ``release`` is part of the key: a new ChEMBL release must invalidate it.
    """
    records = fetch(target_id, types)
    results = analyse(curate_chembl(records, params), params)
    with client() as c:
        target = c.target(target_id)
    results.provenance = provenance.collect(
        params, provenance.chembl_source(target, release, len(records))
    )
    return results


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
        landscape=LandscapeParams(fingerprints=tuple(settings["fingerprints"])),
        model=ModelParams(
            features=FeatureParams(fingerprint=settings["model_fp"]),
            algorithms=tuple(settings["algorithms"]),
            split=settings["split"],
            time_cutoff=settings["time_cutoff"],
            cv_folds=settings["cv_folds"],
            leakage_audit=settings["audit"],
        ),
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


def show_models(results: RunResults) -> None:
    if results.models is None:
        st.info(results.skipped.get("model", "Modelling did not run."))
        return
    st.caption(
        "**leak_free** selects features and resamples inside training folds only, after the "
        "split. **naive** is the common ordering — select and oversample on everything, then "
        "split — which puts copies of training molecules in the test set and picks features "
        "using the held-out rows. The gap between the two rows is how much that ordering "
        "would have flattered these models."
    )
    scores = results.models.scores
    show = scores[
        ["algorithm", "protocol", "train_accuracy", "cv_accuracy", "test_accuracy", "test_mcc"]
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
    st.markdown(f"Best model by cross-validated MCC: **{results.models.best_algorithm}**")

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


@st.cache_data(show_spinner=False)
def demo_data() -> pd.DataFrame:
    """Tiny packaged dataset: no API call and no analysis wait."""
    resource = resources.files("sarscope").joinpath("data/demo.csv")
    with resource.open("rb") as handle:
        return pd.read_csv(handle)


def show_demo_preview() -> None:
    data = demo_data()
    with st.expander("Instant demo dataset", expanded=True):
        st.caption(
            "Bundled locally so a cold app has something useful before ChEMBL responds. "
            "The pActivity values are synthetic and illustrative, not experimental evidence."
        )
        cols = st.columns(3)
        cols[0].metric("Demo molecules", len(data))
        potency_range = f"{data['pactivity'].min():.1f}–{data['pactivity'].max():.1f}"
        cols[1].metric("Potency range", potency_range)
        cols[2].metric("Network wait", "none")
        structure_grid(
            [
                (row.smiles, f"{row.molecule_id} · synthetic pActivity {row.pactivity:.1f}")
                for row in data.nlargest(4, "pactivity").itertuples()
            ]
        )


# -- page ---------------------------------------------------------------------


def sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown("### Settings")
        st.caption("Every default matches the reference workflow unless the label says otherwise.")

        with st.expander("Curation", expanded=True):
            types = st.multiselect("Activity types", ACTIVITY_TYPES, default=["IC50"])
            variant = st.selectbox(
                "Protein variant", ["Wild-type only", "V600E", "All (pooled)"], index=0
            )
            censored = st.checkbox("Keep censored values (>, <)", value=False)
            max_year = st.number_input(
                "Only documents up to year", min_value=1990, max_value=2030, value=2030
            )

        with st.expander("Landscape"):
            fingerprints = st.multiselect(
                "Fingerprints", ["ecfp4", "maccs"], default=["ecfp4", "maccs"]
            )

        with st.expander("Models"):
            algorithms = st.multiselect(
                "Algorithms", FAST_ALGORITHMS, default=["extra_trees", "random_forest"]
            )
            model_fp = st.selectbox("Model fingerprint", ["ecfp4", "maccs"], index=0)
            split = st.radio(
                "Split",
                ["scaffold", "time", "random"],
                index=0,
                help=(
                    "Scaffold keeps a series wholly on one side; time reserves compounds "
                    "first documented after the cutoff."
                ),
            )
            time_cutoff = st.number_input(
                "Time-split training cutoff", min_value=1950, max_value=2029, value=2019
            )
            cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
            audit = st.checkbox(
                "Run the leakage audit",
                value=True,
                help="Also runs the reference workflow's order and reports the gap.",
            )

        st.caption(f"SARscope {__version__} · [source](https://github.com/yboulaamane/sarscope)")

    return {
        "types": types,
        "variant": variant,
        "censored": censored,
        "max_year": None if max_year >= 2030 else int(max_year),
        "fingerprints": fingerprints,
        "algorithms": algorithms,
        "model_fp": model_fp,
        "split": split,
        "time_cutoff": int(time_cutoff),
        "cv_folds": cv_folds,
        "audit": audit,
    }


def main() -> None:
    st.set_page_config(page_title="SARscope", layout="wide", page_icon="🔬")
    settings = sidebar()

    st.title("SARscope")
    st.caption("Target ID in, structure–activity report out.")
    if "analysed" not in st.session_state:
        show_demo_preview()

    reason = unavailable_reason()
    if reason:
        # Everything still works; structures fall back to SMILES text.
        st.warning(f"Structures cannot be drawn here. {reason}")

    left, right = st.columns([3, 1])
    raw_id = left.text_input(
        "ChEMBL target ID", value="CHEMBL5145", help="e.g. CHEMBL5145 (BRAF) or just 5145"
    )
    right.write("")
    go = right.button("Analyse", type="primary", width="stretch")

    if not settings["types"]:
        st.info("Choose at least one activity type in the sidebar.")
        return
    try:
        target_id = normalise_target_id(raw_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Asking ChEMBL…"):
            target, release, n_records = lookup(target_id, tuple(settings["types"]))
    except ChemblError as exc:
        st.error(str(exc))
        return

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

    if go:
        st.session_state["analysed"] = (target_id, tuple(settings["types"]))
    if st.session_state.get("analysed") != (target_id, tuple(settings["types"])):
        st.info("Press **Analyse** to curate these records and run the full analysis.")
        with st.expander("Raw records, before curation"):
            records = fetch(target_id, tuple(settings["types"]))
            for row in range(0, len(FETCH_SUMMARY_FIELDS), 2):
                for col, field in zip(
                    st.columns(2), FETCH_SUMMARY_FIELDS[row : row + 2], strict=False
                ):
                    title, note = FIELD_NOTES[field]
                    with col:
                        st.markdown(f"**{title}**")
                        st.caption(note)
                        st.altair_chart(
                            bar_chart(breakdown(records, field), "records", "Records"),
                            width="stretch",
                        )
        return

    params = build_params(settings)
    try:
        with st.spinner("Curating, then running every analysis. The first run takes a minute…"):
            results = run_analysis(target_id, tuple(settings["types"]), params, release)
    except ChemblError as exc:
        st.error(str(exc))
        return
    except ValueError as exc:
        st.error(f"Analysis stopped: {exc}")
        return

    if len(results.table) > LANDSCAPE_WARN:
        st.warning(
            f"{len(results.table):,} molecules means about "
            f"{len(results.table) ** 2 // 2:,} pairs. That ran, but for datasets this size "
            "the command line is faster: `sarscope run " + target_id + " --out report/`"
        )

    tabs = st.tabs(
        [
            "Overview",
            "Curation",
            "Properties",
            "Scaffolds",
            "R-group SAR",
            "Landscape",
            "Models",
            "Report",
        ]
    )
    with tabs[0]:
        show_overview(results)
    with tabs[1]:
        show_curation(results)
    with tabs[2]:
        show_properties(results)
    with tabs[3]:
        show_scaffolds(results)
    with tabs[4]:
        show_rgroups(results)
    with tabs[5]:
        show_landscape(results)
    with tabs[6]:
        show_models(results)
    with tabs[7]:
        st.caption(
            "The same folder `sarscope run` writes: a self-contained HTML report, every table "
            "as CSV, every figure as PNG, and provenance.json recording the exact settings, "
            "ChEMBL release and package versions behind these numbers."
        )
        st.download_button(
            "Download the report folder (.zip)",
            data=report_zip(results),
            file_name=f"sarscope_{target_id}.zip",
            mime="application/zip",
            type="primary",
        )
        st.json(results.provenance, expanded=False)


main()

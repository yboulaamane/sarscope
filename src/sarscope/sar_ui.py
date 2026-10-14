"""Streamlit's interactive scaffold SAR workspace and portable evidence export."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
from rdkit.Chem import Descriptors

from sarscope.analysis.rgroups import ScaffoldSar, core_states, select_core_state
from sarscope.analysis.sar_map import (
    SarMap,
    build_sar_map,
    filter_series,
    fragment_label,
    rank_transformations,
)
from sarscope.analysis.sar_stress import SarStressTest, claim_pairs, stress_test_sar
from sarscope.curate import assign_classes
from sarscope.depict import to_svg
from sarscope.params import ClassScheme
from sarscope.sar_depict import sar_map_svg

_component = components.declare_component(
    "sarscope_sar_map", path=str(Path(__file__).parent / "components" / "sar_map")
)


def sar_map_archive(
    result: SarMap,
    scope: dict[str, Any],
    svg: str | None,
    *,
    evidence: pd.DataFrame | None = None,
    structures: pd.DataFrame | None = None,
) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, frame in (
            ("position_evidence", result.transformations),
            ("matched_position_pairs", result.pairs),
            ("substituent_statistics", result.series.substituents),
            ("series_members", result.series.members),
            ("annotations", result.annotations),
        ):
            archive.writestr(f"{name}.csv", frame.to_csv(index=False))
        if svg:
            archive.writestr("sar_map.svg", svg)
        if evidence is not None:
            archive.writestr("retained_measurements.csv", evidence.to_csv(index=False))
        if structures is not None:
            included = structures[
                structures["molecule_id"].isin(result.series.members["molecule_id"])
            ]
            archive.writestr("series_structures.csv", included.to_csv(index=False))
        archive.writestr(
            "scope.json",
            json.dumps(
                {
                    **scope,
                    "scaffold": result.series.scaffold,
                    "labelled_core": result.series.core,
                    "effect_threshold_log_units": result.effect_threshold,
                    "pair_limit_reached": result.pair_limit_reached,
                    "excluded_core_molecules": result.excluded_core_molecules,
                    "retained_record_ids": result.record_ids,
                    "method": "Single varying R-group; identical mapped isomeric core and all "
                    "other labelled positions held constant. Shared-assay deltas use exact "
                    "measurement medians per molecule/assay/endpoint/variant, deduplicating "
                    "copies of the same record ID/value. "
                    "Summary gives one vote per molecular pair; disjoint pairs share no compounds.",
                },
                indent=2,
                default=str,
            ),
        )
    return buffer.getvalue()


def _structure(
    smiles: str,
    caption: str,
    *,
    size: tuple[int, int] = (320, 220),
) -> None:
    if smiles == "H":
        st.markdown("**Hydrogen (unsubstituted site)**")
        return
    svg = to_svg(smiles, size)
    if svg:
        st.image(svg, caption=caption)
    else:
        st.code(smiles)


def _pair_view(pair: pd.Series, table: pd.DataFrame, key: str) -> None:
    cols = st.columns(2)
    molecules = []
    for col, suffix in zip(cols, ("from", "to"), strict=True):
        identifier = pair[f"id_{suffix}"]
        block = table[table["molecule_id"] == identifier]
        if block.empty:
            continue
        smiles = str(block.iloc[0]["smiles"])
        molecules.append(Chem.MolFromSmiles(smiles))
        with col:
            _structure(smiles, f"{identifier} · pActivity {pair[f'pactivity_{suffix}']:.2f}")
            if str(identifier).startswith("CHEMBL"):
                st.markdown(
                    f"[ChEMBL compound](https://www.ebi.ac.uk/chembl/explore/compound/{identifier})"
                )
    if len(molecules) == 2 and all(molecule is not None for molecule in molecules):
        changes = []
        for label, function in (
            ("Molecular weight", Descriptors.MolWt),
            ("Calculated logP", Descriptors.MolLogP),
            ("TPSA (Å²)", Descriptors.TPSA),
            ("H-bond donors", Descriptors.NumHDonors),
            ("H-bond acceptors", Descriptors.NumHAcceptors),
            ("Rotatable bonds", Descriptors.NumRotatableBonds),
        ):
            before, after = function(molecules[0]), function(molecules[1])
            changes.append(
                {"Property": label, "From": before, "To": after, "Change": after - before}
            )
        st.dataframe(pd.DataFrame(changes), hide_index=True, width="stretch")
    st.caption(
        f"ΔpActivity (to − from) = {pair['delta_pactivity']:+.2f} · "
        f"{pair['evidence_tier']} · {pair['assay_id'] or 'No shared assay'} "
        f"{pair['endpoint']} · variant {pair['variant'] or 'unknown'}. "
        "Physicochemical differences describe the structures; they do not establish a mechanism."
    )


def show_sar_map(
    series: list[ScaffoldSar],
    table: pd.DataFrame,
    evidence: pd.DataFrame,
    *,
    classes: ClassScheme,
    source: dict[str, Any] | None = None,
) -> None:
    st.markdown("### Interactive SAR map")
    st.caption(
        "Choose a chemical series and build its map. Each colored R-site links to the "
        "substitutions and measured compounds behind the annotation."
    )
    with st.expander("How to read this map"):
        st.markdown(
            "**R1, R2, …** mark places where compounds differ on a shared core. "
            "Click a colored circle or its annotation to open the evidence below.\n\n"
            "**Positive ΔpActivity** means the 'to' substituent is more potent in the "
            "observed comparison. +1 is about a tenfold lower measured concentration; "
            "+2 is about a hundredfold lower concentration.\n\n"
            "**Green / red:** supported increases / decreases. **Amber:** mixed or small "
            "changes. **Grey:** too little matched evidence. Grey does not mean inactive.\n\n"
            "Evidence is strongest when other substitution sites stay the same and both "
            "compounds were measured in the same assay. The map summarizes this dataset; "
            "it cannot establish which unchanged core atoms are essential. "
            "Grades describe evidence amount and consistency; they are not significance tests "
            "or calibrated confidence probabilities."
        )
    if not series:
        st.info("Run R-group decomposition to create the series map.")
        return
    index = st.selectbox(
        "SAR series",
        list(range(len(series))),
        format_func=lambda i: (
            f"Series {i + 1} · {series[i].n_molecules} compounds · "
            f"{len(series[i].positions)} positions"
        ),
    )
    selected = series[index]
    states = core_states(selected)
    series_key = hashlib.sha256(selected.members.to_csv(index=False).encode()).hexdigest()[:12]
    if len(states) > 1:
        chosen_core = st.selectbox(
            "Core stereochemical state",
            states,
            format_func=lambda core: (
                f"State {states.index(core) + 1} · "
                f"{int((selected.members.core_smiles == core).sum())} compounds"
            ),
            key=f"sar_core_{series_key}",
            help="Mapped isomeric core identities are kept separate. The largest state is "
            "selected by count, not potency; unassigned stereo is not treated as assigned.",
        )
        selected = select_core_state(selected, chosen_core)
        st.info(
            f"This scaffold contains {len(states)} mapped core states. Only the selected "
            f"{selected.n_molecules} compounds are compared; other states are not pooled. "
            "State numbers identify structures, not activity classes."
        )
    else:
        selected = select_core_state(selected)
    key = "sar_map_" + hashlib.sha256(selected.core.encode()).hexdigest()[:12]
    records = (
        evidence[evidence["molecule_id"].isin(selected.members["molecule_id"])].copy()
        if not evidence.empty
        else evidence.copy()
    )
    filters: dict[str, list[Any]] = {}
    with st.expander("Refine the SAR evidence", expanded=False):
        fields = (
            ("standard_type", "SAR endpoint"),
            ("assay_chembl_id", "SAR assays"),
            ("source_origin", "SAR data origins"),
            ("assay_variant_mutation", "SAR variant annotations"),
        )
        for field, label in fields:
            if field not in records:
                continue
            values = records[field].fillna("Unreported")
            options = sorted(values.unique().tolist(), key=str)
            filters[field] = st.multiselect(label, options, default=options, key=f"{key}_{field}")
        if "document_year" in records:
            years = records["document_year"].fillna("Unreported")
            options = sorted(years.unique().tolist(), key=str)
            filters["document_year"] = st.multiselect(
                "SAR publication years", options, default=options, key=f"{key}_year"
            )
        effect = st.slider(
            "Minimum SAR effect (log units)",
            0.1,
            2.0,
            0.5,
            0.1,
            key=f"{key}_effect",
            help="Effects smaller than this are labelled small/mixed. "
            "This threshold is recorded in exports.",
        )
        st.caption(
            "Filters reuse curated measurements and recompute medians. Unreported variant "
            "does not mean wild type. Shared-assay comparisons always keep endpoint and "
            "variant together. Displayed molecule distributions can still contain pooled endpoints."
        )
    scope = {
        "filters": filters,
        "source": source or {},
        "classes": asdict(classes),
        "core_state": selected.core,
        "available_core_states": states,
    }
    # Include data content, not just lengths: values and records can change.
    signature = hashlib.sha256(
        (
            selected.members.to_csv(index=False)
            + records.to_csv(index=False)
            + json.dumps(scope, sort_keys=True, default=str)
            + str(effect)
        ).encode()
    ).hexdigest()
    saved = st.session_state.get("sar_map_result")
    rebuild = saved is not None and saved[0] != signature
    if st.button(
        "Update SAR map" if rebuild else "Build interactive SAR map",
        type="primary",
        key=f"{key}_build",
    ):
        with st.spinner("Matching single-position changes and assay evidence…"):
            scoped, scoped_records = filter_series(selected, records, filters)
            scoped.members["activity_class"], _ = assign_classes(
                scoped.members["pactivity"], classes
            )
            result = build_sar_map(scoped, scoped_records, effect_threshold=effect)
            st.session_state["sar_map_result"] = (signature, result)
    saved = st.session_state.get("sar_map_result")
    if saved is None or saved[0] != signature:
        _structure(selected.core, "Preview of this series · labelled substitution positions")
        st.info(
            "This is the shared core for your selected series. Click Build interactive "
            "SAR map above to see substitution effects and their supporting compounds."
            if saved is None
            else "SAR evidence choices changed. Rebuild the map with Update SAR map above."
        )
        return
    result = saved[1]
    if result.series.members.empty:
        st.info("No compounds remain in this evidence subset. Widen the SAR filters.")
        return
    if not selected.positions:
        st.info(
            "This core has no variable attachment positions. Explore matched molecular pairs below."
        )
        return
    if result.pair_limit_reached:
        st.warning(
            "The 20,000-row comparison limit was reached. Narrow the subset; "
            "evidence strength is capped at limited."
        )
    scoped_records = (
        records[records["record_id"].astype(str).isin(result.record_ids)]
        if "record_id" in records
        else records
    )
    position_key = f"{key}_position"
    if st.session_state.get(position_key) not in selected.positions:
        ranked = rank_transformations(result.transformations)
        st.session_state[position_key] = (
            str(ranked.iloc[0]["position"]) if not ranked.empty else selected.positions[0]
        )
    overview = st.columns(3)
    overview[0].metric("Compounds in this series", str(result.series.n_molecules))
    overview[1].metric("Variable positions", str(len(selected.positions)))
    overview[2].metric(
        "Compound pairs with shared assays",
        str(
            result.pairs.query("evidence_tier == 'shared assay'")[["id_from", "id_to"]]
            .drop_duplicates()
            .shape[0]
        ),
    )
    st.caption(
        "Click a highlighted position to explore it. "
        "The strongest supported comparison is selected first."
    )
    svg = sar_map_svg(result, selected=st.session_state[position_key])
    if svg:
        event = _component(svg=svg, key=f"{key}_graphic", default=None)
        if isinstance(event, dict) and event.get("position") in selected.positions:
            if event != st.session_state.get(f"{key}_last_event"):
                st.session_state[f"{key}_last_event"] = event
                st.session_state[position_key] = event["position"]
                st.rerun()
    position = st.selectbox("Inspect SAR position", selected.positions, key=position_key)
    st.caption(
        "Colors describe directional evidence at an R-position. Grey means sparse or unmatched "
        "assay evidence; amber means mixed or small effects. Invariant core atoms and linker "
        "changes that alter the scaffold require separate matched-pair evidence."
    )
    members = result.series.members.copy()
    members["Substituent"] = members[position].map(fragment_label)
    stats = result.series.substituents[result.series.substituents["position"] == position].copy()
    st.markdown(f"**{position} · substituent potency distributions**")
    if len(members) > 2000:
        plotted = members.sample(2000, random_state=42)
        st.caption(
            "Chart samples 2,000 compounds; tables and matched comparisons use the full subset."
        )
    else:
        plotted = members
    if members[position].nunique() > 12:
        coverage = st.selectbox(
            "Substituents in the chart",
            ["12 most common", "All substituents"],
            key=f"{key}_{position}_coverage",
        )
        if coverage == "12 most common":
            common = members[position].value_counts().head(12).index
            plotted = plotted[plotted[position].isin(common)]
            st.caption(
                "The chart shows the 12 most populated substituents; "
                "the evidence retains all alternatives."
            )
    st.altair_chart(
        alt.Chart(plotted)
        .mark_circle(size=55, opacity=0.65, color="#a78bfa")
        .encode(
            x=alt.X(
                "pactivity:Q",
                title="Measured pActivity (higher is more potent)",
                scale=alt.Scale(zero=False),
            ),
            y=alt.Y(
                "Substituent:N",
                title=None,
                sort=alt.EncodingSortField(field="pactivity", op="median", order="descending"),
            ),
            tooltip=["molecule_id:N", "Substituent:N", "pactivity:Q", "activity_class:N"],
        )
        .properties(height=min(600, max(180, 32 * members[position].nunique()))),
        width="stretch",
    )
    st.caption(
        "Substituent medians are descriptive: molecules in different rows can also "
        "differ at other positions."
    )
    friendly = stats[
        ["substituent", "n", "median_pactivity", "min_pactivity", "max_pactivity"]
    ].copy()
    friendly["substituent"] = friendly["substituent"].map(fragment_label)
    st.dataframe(
        friendly.rename(
            columns={
                "substituent": "Substituent",
                "n": "Compounds",
                "median_pactivity": "Median pActivity",
                "min_pactivity": "Lowest pActivity",
                "max_pactivity": "Highest pActivity",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Detailed substituent statistics"):
        st.dataframe(stats, hide_index=True, width="stretch")
    transforms = result.transformations[result.transformations["position"] == position]
    if transforms.empty:
        st.info("No compounds differ only at this position in the selected subset.")
    else:
        _show_transformations(transforms, result, table, scoped_records, key, position)
    with st.expander("Download SAR map and evidence"):
        if svg:
            st.download_button(
                "Download annotated SAR SVG",
                svg,
                file_name="sar_map.svg",
                mime="image/svg+xml",
                key=f"{key}_svg",
            )
        st.download_button(
            "Download SAR evidence package",
            sar_map_archive(result, scope, svg, evidence=scoped_records, structures=table),
            file_name="sar_map_evidence.zip",
            mime="application/zip",
            key=f"{key}_zip",
        )


def _show_transformations(
    transforms: pd.DataFrame,
    result: SarMap,
    table: pd.DataFrame,
    evidence: pd.DataFrame,
    key: str,
    position: str,
) -> None:
    st.markdown("**Controlled substitution evidence**")
    transforms = rank_transformations(transforms)
    chosen = st.selectbox(
        "Inspect transformation",
        transforms.index.tolist(),
        format_func=lambda i: (
            f"{fragment_label(transforms.loc[i, 'fragment_from'])} → "
            f"{fragment_label(transforms.loc[i, 'fragment_to'])} · "
            f"Δ {transforms.loc[i, 'median_delta']:+.2f} · "
            f"{transforms.loc[i, 'evidence_tier']} · {transforms.loc[i, 'endpoint']} · "
            f"{transforms.loc[i, 'variant']}"
        ),
        key=f"{key}_{position}_transform",
    )
    summary = transforms.loc[chosen]
    pairs = result.pairs[
        (result.pairs["position"] == position)
        & (result.pairs["fragment_from"] == summary["fragment_from"])
        & (result.pairs["fragment_to"] == summary["fragment_to"])
        & (result.pairs["evidence_tier"] == summary["evidence_tier"])
        & (result.pairs["endpoint"] == summary["endpoint"])
        & (result.pairs["variant"] == summary["variant"])
    ].copy()
    st.markdown(
        f"**{fragment_label(summary['fragment_from'])} → {fragment_label(summary['fragment_to'])}: "
        f"{summary['interpretation']} · {summary['evidence_strength']} evidence**"
    )
    fragments = st.columns(2)
    for col, direction in zip(fragments, ("from", "to"), strict=True):
        with col:
            _structure(
                str(summary[f"fragment_{direction}"]),
                f"{direction.title()} · {fragment_label(summary[f'fragment_{direction}'])}",
                size=(220, 130),
            )
    delta = float(summary["median_delta"])
    if delta == 0:
        shift = "The median comparison shows no potency shift."
    elif abs(delta) <= 6:
        shift = (
            f"This corresponds to approximately {10 ** abs(delta):,.1f}× "
            f"{'lower' if delta > 0 else 'higher'} measured concentration for the 'to' substituent."
        )
    else:
        shift = "The median concentration difference exceeds a millionfold. "
        shift += "Inspect the source measurements."
    st.markdown(
        f"Across **{int(summary['supporting_pairs'])} matched compound pairs**, changing "
        f"**{fragment_label(summary['fragment_from'])} → "
        f"{fragment_label(summary['fragment_to'])}** "
        f"is associated with a median **{delta:+.2f} pActivity** change. {shift}"
    )
    if summary["evidence_strength"] == "limited":
        st.info(
            "Evidence is limited here. Inspect the individual compounds and assays "
            "before generalizing this substitution."
        )
    elif summary["evidence_strength"] == "mixed":
        st.info(
            "Changes are mixed or below your effect threshold. "
            "This subset does not support one consistent direction."
        )
    cols = st.columns(4)
    cols[0].metric("Median ΔpActivity", f"{summary['median_delta']:+.2f}")
    cols[1].metric("Matched molecular pairs", str(summary["supporting_pairs"]))
    cols[2].metric("Non-overlapping pairs", str(summary["disjoint_pairs"]))
    cols[3].metric("Shared assays", str(summary["assays"]))
    st.caption(
        f"Middle 50% of pair deltas: {summary['q25_delta']:+.2f} to {summary['q75_delta']:+.2f}; "
        f"sign agreement {summary['direction_agreement']:.0%}. This spread is not a confidence "
        "interval. Disjoint pairs share no compounds and are counted conservatively. "
        "Repeated measurements of a pair get one vote in the summary."
    )
    _show_sar_stress(result, summary, evidence, table, key)
    plotted = pairs.head(200).copy()
    plotted["Pair"] = plotted["id_from"] + " → " + plotted["id_to"] + " · " + plotted["assay_id"]
    st.altair_chart(
        alt.Chart(plotted)
        .mark_bar()
        .encode(
            x=alt.X("delta_pactivity:Q", title="ΔpActivity: to − from"),
            y=alt.Y("Pair:N", title=None, sort="-x"),
            color=alt.condition(
                alt.datum.delta_pactivity > 0, alt.value("#10b981"), alt.value("#f87171")
            ),
            tooltip=["Pair:N", "endpoint:N", "variant:N", "delta_pactivity:Q"],
        )
        .properties(height=min(600, max(150, len(plotted) * 24))),
        width="stretch",
    )
    with st.expander("All comparison rows and transformation statistics"):
        st.dataframe(transforms, hide_index=True, width="stretch")
        st.dataframe(pairs, hide_index=True, width="stretch")
    pair_index = st.selectbox(
        "Inspect matched compounds",
        pairs.index.tolist(),
        format_func=lambda i: (
            f"{pairs.loc[i, 'id_from']} → {pairs.loc[i, 'id_to']} · "
            f"{pairs.loc[i, 'assay_id'] or 'pooled'}"
        ),
        key=f"{key}_{position}_{chosen}_pair",
    )
    pair = pairs.loc[pair_index]
    _pair_view(pair, table, key)
    with st.expander("Assay measurements behind this pair"):
        block = (
            evidence[evidence["molecule_id"].isin([pair["id_from"], pair["id_to"]])].copy()
            if not evidence.empty
            else evidence.copy()
        )
        if pair["evidence_tier"] == "shared assay":
            identifiers = set(str(pair["record_ids_from"]).split(";")) | set(
                str(pair["record_ids_to"]).split(";")
            )
            if "record_id" in block:
                block = block[block["record_id"].astype(str).isin(identifiers)]
        st.dataframe(block, hide_index=True, width="stretch")


def sar_stress_archive(
    stress: SarStressTest,
    result: SarMap,
    evidence: pd.DataFrame,
) -> bytes:
    """Portable evidence card, including settings and content hashes."""
    buffer = io.BytesIO()
    files = {
        "sensitivity_checks.csv": stress.checks.to_csv(index=False),
        "context_effects.csv": stress.contexts.to_csv(index=False),
        "threshold_sensitivity.csv": stress.thresholds.to_csv(index=False),
        "matched_cycles.csv": stress.cycles.to_csv(index=False),
        "next_measurements.csv": stress.suggestions.to_csv(index=False),
        "matched_pairs.csv": claim_pairs(result, stress.claim).to_csv(index=False),
        "series_members.csv": result.series.members.to_csv(index=False),
        "retained_measurements.csv": evidence.to_csv(index=False),
    }
    from rdkit import rdBase

    metadata = {
        "claim": stress.claim,
        "verdict": stress.verdict,
        "limitations": stress.limitations,
        "settings": stress.settings,
        "rdkit_version": rdBase.rdkitVersion,
        "sha256": {
            name: hashlib.sha256(content.encode()).hexdigest() for name, content in files.items()
        },
    }
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
        archive.writestr("evidence_card.json", json.dumps(metadata, indent=2, default=str))
    return buffer.getvalue()


def _show_sar_stress(
    result: SarMap,
    claim: pd.Series,
    evidence: pd.DataFrame,
    table: pd.DataFrame,
    key: str,
) -> None:
    with st.expander("SAR stress test · can this substitution effect be trusted?", expanded=True):
        st.caption(
            "Challenge this observed effect, inspect where it changes, and identify a next "
            "measurement. This is not a probability of success or a mechanistic explanation."
        )
        thresholds = st.multiselect(
            "Effect thresholds to test (log units)",
            [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
            default=[0.25, 0.5, 0.75, 1.0],
            key=f"{key}_stress_thresholds",
        )
        omission_limit = st.select_slider(
            "Maximum omissions per test",
            options=[25, 50, 100, 200],
            value=100,
            key=f"{key}_stress_limit",
            help="One test per compound, context, assay or publication. "
            "Larger studies are bounded and any partial analysis is explicitly reported.",
        )
        chosen_thresholds = tuple(sorted(set([result.effect_threshold, *thresholds])))
        signature = hashlib.sha256(
            (
                result.pairs.to_csv(index=False)
                + result.series.members.to_csv(index=False)
                + evidence.to_csv(index=False)
                + claim.to_json()
                + str((chosen_thresholds, omission_limit, result.effect_threshold))
            ).encode()
        ).hexdigest()
        if st.button("Run SAR stress test", key=f"{key}_stress_run", type="primary"):
            with st.spinner("Challenging the observed effect and checking matched cycles…"):
                stress = stress_test_sar(
                    result,
                    claim,
                    evidence,
                    thresholds=chosen_thresholds,
                    max_omissions_per_axis=omission_limit,
                )
                st.session_state[f"{key}_stress_result"] = (signature, stress)
        saved = st.session_state.get(f"{key}_stress_result")
        if saved is None or saved[0] != signature:
            st.info(
                "Run the stress test for this transformation. No new data are downloaded, "
                "no model is trained, and no docking is performed."
            )
            return
        stress = saved[1]
        st.markdown(f"**{stress.verdict}**")
        if any(
            stress.settings[flag]
            for flag in (
                "omissions_truncated",
                "cycle_search_truncated",
                "pair_limit_reached",
            )
        ):
            st.warning(
                "This is a partial stress test: a computational limit was reached. "
                "Narrow the evidence subset or increase the omission limit."
            )
        stability, contexts, next_step = st.tabs(
            ["Does the effect persist?", "Where does it change?", "What to measure next"]
        )
        with stability:
            evaluated = stress.checks[
                (stress.checks.test != "Baseline") & (stress.checks.pairs > 0)
            ]
            cols = st.columns(3)
            cols[0].metric("Evaluable omissions", str(len(evaluated)))
            cols[1].metric(
                "Same directional conclusion", str(int(evaluated.retains_direction.sum()))
            )
            cols[2].metric("Omissions leaving no pairs", str(int((stress.checks.pairs == 0).sum())))
            plotted = stress.checks[stress.checks.pairs > 0].head(200).copy()
            if not plotted.empty:
                plotted["Subset"] = plotted.test + " · " + plotted.omitted.str.slice(0, 48)
                points = (
                    alt.Chart(plotted)
                    .mark_circle(size=65)
                    .encode(
                        x=alt.X(
                            "median_delta:Q",
                            title="Median ΔpActivity after omission",
                            scale=alt.Scale(zero=False),
                        ),
                        y=alt.Y("Subset:N", sort=None, title=None),
                        color=alt.condition(
                            alt.datum.retains_direction, alt.value("#10b981"), alt.value("#f59e0b")
                        ),
                        tooltip=[
                            "test:N",
                            "omitted:N",
                            "pairs:Q",
                            "disjoint_pairs:Q",
                            "median_delta:Q",
                            "interpretation:N",
                        ],
                    )
                )
                zero = (
                    alt.Chart(pd.DataFrame({"zero": [0]}))
                    .mark_rule(color="#94a3b8")
                    .encode(x="zero:Q")
                )
                st.altair_chart(
                    (points + zero).properties(height=min(650, max(160, len(plotted) * 20))),
                    width="stretch",
                )
                if len(stress.checks[stress.checks.pairs > 0]) > 200:
                    st.caption(
                        "Chart: first 200 evaluable rows. The table/export retain all tested rows."
                    )
            st.caption(
                "Compound omissions remove all comparisons involving that compound. Publication "
                "and assay omissions recompute measurement medians, without falling back to pooled "
                "values. Removing the only assay leaves no evidence, not a failed effect. "
                "These points are subset estimates, not confidence intervals."
            )
            st.markdown("**Does changing your effect threshold change the conclusion?**")
            st.dataframe(
                stress.thresholds.rename(
                    columns={
                        "threshold": "Minimum effect (log units)",
                        "pairs": "Compound pairs",
                        "disjoint_pairs": "Non-overlapping pairs",
                        "assays": "Assays",
                        "median_delta": "Median ΔpActivity",
                        "direction_agreement": "Sign agreement",
                        "interpretation": "Conclusion",
                    }
                ),
                hide_index=True,
                width="stretch",
            )
            with st.expander("All sensitivity checks"):
                st.dataframe(stress.checks, hide_index=True, width="stretch")
        with contexts:
            if stress.contexts.empty:
                st.info("No shared-assay contexts are available for this transformation.")
            else:
                chart = stress.contexts.copy()
                chart["Context"] = chart.assay_id + " · " + chart.context
                st.altair_chart(
                    alt.Chart(chart.head(100))
                    .mark_bar(color="#a78bfa")
                    .encode(
                        x=alt.X(
                            "median_delta:Q",
                            title="Median ΔpActivity in each assay / chemical context",
                        ),
                        y=alt.Y("Context:N", sort="-x", title=None),
                        tooltip=[
                            "assay_id:N",
                            "context:N",
                            "pairs:Q",
                            "median_delta:Q",
                            "interpretation:N",
                        ],
                    )
                    .properties(height=min(600, max(160, len(chart.head(100)) * 24))),
                    width="stretch",
                )
                st.dataframe(stress.contexts, hide_index=True, width="stretch")
            if not stress.cycles.empty:
                st.markdown("**Complete matched four-compound cycles**")
                st.caption(
                    "Interaction Δ = substitution effect in the second context minus its effect "
                    "in the first. All four compounds share an assay, endpoint and "
                    "variant annotation. "
                    "A nonzero value alone does not prove nonadditivity beyond assay noise "
                    "or establish a binding mechanism."
                )
                cycle_index = st.selectbox(
                    "Inspect substitution cycle",
                    stress.cycles.index.tolist(),
                    format_func=lambda i: (
                        f"{stress.cycles.loc[i, 'other_position']} · "
                        f"{stress.cycles.loc[i, 'assay_id']} · "
                        f"effect change {stress.cycles.loc[i, 'interaction_delta']:+.2f}"
                    ),
                    key=f"{key}_stress_cycle_{signature[:12]}",
                )
                cycle = stress.cycles.loc[cycle_index]
                st.markdown(
                    f"**{fragment_label(claim['fragment_from'])} → "
                    f"{fragment_label(claim['fragment_to'])} at {claim['position']}**"
                )
                for context_suffix, left_id, right_id, delta_name in (
                    ("from", "id_a", "id_b", "delta_first"),
                    ("to", "id_c", "id_d", "delta_second"),
                ):
                    st.markdown(
                        f"{cycle['other_position']} = "
                        f"**{fragment_label(cycle[f'context_{context_suffix}'])}** · "
                        f"substitution effect **{cycle[delta_name]:+.2f} pActivity**"
                    )
                    for col, id_name in zip(st.columns(2), (left_id, right_id), strict=True):
                        identifier = cycle[id_name]
                        block = table[table.molecule_id == identifier]
                        with col:
                            if not block.empty:
                                _structure(
                                    str(block.iloc[0].smiles), str(identifier), size=(300, 200)
                                )
                            else:
                                st.write(str(identifier))
                st.caption(
                    f"Difference between the two observed substitution effects: "
                    f"{cycle['interaction_delta']:+.2f} log units. Assay {cycle['assay_id']}; "
                    "no assay-noise significance or interaction mechanism is inferred."
                )
                st.dataframe(stress.cycles, hide_index=True, width="stretch")
            else:
                st.info(
                    "No unambiguous complete four-compound cycles were found in the bounded search."
                )
        with next_step:
            st.caption(
                "Evidence-gap suggestions, not predicted potency rankings. Check the original "
                "protocol, chemical stability and synthetic feasibility before planning a study."
            )
            if stress.suggestions.empty:
                st.info(
                    "No specific missing-cycle or conflicting-pair suggestion was identified. "
                    "This does not mean the evidence is complete."
                )
            for number, suggestion in enumerate(stress.suggestions.to_dict("records"), start=1):
                st.markdown(f"**{number}. {suggestion['action']}** — {suggestion['reason']}")
                st.caption(
                    f"Assay {suggestion['assay_id']} · {suggestion['endpoint']} · "
                    f"variant {suggestion['variant']} · "
                    f"supporting compounds {suggestion['supporting_ids']}"
                )
                if suggestion["existing_ids"]:
                    st.write("Existing compounds: " + suggestion["existing_ids"])
                    for identifier in suggestion["existing_ids"].split(";"):
                        if identifier.startswith("CHEMBL") and identifier[6:].isdigit():
                            st.markdown(
                                f"[{identifier} on ChEMBL]"
                                f"(https://www.ebi.ac.uk/chembl/explore/compound/{identifier})"
                            )
                elif suggestion["proposed_smiles"]:
                    _structure(
                        suggestion["proposed_smiles"],
                        "Proposed, unmeasured analogue · no potency predicted",
                    )
                    st.code(suggestion["proposed_smiles"], language=None)
                    st.caption(
                        "Source stereochemical annotations are retained; "
                        "unassigned stereochemistry remains unresolved."
                    )
                if suggestion["proposed_pattern"]:
                    st.write(
                        {
                            p: fragment_label(f)
                            for p, f in json.loads(suggestion["proposed_pattern"]).items()
                        }
                    )
            if not stress.suggestions.empty:
                with st.expander("Suggestion provenance and record identifiers"):
                    st.dataframe(stress.suggestions, hide_index=True, width="stretch")
        for limitation in stress.limitations:
            st.caption(limitation)
        st.download_button(
            "Download SAR stress-test evidence card",
            sar_stress_archive(stress, result, evidence),
            file_name="sar_stress_test.zip",
            mime="application/zip",
            key=f"{key}_stress_download",
        )


def show_matched_pair_explorer(
    pairs: pd.DataFrame, table: pd.DataFrame, evidence: pd.DataFrame
) -> None:
    """Inspect core/linker changes and cliffs outside the R-group decomposition."""
    st.markdown("**Matched transformations · compound explorer**")
    st.caption(
        "Single-cut molecular pairs can compare changes across different scaffold series, "
        "including eligible linker changes. These deltas use curated molecule potencies; "
        "inspect assay measurements before interpreting the change."
    )
    query = st.text_input("Find matched pairs by compound ID", key="sar_mmp_query").strip()
    cutoff = st.slider(
        "Minimum matched-pair |ΔpActivity|", 0.0, 4.0, 0.0, 0.1, key="sar_mmp_cutoff"
    )
    subset = pairs[pairs["delta_pactivity_b_minus_a"].abs() >= cutoff]
    if query:
        subset = subset[
            subset["id_a"].str.contains(query, case=False, regex=False)
            | subset["id_b"].str.contains(query, case=False, regex=False)
        ]
    if subset.empty:
        st.info("No matched pairs meet these choices.")
        return
    scope = hashlib.sha256((query + str(cutoff) + str(len(subset))).encode()).hexdigest()[:12]
    pages = (len(subset) - 1) // 20 + 1
    page = int(st.number_input("Matched-pair page", 1, pages, 1, key=f"sar_mmp_page_{scope}"))
    current = subset.iloc[(page - 1) * 20 : page * 20]
    st.caption(f"{len(subset):,} pairs · page {page} of {pages}")
    selected = st.selectbox(
        "Matched transformation pair",
        current.index.tolist(),
        format_func=lambda i: (
            f"{subset.loc[i, 'id_a']} → {subset.loc[i, 'id_b']} · "
            f"Δ {subset.loc[i, 'delta_pactivity_b_minus_a']:+.2f}"
        ),
        key=f"sar_mmp_pair_{scope}_{page}",
    )
    row = subset.loc[selected]
    _structure(str(row["context"]), "Shared single-cut context")
    pair = pd.Series(
        {
            "id_from": row["id_a"],
            "id_to": row["id_b"],
            "pactivity_from": row["pactivity_a"],
            "pactivity_to": row["pactivity_b"],
            "delta_pactivity": row["delta_pactivity_b_minus_a"],
            "evidence_tier": "curated molecule medians",
            "assay_id": "",
            "endpoint": "",
            "variant": "",
        }
    )
    _pair_view(pair, table, "mmp")
    cols = st.columns(2)
    with cols[0]:
        _structure(str(row["fragment_a"]), "From substituent")
    with cols[1]:
        _structure(str(row["fragment_b"]), "To substituent")
    if not evidence.empty:
        block = evidence[evidence["molecule_id"].isin([row["id_a"], row["id_b"]])]
        with st.expander("Matched-pair assay records"):
            st.dataframe(block, hide_index=True, width="stretch")

"""Explicit, bounded analogue design for curated assays and user-owned series."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Literal, cast

import pandas as pd
import streamlit as st

from sarscope.analogue_design import (
    AnalogueDesign,
    DesignModel,
    design_archive,
    fit_design_model,
    generate_analogues,
    scope_signature,
    score_analogues,
)
from sarscope.analysis.descriptors import CORE_DESCRIPTORS
from sarscope.analysis.rgroups import ScaffoldSar, decompose_scaffold
from sarscope.analysis.sar_map import fragment_label
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_qsar import assay_contexts, dataset_from_assay
from sarscope.params import AnalogueDesignParams, ModelParams
from sarscope.sar_ui import _structure


def _signature(*items: Any) -> str:
    return hashlib.sha256(json.dumps(items, sort_keys=True, default=str).encode()).hexdigest()


def _show_result(
    result: AnalogueDesign,
    model: DesignModel | None,
    key: str,
    *,
    archive: bytes | None = None,
) -> None:
    frame = result.candidates
    st.subheader("Proposed experiments, not confirmed improvements")
    st.caption(
        "Observed ΔpActivity describes supporting measured pairs in other substitution "
        "contexts—not the potency of a new candidate. Positive means higher potency. "
        "Quartiles describe observed effects; they are not prediction intervals."
    )
    if result.provenance["truncated"] or result.provenance["pair_limit_reached"]:
        st.warning(
            "A configured proposal/pair limit was reached; this is not an exhaustive search."
        )
    if frame.empty:
        st.info(
            "No supported one-position swaps were reconstructable for this parent. "
            "Try another parent/position or add measured matched pairs; no chemistry was invented."
        )
    else:
        unmeasured = int(frame.known_ids.eq("").sum())
        st.caption(
            f"{unmeasured} unmeasured combinations · "
            f"{len(frame) - unmeasured} known assay controls."
        )
        if unmeasured == 0:
            st.info(
                "All reconstructable swaps here are already measured in this assay. These are "
                "reference controls, not new designs. A series with additional combinable "
                "R-positions is needed to propose unmeasured combinations."
            )
        overview = [
            "molecule_id",
            "position",
            "known_ids",
            "supporting_pairs",
            "disjoint_pairs",
            "observed_median_delta",
            "contradictory_effects",
            "delta_MW",
            "delta_logP",
            "nearest_measured_similarity",
            "near_activity_cliff",
            "unresolved_stereo",
            "predicted_pactivity",
            "predicted_delta_vs_parent",
            "prediction_status",
        ]
        st.dataframe(
            frame[[name for name in overview if name in frame]],
            column_config={
                "molecule_id": "Proposed analogue",
                "position": "Changed position",
                "known_ids": "Known in selected assay",
                "supporting_pairs": "Measured pairs",
                "disjoint_pairs": "Disjoint measured pairs",
                "observed_median_delta": st.column_config.NumberColumn(
                    "Supporting-pair ΔpActivity",
                    help="Median effect in measured pairs—not the candidate's activity.",
                    format="%.2f",
                ),
                "predicted_pactivity": "Model pActivity",
                "predicted_delta_vs_parent": "Model Δ vs model parent",
                "contradictory_effects": "Conflicting measured effects",
                "delta_MW": "ΔMW (Da)",
                "delta_logP": "ΔlogP",
                "near_activity_cliff": "Near measured cliff",
                "unresolved_stereo": "Stereo review needed",
                "nearest_measured_similarity": "Nearest measured Tanimoto",
                "prediction_status": "Prediction status",
            },
            hide_index=True,
            width="stretch",
        )
        st.vega_lite_chart(
            frame,
            {
                "mark": {"type": "point", "filled": True, "size": 90},
                "encoding": {
                    "x": {
                        "field": "delta_MW",
                        "type": "quantitative",
                        "title": "Molecular-weight change vs parent (Da)",
                    },
                    "y": {
                        "field": "delta_logP",
                        "type": "quantitative",
                        "title": "Calculated logP change vs parent",
                    },
                    "color": {
                        "field": "near_activity_cliff",
                        "type": "nominal",
                        "title": "Near measured cliff",
                        "scale": {"domain": [False, True], "range": ["#299d8f", "#dc6b43"]},
                    },
                    "tooltip": [
                        {"field": "molecule_id", "type": "nominal"},
                        {"field": "position", "type": "nominal"},
                        {"field": "supporting_pairs", "type": "quantitative"},
                        {"field": "known_ids", "type": "nominal"},
                    ],
                },
            },
            width="stretch",
        )
        identifier = st.selectbox(
            "Inspect proposed analogue", frame.molecule_id.tolist(), key=f"{key}_inspect"
        )
        candidate = frame[frame.molecule_id.eq(identifier)].iloc[0]
        st.write(
            f"At {candidate.position}: {fragment_label(candidate.fragment_from)} → "
            f"{fragment_label(candidate.fragment_to)}. All other R-positions remain locked."
        )
        columns = st.columns(2)
        with columns[0]:
            _structure(
                result.parent.iloc[0].smiles, f"Parent · {result.parent.iloc[0].molecule_id}"
            )
        with columns[1]:
            _structure(candidate.smiles, f"Proposed · {identifier} · {candidate.position}")
        st.write(
            f"{candidate.prediction_status}. "
            f"Known in selected assay: {candidate.known_ids or 'no'}."
        )
        st.caption(
            f"Nearest measured analogue: {candidate.nearest_measured_id} · "
            f"ECFP4 Tanimoto {candidate.nearest_measured_similarity:.2f} · "
            f"measured pActivity {candidate.nearest_measured_pactivity:.2f}. "
            "This measured value is not assigned to the candidate."
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "property": CORE_DESCRIPTORS,
                    "parent": [result.parent.iloc[0][name] for name in CORE_DESCRIPTORS],
                    "candidate": [candidate[name] for name in CORE_DESCRIPTORS],
                    "change": [candidate[f"delta_{name}"] for name in CORE_DESCRIPTORS],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.markdown("**Measured supporting pairs · other R-positions held constant**")
        st.dataframe(
            result.evidence_pairs[result.evidence_pairs.candidate_id.eq(identifier)],
            hide_index=True,
            width="stretch",
        )
        if candidate.contradictory_effects:
            st.warning(
                "This swap has both positive and negative measured effects: context matters."
            )
        if candidate.near_activity_cliff:
            st.warning(
                "Close to a measured activity-cliff compound. Similarity alone can miss a large "
                "potency change; this flag is a heuristic, not a failure probability."
            )
        if candidate.unresolved_stereo:
            st.warning(
                "Unresolved stereochemistry: potency scoring is withheld for this candidate."
            )
        st.caption(
            f"Structural alerts (PAINS/Brenk): {candidate.structural_alerts or 'none detected'}. "
            "Alerts do not establish toxicity; no alert does not establish safety. "
            "Synthesis feasibility, selectivity and ADME are not established."
        )
        with st.expander("All candidate properties and prediction bands"):
            st.dataframe(frame, hide_index=True, width="stretch")
    with st.expander("Attempt accounting and provenance"):
        st.dataframe(result.accounting, hide_index=True, width="stretch")
        st.json(result.provenance)
    st.download_button(
        "Download proposed analogues CSV",
        frame.to_csv(index=False),
        file_name="proposed_analogues.csv",
        mime="text/csv",
        key=f"{key}_csv",
    )
    st.download_button(
        "Download evidence package · CSV + SDF + provenance",
        archive if archive is not None else design_archive(result, model),
        file_name="analogue_design.zip",
        mime="application/zip",
        key=f"{key}_download",
    )


def show_analogue_design(
    series_list: list[ScaffoldSar],
    table: pd.DataFrame,
    evidence: pd.DataFrame,
    source: dict[str, Any] | None = None,
    *,
    canonical_tautomer: bool = False,
    key: str = "analogue_design",
) -> None:
    st.subheader("Design next analogues")
    st.write(
        "Choose a parent and editable positions. Recombine small substituents observed at "
        "those positions in measured assay-matched pairs, changing only one position per "
        "candidate. The mapped core and all other substituents stay fixed."
    )
    st.caption(
        "Experimental decision support—not a validated lead optimiser. Generation needs no ML; "
        "optional potency scoring uses a separate assay-scoped 2D model, never the 3D model."
    )
    contexts = assay_contexts(evidence)
    if contexts.empty or not series_list:
        st.info("Prepare SAR series with exact assay-linked potency measurements first.")
        return
    choice = st.selectbox(
        "Design assay / endpoint / variant",
        list(range(len(contexts))),
        format_func=lambda i: (
            f"{contexts.iloc[i].assay_chembl_id} · {contexts.iloc[i].standard_type} · "
            f"{contexts.iloc[i].variant_scope} · {contexts.iloc[i].compounds} compounds"
        ),
        key=f"{key}_context",
    )
    context = contexts.iloc[choice]
    if context.compounds > 500:
        st.info(
            "Choose an assay with at most 500 compounds; this workflow does not silently sample."
        )
        return
    dataset = dataset_from_assay(
        table,
        evidence,
        context.assay_chembl_id,
        context.standard_type,
        context.variant_scope,
        source,
    )
    if (source or {}).get("user_declared_context"):
        dataset.source["kind"] = "user_declared_assay"
        st.caption("Uploaded assay comparability is user-declared, not verified from ChEMBL.")
    st.caption(
        "Activities are reaggregated from this exact assay/endpoint/variant only. "
        "Unreported variant does not mean confirmed wild type."
    )
    available = [
        series
        for series in series_list
        if len(series.members[series.members.molecule_id.isin(dataset.table.molecule_id)]) >= 2
        and series.positions
    ]
    if not available:
        st.info("No prepared series has two members and editable positions in this assay.")
        return
    series_index = st.selectbox(
        "Prepared series",
        list(range(len(available))),
        format_func=lambda i: available[i].scaffold,
        key=f"{key}_series",
    )
    series = available[series_index]
    members = series.members[series.members.molecule_id.isin(dataset.table.molecule_id)]
    parents = members.molecule_id.tolist()
    recommended = (source or {}).get("recommended", {}).get("parent_id")
    parent_id = st.selectbox(
        "Parent compound",
        parents,
        index=parents.index(recommended) if recommended in parents else 0,
        key=f"{key}_parent",
    )
    parent = members[members.molecule_id.eq(parent_id)].iloc[0]
    columns = st.columns(2)
    with columns[0]:
        _structure(parent.molecule_smiles, f"Parent · {parent_id}")
    with columns[1]:
        _structure(parent.core_smiles, "Fixed mapped core · attachment labels identify R-positions")
    st.dataframe(
        pd.DataFrame({"position": series.positions, "parent fragment": parent[series.positions]}),
        hide_index=True,
        width="stretch",
    )
    positions = st.multiselect(
        "Editable positions", series.positions, default=series.positions, key=f"{key}_positions"
    )
    st.caption(
        "Unselected positions are locked. R labels are decomposition labels, not IUPAC numbering."
    )
    with st.expander("Proposal limits and cliff-neighbour flags"):
        maximum = st.slider("Maximum proposals", 1, 100, 50, key=f"{key}_maximum")
        fragment_size = st.slider("Maximum fragment heavy atoms", 1, 20, 10, key=f"{key}_fragment")
        cliff_similarity = st.slider(
            "Cliff-neighbour ECFP4 Tanimoto", 0.5, 1.0, 0.85, 0.05, key=f"{key}_cliff_sim"
        )
        cliff_activity = st.number_input(
            "Measured cliff ΔpActivity", min_value=0.1, value=1.0, key=f"{key}_cliff_delta"
        )
    params = AnalogueDesignParams(
        max_candidates=maximum,
        max_fragment_heavy_atoms=fragment_size,
        cliff_similarity=cliff_similarity,
        cliff_activity=cliff_activity,
    )
    scope = scope_signature(dataset.table, dataset.measurements)
    signature = _signature(
        scope, series.members.to_csv(index=False), parent_id, positions, asdict(params)
    )
    result: AnalogueDesign | None = None
    if st.button("Generate analogue proposals", disabled=not positions, key=f"{key}_generate"):
        try:
            with st.spinner("Reconstructing bounded, one-position substitutions…"):
                result = generate_analogues(
                    series,
                    dataset.table,
                    dataset.measurements,
                    parent_id,
                    positions,
                    params,
                    source=dataset.source,
                )
            st.session_state[f"{key}_result"] = (signature, result)
            st.session_state.pop(f"{key}_scored", None)
        except Exception as exc:
            st.session_state.pop(f"{key}_result", None)
            st.error(f"Proposal generation needs review: {exc}")
    stored = st.session_state.get(f"{key}_result")
    result = stored[1] if stored is not None and stored[0] == signature else None
    if stored is not None and result is None:
        st.info("Settings or input changed. Generate proposals again; old results are hidden.")

    model: DesignModel | None = None
    with st.expander("Optional · fit an assay-scoped 2D potency model"):
        st.caption(
            "ECFP4 regression on the selected assay (30–500 unique structures). The held-out "
            "test must contain ≥10 compounds, R² ≥0.5, training-CV q² ≥0.3 and RMSE below the "
            "training-mean baseline; a finite positive CV residual band is also required. "
            "These are conservative engineering gates, not external scientific validation. "
            "Repeated tuning against this test set invalidates its independence."
        )
        algorithm = st.selectbox(
            "Design estimator", ["random_forest", "extra_trees", "ridge"], key=f"{key}_algorithm"
        )
        split = st.selectbox("Design split", ["random", "scaffold"], key=f"{key}_split")
        fraction = st.slider(
            "Design held-out fraction", 0.2, 0.4, 0.25, 0.05, key=f"{key}_fraction"
        )
        folds = st.selectbox("Design CV folds", [3, 5], key=f"{key}_folds")
        settings = ModelParams(
            regression_algorithms=(algorithm,),
            split=cast(Literal["random", "scaffold"], split),
            test_fraction=fraction,
            cv_folds=folds,
            oversample=False,
            leakage_audit=False,
        )
        model_signature = _signature(scope, asdict(settings), canonical_tautomer, asdict(params))
        if st.button("Fit scoped 2D model", disabled=len(dataset.table) < 30, key=f"{key}_fit"):
            try:
                with st.spinner(
                    "Evaluating train/CV/test partitions, then refitting for deployment…"
                ):
                    fitted = fit_design_model(
                        dataset.table,
                        dataset.measurements,
                        params,
                        settings,
                        canonical_tautomer=canonical_tautomer,
                    )
                st.session_state[f"{key}_model"] = (model_signature, fitted)
                st.session_state.pop(f"{key}_scored", None)
            except Exception as exc:
                st.session_state.pop(f"{key}_model", None)
                st.error(f"Model could not be fitted: {exc}")
        saved_model = st.session_state.get(f"{key}_model")
        if saved_model is not None and saved_model[0] == model_signature:
            model = saved_model[1]
            st.dataframe(model.checks, hide_index=True, width="stretch")
            st.dataframe(model.regression.scores, hide_index=True, width="stretch")
            if not model.checks.passed.all():
                st.warning(
                    "Model checks failed. Evidence-only proposals remain available; "
                    "no potency scoring."
                )
        if st.button(
            "Score generated analogues",
            key=f"{key}_score",
            disabled=result is None or model is None or not model.checks.passed.all(),
        ):
            assert result is not None and model is not None
            try:
                st.session_state[f"{key}_scored"] = (
                    (signature, model_signature),
                    score_analogues(result, model),
                )
            except Exception as exc:
                st.session_state.pop(f"{key}_scored", None)
                st.error(f"Scoring needs review: {exc}")
    if result is not None:
        scored = st.session_state.get(f"{key}_scored")
        if model is not None and scored is not None and scored[0] == (signature, model_signature):
            result = scored[1]
        _show_result(result, model, key)


def show_design_upload(*, key: str = "upload_analogue_design") -> None:
    from sarscope.field_qsar_ui import _upload_dataset

    st.title("SARscope · 2D analogue design")
    st.info("For field-model design and the measured 3D demo, choose 3D QSAR · own compounds.")
    dataset = _upload_dataset(key, needs_alignment=False)
    if dataset is None:
        return
    table = add_scaffolds(dataset.table)
    evidence = dataset.measurements.assign(
        assay_chembl_id="USER_DECLARED_UPLOAD",
        standard_type=dataset.source["endpoint"],
        standard_relation="=",
        assay_variant_mutation=None,
    )
    source = {**dataset.source, "user_declared_context": True}
    options = [s for s, n in table.murcko.value_counts().items() if s and n >= 2]
    if not options:
        st.info("At least two compounds must share a ring scaffold for position-based proposals.")
        return
    recommended = source.get("recommended", {}).get("scaffold")
    scaffold = st.selectbox(
        "Shared scaffold to prepare",
        options,
        index=options.index(recommended) if recommended in options else 0,
        key=f"{key}_scaffold",
    )
    signature = _signature(scope_signature(table, evidence), scaffold)
    if st.button("Prepare design series", key=f"{key}_prepare"):
        with st.spinner("Decomposing the selected scaffold into R-positions…"):
            series = decompose_scaffold(table, scaffold, min_substituent_count=1)
        st.session_state[f"{key}_prepared"] = (signature, series)
    prepared = st.session_state.get(f"{key}_prepared")
    if prepared is None or prepared[0] != signature or prepared[1] is None:
        st.info("Prepare the selected series to choose a parent and editable positions.")
        return
    show_analogue_design(
        [prepared[1]],
        table,
        evidence,
        source,
        canonical_tautomer=False,
        key=f"{key}_design",
    )

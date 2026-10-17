"""The design-and-predict step belongs to the active, unchanged 3D result."""

from __future__ import annotations

import hashlib
import json
from typing import Literal, cast

import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem

from sarscope.analogue_design_ui import _show_result
from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_design import (
    field_design_archive,
    field_design_inputs,
    field_model_signature,
    generate_field_analogues,
    predict_field_analogues,
)
from sarscope.field_interpretation import interpretation_review, interpretation_signature
from sarscope.field_qsar import FieldQsarResult


def show_field_design(result: FieldQsarResult, key: str) -> None:
    st.markdown("**4 · Design experiments · exploratory 3D predictions**")
    if getattr(result.domain, "fitted_pca", None) is None:
        st.info("Rerun this 3D model after the update to retain its original domain transform.")
        return
    st.caption(
        "Choose a parent and one-position substitutions supported by this measured series. "
        "Candidates use the same fitted 3D field PLS, training grid, charge method and field "
        "mask/scaling—not the 2D baseline. Nothing is refitted. Contours and per-field "
        "changes explain model associations, not protein interactions or proven improvements."
    )
    review = interpretation_review(result)
    st.warning(review["interpretation_message"])
    st.caption(
        "The same-split 2D baseline remains above for comparison. Small-pose training "
        "sensitivity is not a candidate-specific confidence interval, and no proposed "
        "potency gain is a confirmed improvement."
    )
    comparable_series_confirmed = False
    if result.prepared.dataset.source.get("design_comparability_review_required"):
        st.warning(
            "This historical CCR5 benchmark has not been independently adjudicated to a "
            "single comparable assay. The model tutorial does not establish that its "
            "matched-pair effects can be transferred between compounds. Review the source "
            "experiments before using them as design evidence."
        )
        comparable_series_confirmed = st.checkbox(
            "I reviewed the source experiments and confirm comparability for matched-pair design",
            key=f"{key}_field_design_comparability",
        )
        if not comparable_series_confirmed:
            st.info(
                "The 3D model and contours remain available above. Analogue-design evidence "
                "requires this separate comparability declaration."
            )
            return
    signature = field_model_signature(result)
    table, _ = field_design_inputs(result)
    table = add_scaffolds(table)
    scaffolds = [sc for sc, count in table.murcko.value_counts().items() if sc and count >= 3]
    if not scaffolds:
        st.info("No scaffold has three measured members for supported R-group proposals.")
        return
    scope_key = f"{key}_field_design"
    scaffold = st.selectbox("3D design scaffold", scaffolds, key=f"{scope_key}_scaffold")
    preparation_key = hashlib.sha256((signature + scaffold).encode()).hexdigest()
    if st.button("Prepare R-positions for 3D design", key=f"{scope_key}_prepare"):
        st.session_state.pop(f"{scope_key}_design", None)
        st.session_state.pop(f"{scope_key}_predictions", None)
        with st.spinner("Decomposing the fitted 3D series; model and alignment remain unchanged…"):
            series = decompose_scaffold(table, scaffold, min_substituent_count=1)
        st.session_state[f"{scope_key}_series"] = (preparation_key, series)
    prepared = st.session_state.get(f"{scope_key}_series")
    if prepared is None or prepared[0] != preparation_key or prepared[1] is None:
        st.info("Prepare the R-positions to select a parent. No new model is trained here.")
        return
    series = prepared[1]
    if not series.positions:
        st.info("No varying R-position is available in this fitted series.")
        return
    members = series.members.molecule_id.tolist()
    train_ids = set(table.iloc[result.regression.train_index].molecule_id)
    parents = [pid for pid in members if pid in train_ids] + [
        pid for pid in members if pid not in train_ids
    ]
    parent = st.selectbox("3D design parent", parents, key=f"{scope_key}_parent")
    parent_row = series.members[series.members.molecule_id.eq(parent)].iloc[0]
    from sarscope.sar_ui import _structure

    _structure(parent_row.core_smiles, "Fixed mapped core · R labels identify editable positions")
    positions = st.multiselect(
        "3D editable positions",
        series.positions,
        default=series.positions,
        key=f"{scope_key}_positions",
    )
    maximum = st.slider("3D proposal batch size", 1, 10, 5, key=f"{scope_key}_maximum")
    design_key = json.dumps(
        [signature, scaffold, series.members.to_csv(index=False), parent, positions, maximum]
    )
    if st.button(
        "Generate proposals for this 3D model", disabled=not positions, key=f"{scope_key}_generate"
    ):
        try:
            design = generate_field_analogues(
                result,
                series,
                parent,
                positions,
                max_candidates=maximum,
                comparable_series_confirmed=comparable_series_confirmed,
            )
            st.session_state[f"{scope_key}_design"] = (design_key, design)
            st.session_state.pop(f"{scope_key}_predictions", None)
        except Exception as exc:
            st.session_state.pop(f"{scope_key}_design", None)
            st.error(f"3D design needs review: {exc}")
    saved = st.session_state.get(f"{scope_key}_design")
    if saved is None or saved[0] != design_key:
        st.info("Generate proposals for the current model, parent and editable positions.")
        return
    design = saved[1]
    current_route = result.prepared.params.alignment
    choices = ["constrained", "rigid_mcs"] if current_route == "prealigned" else [current_route]
    route = st.selectbox(
        "Candidate alignment to the fitted reference", choices, key=f"{scope_key}_route"
    )
    if current_route == "prealigned":
        st.caption(
            "The model retains its supplied alignment. New candidates need a generated "
            "conformation aligned to its fixed reference; this adds an explicit pose assumption."
        )
    consent = False
    if review["requires_exploratory_consent"]:
        st.warning(
            "This 3D model has unmet validation checks or unqualified sensitivity evidence. "
            "Default potency scoring is withheld. "
            "You may explicitly inspect exploratory predictions; geometry/domain failures "
            "remain withheld and no candidate is certified as improved."
        )
        consent = st.checkbox(
            "I understand: show exploratory predictions from this unvalidated 3D model",
            key=f"{scope_key}_exploratory",
        )
    prediction_key = json.dumps([design_key, route, consent, interpretation_signature(result)])
    if st.button(
        "Align and predict with this 3D model",
        key=f"{scope_key}_predict",
        disabled=design.candidates.empty
        or (review["requires_exploratory_consent"] and not consent),
    ):
        try:
            with st.spinner(
                "Preparing up to 10 stereo-checked poses and evaluating the fixed 3D fields…"
            ):
                prediction = predict_field_analogues(
                    result,
                    design,
                    route=cast(Literal["rigid_mcs", "constrained"], route),
                    allow_exploratory=consent,
                )
            st.session_state[f"{scope_key}_predictions"] = (prediction_key, prediction)
        except Exception as exc:
            st.session_state.pop(f"{scope_key}_predictions", None)
            st.error(f"3D prediction needs review: {exc}")
    predicted = st.session_state.get(f"{scope_key}_predictions")
    if predicted is None or predicted[0] != prediction_key:
        st.info(
            f"{len(design.candidates)} proposals prepared. "
            "Run alignment and prediction above to open the proposed-experiments results. "
            "No potency predictions have been made for these settings."
        )
        return
    prediction = predicted[1]
    st.caption("Prediction engine: the displayed 3D field PLS. No RF fit, scoring or fallback.")
    st.caption(
        "Applicability uses the fitting compounds' two-component field-PCA box, not ECFP "
        "similarity. This is a weak domain heuristic, not a reliability guarantee. "
        "Reported empirical bands describe activity levels, not uncertainty in potency gains."
    )
    if prediction.aligned:
        from sarscope.field_qsar_ui import viewer_html

        reference = result.prepared.alignment.molecules[result.prepared.alignment.reference_index]
        components.html(
            viewer_html(
                [
                    Chem.MolToMolBlock(reference),
                    *[Chem.MolToMolBlock(mol) for mol in prediction.aligned[:5]],
                ]
            ),
            height=640,
        )
    if not prediction.contributions.empty:
        st.markdown("**Why does this model predict a change?**")
        st.caption(
            "Each field's signed contribution to candidate minus model-parent pActivity. "
            "These sum to the predicted change; they are not binding energies "
            "or experimental effects."
        )
        st.dataframe(prediction.contributions, hide_index=True, width="stretch")
        chart = prediction.contributions.pivot(
            index="molecule_id", columns="field", values="delta_pactivity_contribution"
        ).fillna(0)
        st.bar_chart(chart, stack=False, width="stretch")
    _show_result(
        prediction.design,
        None,
        scope_key,
        archive=field_design_archive(prediction, result),
    )

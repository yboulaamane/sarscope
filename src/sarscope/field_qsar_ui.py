"""User-triggered assay/upload → alignment review → experimental 3D QSAR."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import altair as alt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem

from sarscope.analysis.align3d import preparation_available
from sarscope.analysis.contours import COLORS, open_dx
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_qsar import (
    BENCHMARK_STATUS,
    FieldDataset,
    FieldQsarResult,
    assay_contexts,
    dataset_from_assay,
    dataset_from_upload,
    subset_dataset,
    upload_table,
)
from sarscope.field_qsar_report import aligned_sdf, field_qsar_archive, unmodelled_field_warning
from sarscope.params import FieldQsarParams, ModelParams
from sarscope.pipeline import prepare_field_qsar_stage, run_field_qsar_stage


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _discard_active(key: str) -> None:
    if st.session_state.pop(f"{key}_active_result", None) is not None:
        st.session_state.pop("report_zip", None)


def viewer_html(blocks: list[str], surfaces: list[dict[str, Any]] | None = None) -> str:
    """JSON-escape uploaded text; the browser fetches 3Dmol.js, not compound URLs."""
    payload = json.dumps({"blocks": blocks, "surfaces": surfaces or []}).replace("<", "\\u003c")
    return """<!doctype html><html><body style="margin:0;background:#19222a">
<div id="view" style="height:450px;width:100%;position:relative"></div>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script><script>
const data = PAYLOAD;
if (typeof $3Dmol === 'undefined') {
const box=document.getElementById('view');
box.textContent='Viewer unavailable. Download SDF/DX for local inspection.';
} else {
const viewer=$3Dmol.createViewer('view',{backgroundColor:'#19222a'});
data.blocks.forEach((block,i)=>{
viewer.addModel(block,'mol');
viewer.setStyle({model:i},{stick:{radius:0.13,opacity:i===0?1:0.45}});
});
data.surfaces.forEach(s=>viewer.addVolumetricData(s.dx,'dx',
{isoval:s.level,color:s.color,opacity:0.55}));
viewer.zoomTo();viewer.render();
}
</script></body></html>""".replace("PAYLOAD", payload)


@st.cache_data(show_spinner=False, max_entries=4)
def _read_upload(data: bytes, suffix: str) -> tuple[pd.DataFrame, list[Any]]:
    return upload_table(data, suffix)


def _upload_dataset(key: str, *, needs_alignment: bool = True) -> FieldDataset | None:
    st.caption(
        (
            "Upload one comparable experimental endpoint. SDF may contain pre-aligned 3D "
            "coordinates and activity properties; CSV/TSV requires SMILES and subsequent "
            "alignment. No docking or target structure is required."
        )
        if needs_alignment
        else (
            "Upload measured activities for one comparable target, assay, endpoint and variant. "
            "CSV/TSV needs SMILES; SDF needs activity properties. No 3D alignment, docking or "
            "target structure is required. Assay comparability is your declaration."
        )
    )
    file = st.file_uploader(
        "Compounds and measured activities", type=["sdf", "csv", "tsv"], key=f"{key}_file"
    )
    if file is None:
        st.code("molecule_id,smiles,pactivity\ncompound_1,CCOc1ccccc1,6.8", language="csv")
        return None
    data = file.getvalue()
    try:
        raw, molecules = _read_upload(data, Path(file.name).suffix)
        if raw.empty:
            raise ValueError("The uploaded file contains no compounds.")
        columns = raw.columns.tolist()
        controls = st.columns(3)
        id_col = controls[0].selectbox(
            "Compound identifier",
            columns,
            index=columns.index("molecule_id") if "molecule_id" in columns else 0,
            key=f"{key}_id",
        )
        smiles_col = controls[1].selectbox(
            "SMILES column",
            columns,
            index=columns.index("smiles") if "smiles" in columns else 0,
            key=f"{key}_smiles",
        )
        candidates = [c for c in columns if c not in (id_col, smiles_col)]
        if not candidates:
            st.info("Add a numeric measured activity property/column to your file.")
            return None
        value_col = controls[2].selectbox(
            "Measured activity",
            candidates,
            index=candidates.index("pactivity") if "pactivity" in candidates else 0,
            key=f"{key}_value",
        )
        controls = st.columns(2)
        endpoint = controls[0].selectbox(
            "Measurement endpoint", ["IC50", "Ki", "Kd", "EC50"], key=f"{key}_endpoint"
        )
        unit = controls[1].selectbox(
            "Input activity units",
            ["p", "nM", "uM", "M"],
            format_func=lambda v: "pActivity (−log10 M)" if v == "p" else v,
            key=f"{key}_unit",
        )
        if not st.checkbox(
            "These are exact, comparable measurements for one target and endpoint",
            key=f"{key}_comparable",
        ):
            st.info(
                "Confirm measurement comparability before preparing the series. "
                "Active/inactive calls are not potencies."
            )
            return None
        return dataset_from_upload(
            raw,
            molecules,
            id_col=id_col,
            smiles_col=smiles_col,
            value_col=value_col,
            unit=unit,
            endpoint=endpoint,
            source_sha256=hashlib.sha256(data).hexdigest(),
        )
    except Exception as exc:
        st.error(f"Input needs review: {exc}")
        return None


def _assay_dataset(
    table: pd.DataFrame, evidence: pd.DataFrame, source: dict[str, Any], key: str
) -> FieldDataset | None:
    contexts = assay_contexts(evidence)
    if contexts.empty:
        st.info(
            "No exact assay-linked evidence is available. "
            "Curate a ChEMBL target first, or upload data."
        )
        return None
    labels = [
        f"{row.assay_chembl_id} · {row.standard_type} · {row.variant_scope} "
        f"· {row.compounds} compounds"
        for row in contexts.itertuples()
    ]
    choice = st.selectbox(
        "One assay and endpoint",
        range(len(contexts)),
        format_func=lambda i: labels[i],
        key=f"{key}_assay",
    )
    row = contexts.iloc[choice]
    st.caption(
        "Potencies below are medians from this assay only—not the target-wide medians. "
        "Unreported variant annotation does not mean confirmed wild type."
    )
    return dataset_from_assay(
        table, evidence, row.assay_chembl_id, row.standard_type, row.variant_scope, source
    )


def show_field_qsar(
    table: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    source: dict[str, Any] | None = None,
    *,
    key: str = "field_qsar",
) -> None:
    st.subheader("3D QSAR · a focused chemical series")
    st.markdown(
        "Compare aligned molecular fields with measured potency, then inspect where the "
        "model associates steric, charge or hydrogen-bond features with activity. "
        "Choose the dataset, review its geometry, and run modelling separately."
    )
    st.warning(BENCHMARK_STATUS + " Contours are hypotheses, not observed protein contacts.")
    mode = "Upload my compounds"
    if table is not None and evidence is not None:
        mode = st.radio(
            "3D-QSAR data source",
            ["Selected ChEMBL assay", "Upload my compounds"],
            horizontal=True,
            key=f"{key}_source",
        )
    dataset = (
        _assay_dataset(table, evidence, source or {}, key)
        if mode == "Selected ChEMBL assay"
        else _upload_dataset(key)
    )
    if dataset is None:
        _discard_active(key)
        return
    scaffold_table = add_scaffolds(dataset.table)
    counts = scaffold_table["murcko"].fillna("Acyclic").value_counts()
    choices = ["All compounds", *counts.index.tolist()]
    scaffold = st.selectbox(
        "Focus the series by scaffold",
        choices,
        index=1 if len(choices) > 1 else 0,
        format_func=lambda value: (
            f"{value} ({counts[value]} compounds)" if value in counts else value
        ),
        key=f"{key}_scaffold",
    )
    available = (
        scaffold_table
        if scaffold == "All compounds"
        else scaffold_table[scaffold_table["murcko"].fillna("Acyclic") == scaffold]
    )
    available_ids = available["molecule_id"].astype(str).tolist()
    selection_scope = _signature(
        {
            "source": dataset.source,
            "scaffold": scaffold,
            "available": available[["molecule_id", "smiles", "pactivity"]].to_csv(index=False),
        }
    )
    if st.session_state.get(f"{key}_selection_scope") != selection_scope:
        # A keyed multiselect otherwise retains the previous scaffold's IDs
        # when its options expand. Reset only on an input/filter change, not
        # ordinary reruns, so manual compound edits are still respected.
        st.session_state[f"{key}_compounds"] = available_ids if len(available_ids) <= 100 else []
        st.session_state[f"{key}_selection_scope"] = selection_scope
        _discard_active(key)
    identifiers = st.multiselect(
        "Compounds in this 3D-QSAR series",
        available_ids,
        key=f"{key}_compounds",
    )
    st.caption(f"{len(identifiers)} of {len(available_ids)} available compounds selected.")
    if not identifiers:
        _discard_active(key)
        st.info("Choose a related series of 20–100 compounds. No random sampling is applied.")
        return
    dataset = subset_dataset(dataset, identifiers)
    st.dataframe(
        dataset.table[["molecule_id", "smiles", "pactivity", "n_measurements", "pactivity_range"]],
        hide_index=True,
        width="stretch",
    )
    if not 20 <= len(dataset.table) <= 100:
        st.info(
            f"{len(dataset.table)} unique compounds selected. Nested modelling needs 20–100; "
            "a very small series supports descriptive SAR, not reliable predictive claims."
        )
        _discard_active(key)
        return
    st.markdown("**1 · Prepare and inspect alignment**")
    has_coordinates = all(
        mol.GetNumConformers() and mol.GetConformer().Is3D() for mol in dataset.molecules
    )
    routes = (
        ["prealigned", "rigid_mcs", "constrained"]
        if has_coordinates
        else ["rigid_mcs", "constrained"]
    )
    route_labels = {
        "prealigned": "Keep uploaded alignment",
        "rigid_mcs": "Rigid MCS to reference",
        "constrained": "Core-constrained conformer sampling",
    }
    controls = st.columns(2)
    route = controls[0].selectbox(
        "Alignment route", routes, format_func=lambda value: route_labels[value], key=f"{key}_route"
    )
    reference = controls[1].selectbox(
        "Reference compound", dataset.table["molecule_id"].tolist(), key=f"{key}_reference"
    )
    reference_index = dataset.table["molecule_id"].tolist().index(reference)
    assess_core = True
    preserve_hydrogens = False
    if route == "prealigned":
        assess_core = st.checkbox(
            "Assess common-core alignment", value=True, key=f"{key}_assess_core"
        )
        preserve_hydrogens = st.checkbox(
            "Keep supplied hydrogen coordinates",
            value=False,
            key=f"{key}_keep_h",
            help="For a reviewed pre-aligned SDF. Existing hydrogen coordinates are retained; "
            "finite-coordinate and stereochemistry checks still apply.",
        )
        if not assess_core:
            st.warning(
                "Supplied coordinates will be retained, but core RMSD/coverage will not be "
                "assessed. Declare and review the external alignment protocol. "
                "This does not pass the alignment validation check."
            )
    if route != "prealigned" and not preparation_available():
        _discard_active(key)
        st.info(
            "Generated alignment needs the optional preparation packages: "
            "pip install 'sarscope[docking]'. "
            "No Vina run is involved. You can instead upload a reviewed pre-aligned 3D SDF."
        )
        return
    settings = st.columns(2)
    min_coverage = settings[0].slider(
        "Minimum core coverage", 0.3, 0.9, 0.5, 0.05, key=f"{key}_coverage"
    )
    max_rmsd = settings[1].number_input(
        "Maximum core RMSD (Å)", 0.1, 5.0, 1.0, 0.1, key=f"{key}_rmsd"
    )
    alignment_params = FieldQsarParams(
        alignment=cast(Literal["prealigned", "rigid_mcs", "constrained"], route),
        assess_prealigned_core=assess_core,
        preserve_input_hydrogens=preserve_hydrogens,
        reference_index=reference_index,
        min_core_coverage=min_coverage,
        max_alignment_rmsd=max_rmsd,
    )
    preparation_key = _signature(
        {
            "source": dataset.source,
            "table": dataset.table.to_csv(index=False),
            "alignment": asdict(alignment_params),
        }
    )
    if st.button("Prepare and review alignment", type="primary", key=f"{key}_prepare"):
        st.session_state.pop(f"{key}_result", None)
        st.session_state.pop(f"{key}_prepared", None)
        _discard_active(key)
        st.session_state[f"{key}_reviewed"] = False
        try:
            with st.spinner("Preparing bounded, stereo-checked alignment…"):
                prepared = prepare_field_qsar_stage(dataset, alignment_params)
            st.session_state[f"{key}_prepared"] = preparation_key, prepared
        except Exception as exc:
            st.error(f"Alignment needs review: {exc}")
    cached = st.session_state.get(f"{key}_prepared")
    if cached is None or cached[0] != preparation_key:
        _discard_active(key)
        st.info(
            "Prepare alignment for the current selection before modelling. "
            "Changing inputs invalidates old results."
        )
        return
    prepared = cached[1]
    st.caption(prepared.alignment.assumption)
    st.metric(
        "Whole-series common heavy-atom core coverage",
        f"{prepared.alignment.common_core_coverage:.0%}"
        if math.isfinite(prepared.alignment.common_core_coverage)
        else "Not assessed",
    )
    st.dataframe(prepared.alignment.report, hide_index=True, width="stretch")
    inspect_ids = st.multiselect(
        "Overlay compounds",
        dataset.table["molecule_id"].tolist(),
        default=dataset.table["molecule_id"].tolist()[:3],
        max_selections=10,
        key=f"{key}_overlay",
    )
    components.html(
        viewer_html(
            [
                Chem.MolToMolBlock(mol)
                for mol in prepared.alignment.molecules
                if mol.GetProp("_Name") in inspect_ids
            ]
        ),
        height=460,
    )
    st.caption(
        "Interactive viewer loads 3Dmol.js from its public host; "
        "compound coordinates are embedded locally in this view."
    )
    st.download_button(
        "Download reviewed alignment", aligned_sdf(prepared), "aligned_series.sdf", key=f"{key}_sdf"
    )
    reviewed = st.checkbox(
        "I reviewed the overlay and alignment assumptions", key=f"{key}_reviewed"
    )
    st.markdown("**2 · Choose fields and validation**")
    definition = st.selectbox(
        "Field definition",
        ["sarscope", "tripos52_se"],
        format_func=lambda value: (
            "SARscope · RDKit/UFF/Crippen fields"
            if value == "sarscope"
            else "Public Tripos 5.2 · steric/electrostatic reference components"
        ),
        key=f"{key}_field_definition",
    )
    if definition == "tripos52_se":
        st.info(
            "Uses published Tripos-5.2 organic-atom radius/well-depth parameters, not UFF. "
            "SE fields only: full SYBYL atom typing, data.lip hydrophobicity and directional "
            "H-bond sites have not been reproduced. Gasteiger here means RDKit "
            "Gasteiger–Marsili, not Gasteiger–Hückel. No vendor-equivalence claim."
        )
    controls = st.columns(3)
    method = controls[0].selectbox(
        "Molecular fields",
        ["comsia", "comfa"],
        format_func=lambda v: (
            "CoMSIA-style · selectable Gaussian fields"
            if v == "comsia"
            else "CoMFA-style · LJ and Coulomb"
        ),
        key=f"{key}_method",
    )
    charge = controls[1].selectbox("Partial charges", ["gasteiger", "mmff94"], key=f"{key}_charge")
    if method == "comfa":
        st.warning(
            "CoMFA remains experimental. Strict overlap handling can refuse scoring; "
            "the opt-in SYBYL-documented fitting-mean policy reports replaced terms explicitly."
        )
    spacing = controls[2].selectbox(
        "Grid spacing (Å)", [2.0, 2.5, 3.0, 1.5, 1.0], key=f"{key}_spacing"
    )
    st.caption(
        "Explicit method variant: RDKit radii, UFF LJ well depths, Crippen atomic logP and "
        "atom-centred donor/acceptor features. "
        "Not a replica of proprietary Tripos parameter tables."
        if definition == "sarscope"
        else "Reference components: public Tripos-5.2 R/K parameters "
        "and explicit-H environment rules. "
        "Block scaling is SARscope's training-only implementation; full SYBYL preprocessing "
        "and atom-typing equivalence have not been established."
    )
    with st.expander("Field and model settings"):
        field_choices = ["steric", "electrostatic"]
        if method == "comsia" and definition == "sarscope":
            field_choices += ["hydrophobic", "hbond_donor", "hbond_acceptor"]
        selected_fields = st.multiselect(
            "Fields included in the model",
            field_choices,
            default=field_choices,
            key=f"{key}_field_subset_{method}_{definition}",
            help="The independent steroid reproduction covers steric, electrostatic and "
            "hydrophobic (SEH) fields. H-bond fields have not been reproduced "
            "against the reference.",
        )
        scaling = st.selectbox(
            "Field scaling",
            ["raw", "block_std", "column_std"],
            index=1 if definition == "tripos52_se" else 0,
            key=f"{key}_scaling_{definition}",
            help="Fitted on training rows only. Column scaling can amplify nearly constant fields; "
            "inspect held-out predictions. Historical combined-set scaling is not offered here.",
        )
        selection = st.selectbox(
            "PLS component selection",
            ["relative_improvement", "max_q2"],
            key=f"{key}_selection",
            format_func=lambda v: (
                "Early-stop improvement rule"
                if v == "relative_improvement"
                else "Highest CV q² across the capped search"
            ),
        )
        controls = st.columns(3)
        fraction = controls[0].slider(
            "Keep most-variable field columns (%)", 5, 100, 30, 5, key=f"{key}_fraction"
        )
        cap = controls[1].number_input("Maximum PLS components", 1, 6, 4, key=f"{key}_components")
        permutations = controls[2].selectbox(
            "Training-label permutations", [19, 99, 199, 0], key=f"{key}_permutations"
        )
        controls = st.columns(3)
        split = controls[0].selectbox(
            "Validation split", ["random", "scaffold"], key=f"{key}_split"
        )
        test_fraction = controls[1].slider(
            "Held-out fraction", 0.2, 0.35, 0.25, 0.05, key=f"{key}_test_fraction"
        )
        folds = controls[2].selectbox(
            "CV folds (outer and component selection)", [3, 4, 5], key=f"{key}_folds"
        )
        controls = st.columns(3)
        margin = controls[0].number_input(
            "Grid margin (Å)", 2.0, 8.0, 4.0, 0.5, key=f"{key}_margin"
        )
        alpha = controls[1].number_input(
            "Gaussian attenuation α (Å⁻²)", 0.1, 0.8, 0.3, 0.05, key=f"{key}_alpha"
        )
        seed = controls[2].number_input("Random seed", 1, 2147483647, 42, key=f"{key}_seed")
        phase = st.selectbox(
            "Grid sampling phase",
            [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)],
            format_func=lambda value: (
                "Default lattice" if not any(value) else f"Fractional grid offset {value}"
            ),
            key=f"{key}_grid_phase",
            help="A sampling-sensitivity setting, not an alignment correction. "
            "Repeatedly choosing it from held-out scores invalidates the hold-out.",
        )
        filter_mode = "percentile"
        if method == "comfa":
            filter_mode = st.selectbox(
                "Field filter", ["percentile", "absolute"], key=f"{key}_filter"
            )
        minimum_sd = st.number_input(
            "Absolute σ cutoff (kcal/mol; CoMFA only)",
            0.0,
            10.0,
            2.0,
            0.5,
            disabled=filter_mode != "absolute",
            key=f"{key}_sd",
        )
        missing_policy, electrostatic_cutoff, dielectric_model, max_filled = (
            "refuse",
            None,
            "constant",
            0.1,
        )
        if method == "comfa":
            missing_policy = st.selectbox(
                "Missing electrostatic descriptors",
                ["refuse", "training_mean"],
                index=1 if definition == "tripos52_se" else 0,
                format_func=lambda value: (
                    "Refuse scoring (strict)"
                    if value == "refuse"
                    else "SYBYL-documented policy · fitting-column means, with counts"
                ),
                key=f"{key}_missing_{definition}",
            )
            controls = st.columns(3)
            electrostatic_cutoff = controls[0].selectbox(
                "Electrostatic plateau (± kcal/mol)",
                [None, 30.0, 50.0],
                index=1 if definition == "tripos52_se" else 0,
                format_func=lambda value: (
                    "No electrostatic plateau" if value is None else str(value)
                ),
                key=f"{key}_electrostatic_cutoff_{definition}",
            )
            dielectric_model = controls[1].selectbox(
                "Dielectric function",
                ["constant", "distance"],
                format_func=lambda value: (
                    "Constant · 1/r" if value == "constant" else "Distance-dependent · 1/r²"
                ),
                key=f"{key}_dielectric",
            )
            max_filled = (
                controls[2].slider(
                    "Mean-filled descriptor warning threshold (%)",
                    0,
                    100,
                    10,
                    5,
                    key=f"{key}_max_mean_filled",
                )
                / 100
            )
            if missing_policy == "training_mean":
                st.caption(
                    "Means and masks are learned in each fitting fold, never from held-out "
                    "compounds. NaN overlap fields stay undefined until this explicit modelling "
                    "step. Each prediction reports the replaced fraction. The warning threshold "
                    "is a configurable engineering flag, not a published validation criterion."
                )
    if not selected_fields:
        _discard_active(key)
        st.info("Choose at least one molecular field before modelling.")
        return
    st.caption(
        "Random splitting within one series measures interpolation. Scaffold splitting may be "
        "impossible for one scaffold. PLS components are selected inside training CV and capped "
        "at n/5 in the smallest fitting fold. "
        "The 2D RF baseline uses identical train/test and CV splits."
    )
    params = FieldQsarParams(
        **{
            **asdict(alignment_params),
            "method": method,
            "field_definition": definition,
            "field_subset": tuple(selected_fields),
            "field_scaling": scaling,
            "component_selection": selection,
            "charge_model": charge,
            "spacing": spacing,
            "margin": margin,
            "grid_phase": phase,
            "alpha": alpha,
            "keep_fraction": fraction / 100,
            "component_cap": int(cap),
            "permutations": permutations,
            "seed": int(seed),
            "filter_mode": filter_mode,
            "minimum_sd": minimum_sd,
            "missing_field_policy": missing_policy,
            "max_imputed_fraction": max_filled,
            "electrostatic_cutoff": electrostatic_cutoff,
            "dielectric_model": dielectric_model,
        }
    )
    model = ModelParams(
        split=cast(Literal["random", "scaffold"], split),
        test_fraction=test_fraction,
        cv_folds=folds,
        seed=int(seed),
    )
    model_key = _signature([preparation_key, asdict(params), asdict(model)])
    if st.button(
        "Run experimental 3D QSAR", type="primary", disabled=not reviewed, key=f"{key}_run"
    ):
        st.session_state.pop(f"{key}_result", None)
        _discard_active(key)
        st.session_state.pop("report_zip", None)
        try:
            from dataclasses import replace

            with st.spinner(
                "Calculating fields, nested PLS, matched-split 2D baseline and permutations…"
            ):
                result = run_field_qsar_stage(replace(prepared, params=params), model)
            st.session_state[f"{key}_result"] = model_key, result
        except Exception as exc:
            st.error(f"3D QSAR could not be scored: {exc}")
    cached = st.session_state.get(f"{key}_result")
    if cached is None or cached[0] != model_key:
        _discard_active(key)
        st.info("Run the model for these settings. Earlier results are hidden when inputs change.")
        return
    result = cached[1]
    st.session_state[f"{key}_active_result"] = result
    show_field_result(result, key)


def show_field_result(result: FieldQsarResult, key: str) -> None:
    st.markdown("**3 · Evaluate predictions and explore associations**")
    st.warning(result.status)
    st.dataframe(result.checks, hide_index=True, width="stretch")
    columns = [
        "algorithm",
        "train_r2",
        "cv_pooled_q2",
        "test_r2",
        "test_rmse",
        "test_mae",
        "test_spearman",
        "n_features",
        "selected_components",
        "permutation_pvalue",
    ]
    st.dataframe(result.statistics[columns], hide_index=True, width="stretch")
    test = result.predictions[result.predictions["split"] == "held_out"]
    plot = test.melt(
        id_vars=["molecule_id", "pactivity"],
        value_vars=["prediction", "baseline_prediction"],
        var_name="model",
        value_name="predicted",
    )
    scatter = (
        alt.Chart(plot)
        .mark_circle(size=75)
        .encode(
            x=alt.X("pactivity:Q", title="Observed pActivity", scale=alt.Scale(zero=False)),
            y=alt.Y("predicted:Q", title="Held-out prediction", scale=alt.Scale(zero=False)),
            color="model:N",
            tooltip=["molecule_id", "pactivity", "predicted", "model"],
        )
    )
    st.altair_chart(scatter, width="stretch")
    st.altair_chart(
        alt.Chart(test)
        .mark_circle(size=65)
        .encode(
            x=alt.X(
                "prediction:Q", title="Held-out predicted pActivity", scale=alt.Scale(zero=False)
            ),
            y=alt.Y("residual:Q", title="Observed − predicted"),
            tooltip=["molecule_id", "residual"],
        ),
        width="stretch",
    )
    if not result.permutations.empty:
        histogram = (
            alt.Chart(result.permutations)
            .mark_bar(color="#ac8dd3")
            .encode(
                x=alt.X("q2:Q", bin=alt.Bin(maxbins=15), title="Permuted-label nested CV q²"),
                y="count()",
            )
        )
        line = (
            alt.Chart(pd.DataFrame({"q2": [result.statistics.iloc[0].cv_pooled_q2]}))
            .mark_rule(color="#26ad88", strokeWidth=3)
            .encode(x="q2:Q")
        )
        st.altair_chart(histogram + line, width="stretch")
    st.caption(
        "Permutation testing repeats component selection on training labels only. A small p-value "
        "does not establish assay validity, correct alignment or prospective utility. "
        "Nineteen permutations is only a coarse screen; "
        "zero cannot support a chance-correlation claim."
    )
    st.metric("Held-out compounds inside the training PCA box", f"{result.domain.coverage:.0%}")
    st.dataframe(
        test[["molecule_id", "pactivity", "prediction", "residual", "pca_box_in_domain"]],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "PCA-box membership is a coarse applicability diagnostic, not a reliability guarantee."
    )
    if result.prepared.params.missing_field_policy == "training_mean":
        st.dataframe(
            test[
                [
                    "molecule_id",
                    "missing_retained_fields",
                    "mean_filled_fraction",
                    "missing_electrostatic_fraction",
                    "mean_fill_within_limit",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "Mean-filled predictions are not based on all retained descriptors. "
            "A molecule can lie inside the PCA box because its missing terms were replaced "
            "with averages; domain membership does not remove that uncertainty."
        )
    field_warning = unmodelled_field_warning(result.contours.summary)
    if field_warning:
        st.warning(field_warning)
    active_fields = result.contours.summary.loc[
        result.contours.summary.retained_columns.gt(0), "field"
    ].tolist()
    field = st.selectbox("Contour field", active_fields, key=f"{key}_contour")
    positive, negative = result.contours.levels[field]
    surfaces = [
        {
            "dx": open_dx(result.fields.grid, result.contours.maps[field]),
            "level": level,
            "color": color,
        }
        for level, color in zip((positive, negative), COLORS[field], strict=True)
        if level is not None
    ]
    reference = result.prepared.alignment.molecules[result.prepared.params.reference_index]
    components.html(viewer_html([Chem.MolToMolBlock(reference)], surfaces), height=460)
    st.dataframe(result.contours.summary, hide_index=True, width="stretch")
    st.caption(
        "Contours show coefficient × training standard deviation "
        "at the chosen per-sign percentile. "
        "Unmodelled locations are zero, not evidence of no effect. Field percentages describe this "
        "model's absolute contributions—not binding energy or independent explained variance."
    )
    with st.expander("Charge model and component-selection audit"):
        st.dataframe(pd.DataFrame(result.fields.charges), hide_index=True)
        st.dataframe(result.regression.fitted_model.selection_table_, hide_index=True)
        st.caption(
            "This final inner-selection table is not the outer nested-CV performance estimate."
        )
    show_training_stability(result, key)
    if st.button("Prepare 3D-QSAR evidence download", key=f"{key}_export"):
        st.session_state[f"{key}_archive"] = (
            _signature(result.provenance),
            field_qsar_archive(result),
        )
    archived = st.session_state.get(f"{key}_archive")
    if archived and archived[0] == _signature(result.provenance):
        st.download_button(
            "Download 3D-QSAR report, aligned SDF and contour grids",
            archived[1],
            "sarscope_field_qsar.zip",
            "application/zip",
            key=f"{key}_download",
        )


def show_training_stability(result: FieldQsarResult, key: str) -> None:
    """Explicit bounded diagnostic; never starts on first render or uses test labels."""
    with st.expander("Training-only stability · alignment, charges and grid"):
        st.caption(
            "Refits nested CV on fitting compounds only: two charge models, four half-step "
            "grid offsets, three 0.25 Å / 5° relative pose perturbations, and a common-frame "
            "translation control. This can take a minute. Perturbed poses are not verified "
            "binding modes. Numerical flags are not universal QSAR validity thresholds."
        )
        model_record = result.provenance["model"]
        if result.prepared.params.method != "comsia" or model_record["split"] != "random":
            st.info("This diagnostic currently supports Gaussian fields with random nested CV.")
            return
        if st.button("Check training-only stability", key=f"{key}_stability"):
            from sarscope.field_sensitivity import run_sensitivity, training_dataset

            try:
                with st.spinner("Checking training-only grid, charge and pose sensitivity…"):
                    data = training_dataset(result.prepared, result.regression.train_index.tolist())
                    model = ModelParams(
                        split="random",
                        cv_folds=model_record["cv_folds"],
                        regression_cv=model_record["regression_cv"],
                        seed=model_record["seed"],
                    )
                    stability = run_sensitivity(data, result.prepared.params, model)
                result.training_stability = {
                    "summary": stability.summary,
                    "predictions": stability.predictions,
                    "contributions": stability.contributions,
                    "provenance": stability.provenance,
                }
                result.provenance["training_stability"] = stability.provenance
                st.session_state.pop(f"{key}_archive", None)
                st.session_state.pop("report_zip", None)
            except Exception as exc:
                st.error(f"Stability diagnostics could not complete: {exc}")
        if result.training_stability is None:
            return
        summary = result.training_stability["summary"]
        st.dataframe(summary, hide_index=True, width="stretch")
        if not summary.engineering_stability_flag.all():
            st.warning(
                "Some stability flags failed. Review the affected variants; a small charge "
                "or pose change may materially change this model's conclusions. "
                "This does not by itself identify a coding error."
            )
        plotted = summary[summary.completed & summary.kind.ne("baseline")]
        if not plotted.empty:
            st.altair_chart(
                alt.Chart(plotted)
                .mark_bar()
                .encode(
                    y=alt.Y("variant:N", title="Training-only perturbation"),
                    x=alt.X(
                        "oof_change_p95:Q",
                        title="95th percentile OOF prediction change (log units)",
                    ),
                    color="kind:N",
                    tooltip=["variant", "oof_change_p95", "oof_change_max"],
                ),
                width="stretch",
            )
        st.caption(
            "All stability evidence is included when you prepare the 3D-QSAR download. "
            "No alternative is automatically selected and failed models are not certified."
        )

"""User-triggered assay/upload → alignment review → experimental 3D QSAR."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem

from sarscope.analysis.align3d import preparation_available
from sarscope.analysis.contours import COLORS, open_dx
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_interpretation import interpretation_review, training_pose_ranges
from sarscope.field_qsar import (
    BENCHMARK_STATUS,
    FieldDataset,
    FieldQsarResult,
    assay_contexts,
    dataset_from_assay,
    dataset_from_upload,
    field_split_plan,
    subset_dataset,
    upload_table,
)
from sarscope.field_qsar_report import (
    METRIC_DEFINITIONS,
    aligned_sdf,
    charge_audit,
    field_qsar_archive,
    unmodelled_field_warning,
)
from sarscope.params import FieldQsarParams, ModelParams
from sarscope.pipeline import prepare_field_qsar_stage, run_field_qsar_stage


def _signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def _discard_active(key: str) -> None:
    if st.session_state.pop(f"{key}_active_result", None) is not None:
        st.session_state.pop("report_zip", None)


def viewer_html(blocks: list[str], surfaces: list[dict[str, Any]] | None = None) -> str:
    """JSON-escape uploaded text; the browser fetches 3Dmol.js, not compound URLs."""
    orientation = None
    if blocks:
        from scipy.spatial.transform import Rotation

        mol = Chem.MolFromMolBlock(blocks[0], sanitize=False, removeHs=True)
        if mol is not None and mol.GetNumConformers() and mol.GetNumAtoms() >= 3:
            xyz = mol.GetConformer().GetPositions()
            _, _, axes = np.linalg.svd(xyz - xyz.mean(axis=0), full_matrices=False)
            if np.linalg.det(axes) < 0:
                axes[2] *= -1
            # Camera only: face the molecule's broad plane with a slight 3D tilt.
            # Never rotate model/contour coordinates or change the alignment.
            orientation = (
                (Rotation.from_euler("xy", [18, -15], degrees=True) * Rotation.from_matrix(axes))
                .as_quat()
                .tolist()
            )
    payload = json.dumps(
        {"blocks": blocks, "surfaces": surfaces or [], "orientation": orientation}
    ).replace("<", "\\u003c")
    return """<!doctype html><html><body style="margin:0;background:#19222a">
<div id="view" style="height:600px;width:100%;position:relative"></div>
<button id="reset" style="position:absolute;top:12px;left:12px;z-index:2">Reset view</button>
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
function resetView(){
viewer.zoomTo({model:0});
if(data.orientation){const view=viewer.getView();
view.splice(4,4,...data.orientation);viewer.setView(view);}
viewer.zoom(1.25);viewer.render();
}
resetView();document.getElementById('reset').onclick=resetView;
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


@st.cache_data(show_spinner=False, max_entries=2)
def _field_example_dataset(name: str, source_hash: str) -> FieldDataset:
    from sarscope.field_example import field_example_dataset

    return field_example_dataset(name)


def _restore_example_settings(key: str, reference: str) -> None:
    """Explicit reset only: never fit, auto-review, or relax a validation check."""
    from sarscope.analysis.fields import COMSIA_FIELDS

    defaults = {
        "scaffold": "All compounds",
        "route": "prealigned",
        "reference": reference,
        "assess_core": True,
        "keep_h": False,
        "coverage": 0.5,
        "rmsd": 1.0,
        "field_definition": "sarscope",
        "method": "comsia",
        "charge": "gasteiger",
        "spacing": 2.0,
        "scaling_sarscope": "raw",
        "selection": "relative_improvement",
        "field_subset_comsia_sarscope": list(COMSIA_FIELDS),
        "fraction": 30,
        "components": 4,
        "permutations": 19,
        "split": "random",
        "test_fraction": 0.25,
        "folds": 3,
        "margin": 4.0,
        "alpha": 0.3,
        "seed": 42,
        "grid_phase": (0.0, 0.0, 0.0),
        "reviewed": False,
    }
    st.session_state.update({f"{key}_{suffix}": value for suffix, value in defaults.items()})
    for suffix in ("prepared", "result", "model_error", "archive", "selection_scope", "overlay"):
        st.session_state.pop(f"{key}_{suffix}", None)
    _discard_active(key)


def _field_example_controls(key: str) -> FieldDataset | None:
    from sarscope.field_example import FIELD_EXAMPLES, field_example_archive, field_example_files

    with st.expander("Try a measured 3D-QSAR example"):
        name = st.selectbox(
            "Measured 3D example",
            list(FIELD_EXAMPLES),
            format_func=lambda value: FIELD_EXAMPLES[value].label,
            key=f"{key}_example_name",
        )
        spec = FIELD_EXAMPLES[name]
        contents = field_example_files(name)
        source = json.loads(contents["source.json"])
        verification = json.loads(contents["verification.json"])
        stats = next(s for s in verification["statistics"] if s["algorithm"] == "field_pls")
        if name == "ccr5":
            st.caption(
                "75 real CCR5 antagonist measurements from a published, pre-aligned series "
                "(Aher et al., 2007; deposited by Haga et al., 2025). Deposited log activities "
                "are unchanged—not invented pIC50 values. All 75 records are included. "
                "The neutral-parent preparation preserves heavy-atom coordinates and bonds; "
                "protonation changes are audited. These are not verified crystal poses."
            )
        else:
            st.caption(
                "EGFR limitation case: 50 real IC50 measurements from ChEMBL assay CHEMBL680017 "
                "(Bridges et al., 1996). Generated rigid-MCS poses, not observed binding poses. "
                "The weak results are retained; this is not the recommended positive demo."
            )
        st.caption(
            f"Recorded native 3D PLS default run: held-out R² {stats['test_r2']:.3f}; "
            f"pooled nested-CV q² {stats['cv_pooled_q2']:.3f}; RMSE {stats['test_rmse']:.3f}. "
            "These are 3D results, not the 2D RF baseline. "
            "Changing inputs/settings changes results."
        )
        if name == "ccr5":
            st.warning(
                "Within-series tutorial, not independent validation. Pooled q² ≈ 0.499 is "
                "below the 0.5 check; it remains failed despite positive held-out R². "
                "This does not validate new compounds or guarantee other alignments/splits."
            )
            with st.expander("Other fixed splits · all outcomes retained"):
                st.dataframe(
                    pd.DataFrame(verification["repeat_splits"]), hide_index=True, width="stretch"
                )
                st.caption("Primary seed remains 42; no best-seed selection is applied.")
        st.download_button(
            "Download 3D example CSV",
            contents[spec.csv],
            spec.csv,
            "text/csv",
            key=f"{key}_example_csv",
        )
        st.download_button(
            "Download aligned example SDF",
            contents[spec.sdf],
            spec.sdf,
            "chemical/x-mdl-sdfile",
            key=f"{key}_example_sdf",
        )
        st.download_button(
            "Download 3D example source and walkthrough",
            field_example_archive(name),
            f"{name}_measured_3d_example.zip",
            "application/zip",
            key=f"{key}_example_zip",
        )
        selected = st.checkbox("Use measured 3D-QSAR example", key=f"{key}_example")
        st.button(
            "Restore example model defaults",
            key=f"{key}_example_reset",
            on_click=_restore_example_settings,
            args=(key, source["reference_id"]),
            help="Restores settings and clears results. Does not prepare, review or run a model.",
        )
        st.caption(
            f"Data: {source['attribution']} · {source['license']}. Separate from the code licence."
        )
        st.caption(
            "Use the aligned SDF to reproduce the recorded run. CSV has no coordinates; "
            "regenerating conformers/alignment may give different results."
        )
    if not selected:
        return None
    st.info(
        "Example loaded with aligned coordinates. Keep uploaded alignment, prepare/review "
        "the overlay, then run 3D QSAR. The design-and-predict step appears below those "
        "results and uses that same fitted 3D model. No modelling starts automatically."
    )
    return _field_example_dataset(name, hashlib.sha256(contents["source.json"]).hexdigest())


def show_field_qsar(
    table: pd.DataFrame | None = None,
    evidence: pd.DataFrame | None = None,
    source: dict[str, Any] | None = None,
    *,
    key: str = "field_qsar",
) -> None:
    st.subheader("3D QSAR · uncertainty-aware 3D SAR")
    st.markdown(
        "Explore a closely related chemical series with comparable measured activities and "
        "a reviewed alignment. Compare 3D predictions with a 2D baseline, stress-test small "
        "pose changes, then inspect contour and design hypotheses. "
        "This is not automatic lead optimisation; predictive claims must be earned per series."
    )
    st.caption(
        "For interpretable 3D SAR, prefer a reviewed pre-aligned SDF. CSV/SMILES remain "
        "available, but generated conformers and a shared scaffold do not establish "
        "a common binding pose. No modelling or sensitivity check starts automatically."
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
    example = _field_example_controls(key) if mode == "Upload my compounds" else None
    dataset = (
        example
        if example is not None
        else (
            _assay_dataset(table, evidence, source or {}, key)
            if mode == "Selected ChEMBL assay"
            else _upload_dataset(key)
        )
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
        index=0,
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
    if scaffold != "All compounds":
        st.caption(
            f"Scaffold filter: {len(available_ids)} of {len(dataset.table)} compounds remain. "
            "Choose All compounds above to restore the full uploaded/assay series."
        )
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

    def select_all() -> None:
        st.session_state[f"{key}_compounds"] = available_ids

    st.button(
        "Select all available compounds",
        on_click=select_all,
        key=f"{key}_select_all",
        disabled=len(available_ids) > 100,
        help="Selects all compounds within the current scaffold filter (maximum 100).",
    )
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
            "alignment and modelling controls appear once 20–100 are selected. "
            "Restore All compounds in the scaffold filter and select all available compounds, "
            "or add comparable measurements. A smaller series supports descriptive SAR only."
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
        "Reference compound",
        dataset.table["molecule_id"].tolist(),
        index=(
            dataset.table["molecule_id"].tolist().index(dataset.source["reference_id"])
            if dataset.source.get("reference_id") in dataset.table["molecule_id"].tolist()
            else 0
        ),
        key=f"{key}_reference",
    )
    reference_index = dataset.table["molecule_id"].tolist().index(reference)
    if route != "prealigned":
        st.warning(
            "Generated alignment is an exploratory pose hypothesis, not an observed binding "
            "mode. Inspect the whole congeneric series and run the training-only pose "
            "sensitivity check before interpreting contours or proposed potency gains."
        )
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
    if prepared.alignment.report.warning.astype(bool).any():
        st.warning(
            "Some compounds fail the chosen alignment thresholds. Focus on a tighter related "
            "series or review the supplied coordinates; do not treat this overlay as "
            "evidence of a shared binding mode."
        )
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
        height=640,
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
        "Grid spacing (Å)", [1.0, 1.5, 2.0, 2.5, 3.0], index=2, key=f"{key}_spacing"
    )
    st.caption(
        (
            "CoMSIA-style Gaussian fields: RDKit radii, partial charges, Crippen atomic logP "
            "and atom-centred donor/acceptor features. "
            if method == "comsia"
            else "CoMFA-style fields: Lennard–Jones steric energy using RDKit radii/UFF "
            "well depths and Coulomb electrostatics. No hydrophobic or H-bond fields. "
        )
        + "Not a replica of proprietary Tripos parameter tables."
        if definition == "sarscope"
        else (
            "CoMSIA-style Gaussian SE fields: "
            if method == "comsia"
            else "CoMFA-style LJ/Coulomb SE fields: "
        )
        + "public Tripos-5.2 R/K parameters and explicit-H environment rules. "
        "Block scaling is SARscope's training-only implementation; full SYBYL preprocessing "
        "and atom-typing equivalence have not been established."
    )
    missing_policy = "refuse"
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
            help="A retained Coulomb descriptor can be undefined at steric overlaps. "
            "Strict mode refuses that prediction. Mean filling uses fitting rows only "
            "and adds uncertainty; it does not validate the model.",
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
            "Gaussian attenuation α (Å⁻²)",
            0.1,
            0.8,
            0.3,
            0.05,
            disabled=method != "comsia",
            key=f"{key}_alpha",
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
        electrostatic_cutoff, dielectric_model, max_filled = (
            None,
            "constant",
            0.1,
        )
        if method == "comfa":
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
    # Do not redisplay pre-update session results with the old check/metric schema.
    model_key = _signature(
        ["2026-10-17-uncertainty-aware-sar", preparation_key, asdict(params), asdict(model)]
    )
    split_error = None
    try:
        field_split_plan(dataset.table, model)
    except ValueError as exc:
        split_error = str(exc)
        st.warning(split_error)
    if not reviewed:
        st.info("To enable Run, tick 'I reviewed the overlay and alignment assumptions' above.")
    if st.button(
        "Run experimental 3D QSAR",
        type="primary",
        disabled=not reviewed or split_error is not None,
        key=f"{key}_run",
    ):
        st.session_state.pop(f"{key}_result", None)
        st.session_state.pop(f"{key}_model_error", None)
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
            st.session_state[f"{key}_model_error"] = (model_key, str(exc))
    error = st.session_state.get(f"{key}_model_error")
    if error is not None and error[0] == model_key:
        _discard_active(key)
        st.error(f"3D QSAR could not be scored: {error[1]}")
        return
    cached = st.session_state.get(f"{key}_result")
    if cached is None or cached[0] != model_key:
        _discard_active(key)
        if reviewed and split_error is None:
            st.info(
                "Run the model for these settings. Earlier results are hidden when inputs change."
            )
        return
    result = cached[1]
    st.session_state[f"{key}_active_result"] = result
    show_field_result(result, key)
    from sarscope.field_design_ui import show_field_design

    show_field_design(result, key)


def show_field_result(result: FieldQsarResult, key: str) -> None:
    st.markdown("**3 · Evaluate predictions and explore associations**")
    st.warning(result.status)
    st.dataframe(result.checks, hide_index=True, width="stretch")
    st.caption(
        "A check mark means its stated criterion passed, not that a larger value is better. "
        "For the outside-grid count, zero means no held-out compound extends beyond the grid. "
        "Calibration's value is Pearson r²; it is not held-out predictive R²."
    )
    columns = [
        "algorithm",
        "train_r2",
        "cv_pooled_q2",
        "cv_fold_mean_r2",
        "cv_fold_sd_r2",
        "test_r2",
        "test_rmse",
        "test_mae",
        "test_spearman",
        "n_features",
        "selected_components",
        "permutation_pvalue",
    ]
    st.dataframe(
        result.statistics[[name for name in columns if name in result.statistics]],
        column_config={
            name: st.column_config.NumberColumn(
                {
                    "cv_pooled_q2": "Pooled outer-CV q²",
                    "cv_fold_mean_r2": "Mean fold R²",
                    "cv_fold_sd_r2": "Fold R² SD",
                }.get(name, name),
                help=description,
                format="%.3f",
            )
            for name, description in METRIC_DEFINITIONS.items()
        },
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Pooled q² uses all out-of-fold errors and the full training-set SST. Mean fold R² "
        "averages scores with each fold's own SST: small, narrow-potency folds can have "
        "extremely negative R² and inflate its SD. They are not interchangeable. "
        "Definitions also accompany statistics.csv in the evidence download."
    )
    baseline = interpretation_review(result)["baseline"]
    if baseline:
        st.markdown("**Keep the simpler 2D model in view**")
        controls = st.columns(3)
        for control, label, name in zip(
            controls,
            ["3D field PLS · test RMSE", "2D ECFP4 RF · test RMSE", "Training mean · test RMSE"],
            ["field_rmse", "rf_rmse", "mean_rmse"],
            strict=True,
        ):
            control.metric(label, f"{baseline[name]:.3f}" if baseline[name] is not None else "—")
        difference = baseline["field_minus_rf_rmse"]
        if difference is not None:
            st.caption(
                (
                    "The 2D baseline has lower test error on this split. "
                    if difference > 0
                    else "The 3D model has no higher test error on this split. "
                )
                + "Same-split point estimates are not a test of superiority. "
                "Do not choose settings repeatedly from these held-out scores."
            )
    test = result.predictions[result.predictions["split"] == "held_out"]
    plot = test.melt(
        id_vars=["molecule_id", "pactivity"],
        value_vars=["prediction", "baseline_prediction"],
        var_name="model",
        value_name="predicted",
    )
    plot["model"] = plot["model"].map(
        {"prediction": "3D field PLS", "baseline_prediction": "2D ECFP4 RF baseline"}
    )
    limits = [
        float(plot[["pactivity", "predicted"]].min().min()),
        float(plot[["pactivity", "predicted"]].max().max()),
    ]
    padding = max(0.1, (limits[1] - limits[0]) * 0.05)
    limits = [limits[0] - padding, limits[1] + padding]
    scale = alt.Scale(domain=limits, zero=False)
    scatter = (
        alt.Chart(plot)
        .mark_point(size=75, filled=True)
        .encode(
            x=alt.X(
                "pactivity:Q",
                title="Observed pActivity",
                scale=scale,
                axis=alt.Axis(tickCount=6, format=".2f"),
            ),
            y=alt.Y(
                "predicted:Q",
                title="Held-out prediction",
                scale=scale,
                axis=alt.Axis(tickCount=6, format=".2f"),
            ),
            color=alt.Color(
                "model:N",
                title="Model",
                scale=alt.Scale(
                    domain=["3D field PLS", "2D ECFP4 RF baseline"], range=["#26ad88", "#dc7c35"]
                ),
            ),
            shape=alt.Shape("model:N", legend=None),
            tooltip=["molecule_id", "pactivity", "predicted", "model"],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"value": limits}))
        .mark_line(color="#929ba6", strokeDash=[6, 4])
        .encode(
            x=alt.X(
                "value:Q",
                title="Observed pActivity",
                scale=scale,
                axis=alt.Axis(tickCount=6, format=".2f"),
            ),
            y=alt.Y(
                "value:Q",
                title="Held-out prediction",
                scale=scale,
                axis=alt.Axis(tickCount=6, format=".2f"),
            ),
        )
    )
    st.altair_chart((diagonal + scatter).properties(height=400), width="stretch")
    st.caption("Dashed y = x line: perfect prediction. Teal: 3D PLS; orange: 2D RF baseline.")
    st.altair_chart(
        alt.Chart(test)
        .mark_circle(size=65)
        .encode(
            x=alt.X(
                "prediction:Q",
                title="Held-out predicted pActivity",
                scale=alt.Scale(zero=False),
                axis=alt.Axis(tickCount=6, format=".2f"),
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
            f"Green vertical line: observed pooled outer-CV q² = "
            f"{result.statistics.iloc[0].cv_pooled_q2:.3f}; bars: permuted-label q²."
        )
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
    show_training_stability(result, key)
    review = interpretation_review(result)
    st.markdown("**Contours · exploratory spatial associations**")
    if review["pose_status"] != "within_tested_limits":
        st.warning(review["pose_message"])
    st.caption(
        "Even a passing prediction-sensitivity check does not establish stable contour "
        "locations. These maps are not protein contacts or instructions guaranteed to "
        "improve potency. Inspect competing SAR explanations and measured pairs."
    )
    if review["dominant_field"]:
        dominant = review["dominant_field"]
        st.warning(
            f"Field imbalance: {dominant['field']} accounts for {dominant['percent']:.1f}% "
            "of absolute contour contribution. Review the scaling assumptions; this does "
            "not mean other molecular properties are biologically irrelevant. "
            "The 95% warning is an interpretation flag, not a validation criterion."
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
    components.html(viewer_html([Chem.MolToMolBlock(reference)], surfaces), height=640)
    summary = result.contours.summary
    st.dataframe(
        summary[
            [
                "field",
                "absolute_contribution_percent",
                "retained_columns",
                "positive_level",
                "negative_level",
            ]
        ],
        column_config={
            "absolute_contribution_percent": st.column_config.NumberColumn(
                "Contribution (%)", format="%.1f"
            ),
            "retained_columns": "Retained columns",
            "positive_level": st.column_config.NumberColumn("Positive level", format="%.3g"),
            "negative_level": st.column_config.NumberColumn("Negative level", format="%.3g"),
        },
        hide_index=True,
        width="stretch",
    )
    interpretation = summary[summary.field.eq(field)].iloc[0]
    st.caption(
        f"{field.replace('_', ' ').title()} · {interpretation.positive_color}: "
        f"{interpretation.positive_interpretation}. {interpretation.negative_color}: "
        f"{interpretation.negative_interpretation}."
    )
    with st.expander("Full contour definitions and colours"):
        st.dataframe(summary.T, width="stretch")
    st.caption(
        "Contours show coefficient × training standard deviation "
        "at the chosen per-sign percentile. "
        "Unmodelled locations are zero, not evidence of no effect. Field percentages describe this "
        "model's absolute contributions—not binding energy or independent explained variance."
    )
    with st.expander("Charge model and component-selection audit"):
        st.markdown("**Charges per compound**")
        audit = charge_audit(result)
        st.dataframe(audit, hide_index=True, width="stretch")
        st.caption(
            "row is the zero-based index in the aligned series (0 is the first compound). "
            "Requested/used identify charge methods; reason describes fallback only. "
            "These rows are compounds, not PLS components."
        )
        st.markdown("**PLS component-selection search**")
        model = result.regression.fitted_model
        st.dataframe(model.selection_table_, hide_index=True, width="stretch")
        st.caption(
            f"Requested maximum: {result.prepared.params.component_cap}; selected: "
            f"{model.selected_components_}. The maximum is not a requirement to try every count. "
            "The smallest inner fitting fold's n/5, feature count/rank and early-stop rule "
            "can shorten the search. This final inner-selection table is not the outer "
            "nested-CV performance estimate. " + getattr(model, "selection_stop_reason_", "")
        )
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
    st.markdown("**Alignment sensitivity · check before interpreting 3D SAR**")
    with st.expander("Training-only stability · alignment, charges and grid", expanded=True):
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
            from sarscope.field_design import field_model_signature
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
                fit_signature = field_model_signature(result)
                stability_record = {**stability.provenance, "fitted_model_signature": fit_signature}
                result.training_stability = {
                    "fitted_model_signature": fit_signature,
                    "summary": stability.summary,
                    "predictions": stability.predictions,
                    "contributions": stability.contributions,
                    "provenance": stability_record,
                }
                result.provenance["training_stability"] = stability_record
                result.provenance.pop("training_stability_attempt_error", None)
                st.session_state.pop(f"{key}_archive", None)
                st.session_state.pop("report_zip", None)
            except Exception as exc:
                result.provenance["training_stability_attempt_error"] = str(exc)
                st.session_state.pop(f"{key}_archive", None)
                st.session_state.pop("report_zip", None)
                st.error(f"Stability diagnostics could not complete: {exc}")
        if result.training_stability is None:
            st.info(interpretation_review(result)["pose_message"])
            return
        summary = result.training_stability["summary"]
        st.dataframe(summary.round(4), hide_index=True, width="stretch")
        if (
            "engineering_stability_flag" not in summary
            or not summary.engineering_stability_flag.all()
        ):
            st.warning(
                "Some stability flags failed. Review the affected variants; a small charge "
                "or pose change may materially change this model's conclusions. "
                "This does not by itself identify a coding error."
            )
        plotted = (
            summary[summary.completed & summary.kind.ne("baseline")].copy()
            if {"completed", "kind"} <= set(summary)
            else pd.DataFrame()
        )
        if not plotted.empty and {"oof_change_p95", "oof_change_max"} <= set(plotted):
            plotted["variant_label"] = plotted.variant.str.replace("_", " ").str.capitalize()
            plotted["oof_change_p95"] = plotted.oof_change_p95.round(8)
            st.altair_chart(
                alt.Chart(plotted)
                .mark_point(filled=True, size=100)
                .encode(
                    y=alt.Y(
                        "variant_label:N",
                        title="Training-only perturbation",
                        axis=alt.Axis(labelLimit=280),
                    ),
                    x=alt.X(
                        "oof_change_p95:Q",
                        title="95th percentile OOF prediction change (log units)",
                    ),
                    color="kind:N",
                    tooltip=["variant", "oof_change_p95", "oof_change_max"],
                )
                .properties(height=max(220, 30 * len(plotted))),
                width="stretch",
            )
        st.caption(
            "A point at zero (including common translation) means no measurable prediction "
            "change at display precision; it is not missing data. "
            "All stability evidence is included when you prepare the 3D-QSAR download. "
            "No alternative is automatically selected and failed models are not certified."
        )
        ranges = training_pose_ranges(result)
        if not ranges.empty:
            st.markdown("**Training compounds · variation across pose stress tests**")
            displayed = ranges.head(20)
            st.altair_chart(
                alt.Chart(displayed)
                .mark_rule(color="#dc7c35", strokeWidth=5)
                .encode(
                    y=alt.Y(
                        "molecule_id:N",
                        sort=displayed.molecule_id.tolist(),
                        title="Training compound",
                        axis=alt.Axis(labelLimit=180),
                    ),
                    x=alt.X(
                        "minimum_pose_oof:Q",
                        title="Training-only perturbed OOF prediction",
                        scale=alt.Scale(zero=False),
                    ),
                    x2="maximum_pose_oof:Q",
                    tooltip=[
                        "molecule_id",
                        "minimum_pose_oof",
                        "maximum_pose_oof",
                        "pose_oof_range",
                        "completed_pose_checks",
                    ],
                )
                .properties(height=max(160, 22 * len(displayed))),
                width="stretch",
            )
            st.dataframe(ranges.round(3), hide_index=True, width="stretch")
            st.caption(
                "Plot: up to 20 largest training-only ranges; table: all fitting compounds. "
                "Minimum/maximum across the completed perturbed, refitted training OOF models. "
                "These are not new-compound prediction intervals or uncertainty in potency "
                "gains. Held-out compounds are excluded."
            )
        review = interpretation_review(result)
        if review["pose_status"] == "within_tested_limits":
            st.info(review["pose_message"])
        else:
            st.warning(review["pose_message"])

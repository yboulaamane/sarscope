"""Streamlit controls for a single cliff pair; importing this never runs docking."""

from __future__ import annotations

import io
from dataclasses import asdict
from typing import Any

import pandas as pd
import streamlit as st

from sarscope.docking import (
    MAX_INPUT_BYTES,
    DockingError,
    DockingSettings,
    docking_request_key,
    docking_result_zip,
    docking_unavailable_reason,
    run_pair_docking,
)


def reference_box(sdf: bytes) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """A bound reference must already share the uploaded receptor coordinate frame."""
    from rdkit import Chem

    if not sdf or len(sdf) > MAX_INPUT_BYTES:
        raise DockingError("Bound-reference SDF must be nonempty and at most 3 MiB.")
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(sdf), removeHs=False))
    if len(molecules) != 1 or molecules[0] is None or not molecules[0].GetNumConformers():
        raise DockingError("Upload exactly one bound reference ligand with 3D coordinates.")
    molecule = molecules[0]
    conformer = molecule.GetConformer()
    if not conformer.Is3D():
        raise DockingError("The reference must contain bound 3D coordinates, not a 2D depiction.")
    xyz = [
        list(conformer.GetAtomPosition(a.GetIdx()))
        for a in molecule.GetAtoms()
        if a.GetAtomicNum() > 1
    ]
    import numpy as np

    positions = np.asarray(xyz)
    if not len(positions) or not np.isfinite(positions).all():
        raise DockingError("Reference ligand has invalid coordinates.")
    lower, upper = positions.min(axis=0), positions.max(axis=0)
    center = tuple(map(float, (lower + upper) / 2))
    size = tuple(map(float, np.maximum(upper - lower + 8, 6)))
    if max(size) > 25:
        raise DockingError("This reference needs a box wider than the 25 Å browser limit.")
    return center, size


def _html(html: str, *, height: int) -> None:
    """Isolate library HTML: ProLIF diagrams reuse DOM IDs and global JavaScript."""
    from streamlit.components.v1 import html as component_html

    component_html(html, height=height, scrolling=True)


def show_docking_result(result: dict[str, Any], *, key: str) -> None:
    a, b = result["ligands"]
    exploratory = any(
        ligand.get("stereochemistry", {}).get("status") == "exploratory_selection"
        for ligand in (a, b)
    )
    delta_activity = b["pactivity"] - a["pactivity"]
    delta_score = b["score_kcal_mol"] - a["score_kcal_mol"]
    columns = st.columns(4)
    columns[0].metric("A · Vina score", f"{a['score_kcal_mol']:.2f} kcal/mol")
    columns[1].metric("B · Vina score", f"{b['score_kcal_mol']:.2f} kcal/mol")
    columns[2].metric(
        "Source ΔpActivity · B − A" if exploratory else "ΔpActivity · B − A",
        f"{delta_activity:+.2f}",
    )
    columns[3].metric("ΔVina score · B − A", f"{delta_score:+.2f} kcal/mol")
    st.caption(
        f"A = {a['molecule_id']}; B = {b['molecule_id']}. Positive ΔpActivity favours B; "
        "negative ΔVina score favours B. These quantities have different units and are "
        "not interchangeable. All diagrams analyse each ligand's top-ranked pose only."
    )
    if exploratory:
        st.warning(
            "Exploratory stereoisomer comparison: experimental activities belong to the "
            "original source records, not verified assignments to these selected states. "
            "Docking-versus-potency rank agreement is therefore not assessed."
        )
        st.dataframe(
            [
                {
                    "ligand": slot,
                    "molecule_id": ligand["molecule_id"],
                    "source_smiles": ligand.get("source_smiles", ligand["smiles"]),
                    "docked_smiles": ligand["smiles"],
                }
                for slot, ligand in zip(("A", "B"), (a, b), strict=True)
            ],
            hide_index=True,
            width="stretch",
        )
    elif abs(delta_score) < 0.05:
        st.info(
            "The docking scores are nearly tied (<0.05 kcal/mol); "
            "do not infer meaningful separation."
        )
    elif delta_activity * delta_score < 0:
        st.info(
            "Docking ranks the experimentally more potent member higher in this pair. "
            "This is not general validation."
        )
    else:
        st.warning(
            "Docking ranks the experimentally less potent member higher in this pair. "
            "Keep this failure in the analysis."
        )
    st.warning(
        "Docking scores and contacts are approximate, pose-based hypotheses. They do not "
        "establish the mechanism of the activity cliff or replace experimental potency. "
        "Protein residue chemistry uses ProLIF templates and implicit H-bond definitions, "
        "not a pKa calculation. Ligand charges and tautomer states come from the input SMILES."
    )
    if st.checkbox("Show overlaid 3D docking poses", value=True, key=f"{key}_3d"):
        try:
            import py3Dmol

            viewer = py3Dmol.view(width="100%", height=400)
            viewer.addModel(result["protein_pdb"], "pdb")
            viewer.setStyle({"model": 0}, {"cartoon": {"color": "spectrum", "opacity": 0.6}})
            for index, (ligand, color) in enumerate(
                zip((a, b), ("cyanCarbon", "magentaCarbon"), strict=True), start=1
            ):
                viewer.addModel(ligand["sdf"], "sdf")
                viewer.setStyle({"model": index}, {"stick": {"colorscheme": color}})
            viewer.zoomTo({"model": [1, 2]})
            _html(viewer._make_html(), height=420)
            st.caption("A: cyan carbons; B: magenta carbons. Receptor shown as a cartoon.")
        except (ImportError, RuntimeError) as exc:
            st.info(f"3D viewer unavailable: {exc}. Download the SDF poses for an external viewer.")
    st.markdown("**2D interaction diagrams · ProLIF**")
    for column, slot, ligand in zip(st.columns(2), ("A", "B"), (a, b), strict=True):
        with column:
            st.markdown(f"**{slot} · {ligand['molecule_id']}**")
            if ligand["diagram_html"]:
                _html(ligand["diagram_html"], height=550)
            else:
                st.info("No interactions detected under the selected geometric definitions.")
    st.markdown("**Retained, lost and gained contacts · A → B**")
    changes = pd.DataFrame(result["interaction_changes"])
    if changes.empty:
        st.info("No residue/type contacts were detected for either top-ranked pose.")
    else:
        counts = changes["change_A_to_B"].value_counts()
        for column, name in zip(st.columns(3), ("retained", "lost", "gained"), strict=True):
            column.metric(name.title(), str(counts.get(name, 0)))
        st.dataframe(
            changes,
            hide_index=True,
            width="stretch",
            column_config={
                "protein_residue": "Protein residue",
                "interaction": "Interaction type",
                "change_A_to_B": "Change · A → B",
                "distance_A_Angstrom": st.column_config.NumberColumn(
                    "A · shortest distance (Å)", format="%.2f"
                ),
                "distance_B_Angstrom": st.column_config.NumberColumn(
                    "B · shortest distance (Å)", format="%.2f"
                ),
            },
        )
        st.caption(
            "Retained means the same residue and interaction type, not necessarily the "
            "same ligand atoms. Distances are minimum detected contact distances in Å."
        )
    with st.expander("Docking protocol, limitations and preparation warnings"):
        st.json(result["manifest"])
        for message in result["manifest"].get("warnings", []):
            st.warning(message)
    st.download_button(
        "Download pair docking results",
        docking_result_zip(result),
        file_name="cliff_pair_docking.zip",
        mime="application/zip",
        key=f"{key}_zip",
    )


def _uploaded_receptor(key: str) -> dict[str, Any] | None:
    label = st.text_input("Receptor label / PDB ID", key=f"{key}_label").strip()
    protein = st.file_uploader("Matching prepared protein PDB", type=["pdb"], key=f"{key}_protein")
    receptor = st.file_uploader(
        "Prepared rigid receptor PDBQT", type=["pdbqt"], key=f"{key}_receptor"
    )
    reference = st.file_uploader(
        "Bound reference ligand SDF (optional)",
        type=["sdf"],
        key=f"{key}_reference",
        help="Bound 3D coordinates in the receptor frame, not a free conformer.",
    )
    if st.button(
        "Set box from bound reference", disabled=reference is None, key=f"{key}_box_reference"
    ):
        try:
            assert reference is not None
            center, size = reference_box(reference.getvalue())
            for axis, c, s in zip("xyz", center, size, strict=True):
                st.session_state[f"{key}_center_{axis}"] = c
                st.session_state[f"{key}_size_{axis}"] = s
        except DockingError as exc:
            st.error(str(exc))
    if protein is None or receptor is None or not label:
        return None
    try:
        if protein.size > MAX_INPUT_BYTES or receptor.size > MAX_INPUT_BYTES:
            raise DockingError("Each receptor upload must be at most 3 MiB.")
        return {
            "protein_pdb": protein.getvalue().decode("utf-8"),
            "receptor_pdbqt": receptor.getvalue().decode("utf-8"),
            "receptor_label": label,
        }
    except (DockingError, UnicodeDecodeError) as exc:
        st.error(str(exc))
        return None


def show_cliff_docking(
    ligands: list[dict[str, Any]], *, key: str, target: dict[str, Any] | None = None
) -> None:
    """All settings belong to this step, not automatic sidebar presets."""
    with st.expander("Dock this cliff pair · optional Vina + ProLIF"):
        if not st.checkbox("Enable pair docking controls", key=f"{key}_enabled"):
            st.caption(
                "Fetch and prepare a PDB receptor, or upload prepared files. "
                "Nothing docks until you review the site and click Dock this pair."
            )
            return
        reason = docking_unavailable_reason()
        if reason:
            st.info(reason)
            return
        st.caption(
            "Two molecules only, one CPU and one active structure job per host. "
            "Ligands are prepared from the selected pair's SMILES. Review the protein "
            "variant, chemical states and pocket before docking. No pH-state enumeration "
            "or redocking validation is performed automatically."
        )
        from sarscope.ligand_states_ui import review_ligand_states

        docking_ligands = review_ligand_states(ligands, key=key)
        if docking_ligands is None:
            st.info(
                "Resolve and confirm the ligand-state choices above before docking. "
                "You can still review and prepare the receptor below."
            )
        st.markdown(
            "Need receptor files? See the [preparation and interpretation guide]"
            "(https://github.com/yboulaamane/sarscope/blob/main/docs/cliff-docking.md)."
        )
        mode = st.radio(
            "Receptor source",
            ["Fetch & prepare PDB", "Upload prepared files"],
            horizontal=True,
            key=f"{key}_mode",
        )
        if mode == "Fetch & prepare PDB":
            from sarscope.receptor_ui import show_auto_receptor

            prepared = show_auto_receptor(target, key=key)
        else:
            prepared = _uploaded_receptor(key)
        centers, sizes = [], []
        for column, axis in zip(st.columns(3), "xyz", strict=True):
            with column:
                centers.append(
                    st.number_input(
                        f"Box center {axis.upper()} (Å)",
                        min_value=-10000.0,
                        max_value=10000.0,
                        value=None,
                        step=0.5,
                        key=f"{key}_center_{axis}",
                    )
                )
                sizes.append(
                    st.number_input(
                        f"Box size {axis.upper()} (Å)",
                        min_value=6.0,
                        max_value=25.0,
                        value=20.0,
                        step=0.5,
                        key=f"{key}_size_{axis}",
                    )
                )
        exhaustiveness = st.slider("Docking exhaustiveness", 1, 16, 8, key=f"{key}_exhaustiveness")
        seed = int(
            st.number_input(
                "Docking random seed",
                min_value=1,
                max_value=2147483647,
                value=42,
                key=f"{key}_seed",
            )
        )
        timeout = st.slider(
            "Total pair-job timeout (seconds)", 30, 300, 180, step=30, key=f"{key}_timeout"
        )
        confirmed = st.checkbox(
            "I confirm target/variant, ligand states and the binding-site box are appropriate",
            key=f"{key}_confirmed",
        )
        request: dict[str, Any] | None = None
        settings = None
        if (
            docking_ligands is not None
            and prepared is not None
            and all(c is not None for c in centers)
        ):
            try:
                assert all(c is not None for c in centers)
                center_values = [c for c in centers if c is not None]
                settings = DockingSettings(
                    (center_values[0], center_values[1], center_values[2]),
                    (sizes[0], sizes[1], sizes[2]),
                    exhaustiveness,
                    seed,
                    timeout_seconds=timeout,
                )
                request = {
                    "ligands": docking_ligands,
                    "protein_pdb": prepared["protein_pdb"],
                    "receptor_pdbqt": prepared["receptor_pdbqt"],
                    "settings": asdict(settings),
                    "receptor_label": prepared["receptor_label"],
                    "receptor_provenance": prepared.get("provenance"),
                }
            except (DockingError, UnicodeDecodeError) as exc:
                st.error(str(exc))
        request_key = docking_request_key(request) if request else None
        stored = st.session_state.get(f"{key}_result")
        if st.button(
            "Dock this pair",
            type="primary",
            disabled=request is None or not confirmed,
            key=f"{key}_run",
        ):
            if stored is not None and stored[0] == request_key:
                st.info("Showing the completed cached job. Change the seed/settings for a new run.")
            else:
                with st.status("Starting bounded pair docking…", expanded=True) as status:
                    try:
                        assert request is not None and settings is not None
                        assert docking_ligands is not None
                        result = run_pair_docking(
                            docking_ligands,
                            request["protein_pdb"],
                            request["receptor_pdbqt"],
                            settings,
                            receptor_label=request["receptor_label"],
                            receptor_provenance=request["receptor_provenance"],
                            on_progress=st.write,
                        )
                        st.session_state[f"{key}_result"] = (request_key, result)
                        stored = (request_key, result)
                        status.update(
                            label="Pair docking complete", state="complete", expanded=False
                        )
                    except DockingError as exc:
                        status.update(label="Pair docking stopped", state="error")
                        st.error(str(exc))
        if stored is not None:
            if stored[0] != request_key:
                st.info(
                    "Pair, receptor or docking settings changed. The previous result is "
                    "hidden; click Dock this pair for the current inputs."
                )
            else:
                show_docking_result(stored[1], key=key)

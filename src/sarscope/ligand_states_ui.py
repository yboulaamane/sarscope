"""User-controlled stereo review, before receptor or ligand preparation."""

from __future__ import annotations

from typing import Any

import streamlit as st

from sarscope.docking import DockingError, docking_request_key
from sarscope.ligand_states import enumerate_ligand_states, inspect_ligand, validate_ligand_state


def _depict(smiles: str) -> None:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    drawer = rdMolDraw2D.MolDraw2DSVG(420, 240)
    drawer.drawOptions().addAtomIndices = True
    drawer.drawOptions().addStereoAnnotation = True
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, Chem.MolFromSmiles(smiles))
    drawer.FinishDrawing()
    st.image(drawer.GetDrawingText())


def review_ligand_states(ligands: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]] | None:
    """Return copies only. No source-data mutation, default stereoisomer or worker job."""
    selected_ligands = []
    ready = True
    for slot, ligand in zip(("A", "B"), ligands, strict=True):
        source = ligand["smiles"]
        local = f"{key}_stereo_{slot}_{docking_request_key(ligand)[:12]}"
        try:
            inspection = inspect_ligand(source)
            features = inspection["unassigned_features"]
            if not features:
                selected_ligands.append(ligand.copy())
                continue
            st.warning(
                f"Ligand {slot} · {ligand['molecule_id']}: {len(features)} undefined stereo "
                "feature(s). Select a docking-only state before preparation."
            )
            st.caption(
                "The source may represent an unspecified isomer or mixture. Selecting a state "
                "does not establish which stereoisomer was measured in the assay. "
                "Original SMILES and pActivity remain unchanged."
            )
            with st.expander(f"Inspect ligand {slot} · original structure and undefined features"):
                _depict(source)
                st.code(source, language=None)
                st.dataframe(features, hide_index=True, width="stretch")
                st.caption("Feature indices are zero-based RDKit atom or bond indices.")
            if st.button(f"Generate alternatives · ligand {slot}", key=f"{local}_generate"):
                try:
                    st.session_state[f"{local}_options"] = enumerate_ligand_states(source)
                except DockingError as exc:
                    st.warning(str(exc))
            options = st.session_state.get(f"{local}_options", [])
            mode = st.radio(
                f"Stereo input · ligand {slot}",
                ["Select generated alternative", "Provide fully specified SMILES"],
                horizontal=True,
                key=f"{local}_mode",
            )
            if mode == "Select generated alternative":
                chosen = (
                    st.selectbox(
                        f"Docking stereoisomer · ligand {slot}",
                        options,
                        index=None,
                        key=f"{local}_choice",
                    )
                    if options
                    else None
                )
            else:
                chosen = (
                    st.text_input(
                        f"Fully specified docking SMILES · ligand {slot}", key=f"{local}_manual"
                    ).strip()
                    or None
                )
            if not chosen:
                ready = False
                continue
            provenance = validate_ligand_state(source, chosen)
            with st.expander(f"Preview selected docking state · ligand {slot}"):
                _depict(provenance["selected_smiles"])
                st.code(provenance["selected_smiles"], language=None)
            confirmation = docking_request_key(provenance)[:12]
            if not st.checkbox(
                f"I accept ligand {slot} as an exploratory stereoisomer, "
                "not a verified assignment of its experimental activity",
                key=f"{local}_accept_{confirmation}",
            ):
                ready = False
            selected_ligands.append(
                ligand | {"smiles": provenance["selected_smiles"], "source_smiles": source}
            )
        except DockingError as exc:
            st.error(f"Ligand {slot}: {exc}")
            ready = False
    return selected_ligands if ready else None

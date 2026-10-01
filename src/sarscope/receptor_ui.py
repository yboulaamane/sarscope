"""User-controlled PDB discovery, site selection and no-upload receptor preparation."""

from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd
import streamlit as st

from sarscope.docking import DockingError, DockingSettings, docking_request_key
from sarscope.receptor import (
    check_site_components,
    inspect_structure,
    predict_pockets,
    prepare_receptor,
)
from sarscope.sources.pdb import PdbClient, normalise_pdb_id


@st.cache_data(ttl=86400, max_entries=32, show_spinner=False)
def search_structures(accession: str) -> list[dict[str, Any]]:
    with PdbClient() as client:
        return client.search(accession)


def _preview(protein: str, site: dict[str, Any]) -> None:
    from sarscope.docking_ui import _html

    try:
        import py3Dmol

        view = py3Dmol.view(width="100%", height=360)
        view.addModel(protein, "pdb")
        view.setStyle({"cartoon": {"color": "spectrum"}})
        if site.get("pdb"):
            view.addModel(site["pdb"], "pdb")
            view.setStyle({"model": 1}, {"stick": {"colorscheme": "cyanCarbon"}})
        center, size = site["center"], site["size"]
        view.addBox(
            {
                "center": dict(zip("xyz", center, strict=True)),
                "dimensions": dict(zip("whd", size, strict=True)),
                "color": "orange",
                "wireframe": True,
            }
        )
        if site["kind"] == "predicted_pocket":
            coords = site["coordinates"]
            for point in coords[:: max(1, len(coords) // 100)][:100]:
                view.addSphere(
                    {
                        "center": dict(zip("xyz", point, strict=True)),
                        "radius": 0.45,
                        "color": "magenta",
                        "opacity": 0.6,
                    }
                )
        view.zoomTo()
        _html(view._make_html(), height=380)
        st.caption(
            "Orange: proposed docking box. Cyan: bound ligand; magenta: predicted pocket points."
        )
    except ImportError:
        st.info("Install py3Dmol for the site preview; coordinates remain visible below.")


def show_auto_receptor(target: dict[str, Any] | None, *, key: str) -> dict[str, Any] | None:
    target = target or {}
    accessions = sorted(
        {c["accession"] for c in target.get("target_components", []) if c.get("accession")}
    )
    accession = (
        st.selectbox("Target UniProt accession", accessions, key=f"{key}_accession")
        if accessions
        else ""
    )
    if not accession:
        st.info(
            "No target UniProt mapping is available here. "
            "Enter a PDB ID and verify its identity yourself."
        )
    expected = target.get("activity_variant")
    variant_label = (
        "All mutation annotations pooled"
        if expected == "any"
        else expected or "No mutation annotation (not confirmed wild-type)"
    )
    st.caption(f"Activity-data variant filter: {variant_label}.")
    if st.button("Find PDB structures", disabled=not accession, key=f"{key}_search"):
        try:
            with st.spinner("Finding up to 20 experimental structures matched to UniProt…"):
                st.session_state[f"{key}_candidates"] = (
                    accession,
                    search_structures(str(accession)),
                )
        except DockingError as exc:
            st.error(str(exc))
    cached = st.session_state.get(f"{key}_candidates")
    candidates = cached[1] if cached and cached[0] == accession else []
    selected = None
    if candidates:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        k: c[k]
                        for k in (
                            "pdb_id",
                            "resolution_A",
                            "method",
                            "ligands",
                            "chain_mutations",
                            "title",
                        )
                    }
                    for c in candidates
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        selected = st.selectbox(
            "Experimental structure",
            [c["pdb_id"] for c in candidates],
            index=None,
            key=f"{key}_candidate",
        )
    elif cached and cached[0] == accession:
        st.info(
            "No experimental structures found in this bounded search. "
            "A known PDB ID can still be tried."
        )
    manual = (
        st.text_input("PDB ID (optional override)", key=f"{key}_pdb_id", max_chars=4)
        .strip()
        .upper()
    )
    identifier = manual or selected
    if st.button("Fetch selected structure", disabled=not identifier, key=f"{key}_fetch"):
        try:
            identifier = normalise_pdb_id(str(identifier))
            with st.spinner("Downloading structure coordinates and verifying chain mapping…"):
                with PdbClient() as client:
                    metadata = client.metadata([identifier], accession=str(accession))
                    if not metadata:
                        raise DockingError("No metadata found for that experimental PDB ID.")
                    entry = metadata[0]
                    if accession and not entry["matched_chains"]:
                        raise DockingError(
                            "This PDB has no chain mapped to the selected target UniProt accession."
                        )
                    cif = client.coordinates(identifier)
                st.session_state[f"{key}_raw"] = ((identifier, accession), entry, cif)
        except DockingError as exc:
            st.error(str(exc))
    raw = st.session_state.get(f"{key}_raw")
    if raw is None or raw[0] != (identifier, accession):
        return None
    _, entry, cif = raw
    st.markdown(f"**{entry['pdb_id']} · {entry['title']}**")
    st.caption(
        "PDB mutation annotations are shown below; "
        "UniProt identity alone does not verify a variant or construct."
    )
    st.json(entry["chain_mutations"])
    chains = st.multiselect(
        "Protein chains to retain",
        entry["protein_chains"],
        default=(entry["matched_chains"] or entry["protein_chains"])[:1],
        key=f"{key}_chains",
    )
    if not chains:
        return None
    if accession and not set(chains).intersection(entry["matched_chains"]):
        st.error("Retain at least one chain mapped to the target accession.")
        return None
    if (
        expected
        and expected != "any"
        and not any(
            str(expected).upper() in entry["chain_mutations"].get(c, "").upper() for c in chains
        )
    ):
        st.warning(
            "PDB mutation annotations do not confirm the activity-data variant. "
            "Verify sequence/construct compatibility before proceeding."
        )
    inspection_key = (hashlib.sha256(cif.encode()).hexdigest(), tuple(chains))
    inspection = st.session_state.get(f"{key}_inspection")
    try:
        if inspection is None or inspection[0] != inspection_key:
            inspection = (inspection_key, inspect_structure(cif, chains))
            st.session_state[f"{key}_inspection"] = inspection
    except DockingError as exc:
        st.error(str(exc))
        return None
    structure = inspection[1]
    protein = structure["protein_pdb"]
    protein_hash = hashlib.sha256(protein.encode()).hexdigest()
    st.caption(
        "Model 1 and selected asymmetric-unit chains are used. Waters are removed; "
        "highest summed-occupancy altlocs are selected. No missing residues are rebuilt."
    )
    with st.expander("Removed components and coordinate-selection decisions"):
        st.json(structure["provenance"])
    references = structure["references"]
    if references:
        st.caption(
            "Bound-reference candidates: organic components with ≥6 heavy atoms within "
            "6 Å of the selected protein. This is proximity, not proof of biological relevance."
        )
    else:
        st.info(
            "No qualifying bound reference found. "
            "Predict candidate pockets below, or define the box manually."
        )
    if st.button("Predict candidate pockets · fpocket", key=f"{key}_predict"):
        with st.status("Predicting candidate pockets…", expanded=True) as status:
            try:
                computed = predict_pockets(protein, on_progress=st.write)
                st.session_state[f"{key}_pockets"] = (protein_hash, computed)
                status.update(label="Pocket prediction complete", state="complete", expanded=False)
            except DockingError as exc:
                status.update(label="Pocket prediction stopped", state="error")
                st.error(str(exc))
    predicted = st.session_state.get(f"{key}_pockets")
    prediction: dict[str, Any] | None = (
        predicted[1] if predicted and predicted[0] == protein_hash else None
    )
    pockets = prediction["pockets"] if prediction else []
    if prediction:
        for message in prediction.get("warnings", []):
            st.warning(message)
        st.warning(
            "Predicted pockets are candidates, not validated binding sites. "
            "Rank/druggability do not establish the site relevant to your activity data."
        )
        if pockets:
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            k: p[k]
                            for k in (
                                "id",
                                "rank",
                                "score",
                                "druggability_score",
                                "volume_A3",
                                "size",
                                "fits_browser_limit",
                            )
                        }
                        for p in pockets
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        else:
            st.info("fpocket found no candidate pockets. Use a manual box or another structure.")
    sites = references + pockets
    labels = {
        s["id"]: ("Bound ligand · " if s["kind"] == "bound_ligand" else "Predicted pocket · ")
        + s["id"]
        for s in sites
    }
    choices = list(labels) + ["manual"]
    selection = st.selectbox(
        "Docking site",
        choices,
        format_func=lambda x: labels.get(x, "Manual box"),
        index=0 if len(references) == 1 else None,
        key=f"{key}_site",
    )
    site = next((s for s in sites if s["id"] == selection), None)
    if selection is None:
        return None
    if site:
        st.write({"center_Å": site["center"], "size_Å": site["size"]})
        if not site["fits_browser_limit"]:
            st.error(
                "This site's full bounding box exceeds 25 Å. It is not silently clipped; "
                "select another site or define a justified manual box."
            )
            return None
        if st.checkbox("Preview selected site and proposed box", value=True, key=f"{key}_preview"):
            _preview(protein, site)
        try:
            check_site_components(
                structure["provenance"],
                DockingSettings(center=tuple(site["center"]), size=tuple(site["size"])),
            )
        except DockingError as exc:
            st.error(str(exc))
            return None
    provenance = structure["provenance"] | {
        "pdb": entry,
        "source_url": f"https://files.rcsb.org/download/{entry['pdb_id']}.cif",
        "target_chembl_id": target.get("target_chembl_id"),
        "target_accession": accession,
        "activity_variant": expected,
        "site": {
            k: v
            for k, v in (site or {"kind": "manual", "id": "manual"}).items()
            if k not in {"coordinates", "pdb"}
        },
        "reference_pdb": site.get("pdb") if site else None,
        "pocket_engine": prediction["engine"]
        | {"input_protein_sha256": prediction["protein_sha256"], "method": prediction["method"]}
        if site and prediction is not None and site["kind"] == "predicted_pocket"
        else None,
    }
    selection_hash = docking_request_key({"protein": protein_hash, "provenance": provenance})
    confirmed = st.checkbox(
        "I reviewed the protein construct/variant, retained chains and selected site",
        key=f"{key}_review_{selection_hash[:12]}",
    )
    st.caption(
        "Meeko will assign template-based hydrogen/charge states, "
        "not optimise protonation at a chosen pH. Unsupported/missing residues stop preparation."
    )
    if st.button("Prepare receptor · Meeko", disabled=not confirmed, key=f"{key}_prepare"):
        with st.status("Preparing receptor…", expanded=True) as status:
            try:
                result = prepare_receptor(protein, provenance, on_progress=st.write)
                st.session_state[f"{key}_prepared"] = (selection_hash, result)
                if site:
                    for axis, center, size in zip("xyz", site["center"], site["size"], strict=True):
                        st.session_state[f"{key}_center_{axis}"] = center
                        st.session_state[f"{key}_size_{axis}"] = size
                status.update(
                    label="Receptor ready; review the grid and dock below",
                    state="complete",
                    expanded=False,
                )
            except DockingError as exc:
                status.update(label="Receptor preparation stopped", state="error")
                st.error(str(exc))
    prepared = st.session_state.get(f"{key}_prepared")
    if prepared is None or prepared[0] != selection_hash:
        return None
    result = prepared[1]
    st.success("Matching protein PDB and receptor PDBQT prepared. No file upload needed.")
    for message in result["provenance"].get("preparation_warnings", []):
        st.warning(message)
    if result["provenance"].get("max_heavy_atom_shift_A", 0) > 0.5:
        st.warning(
            "Meeko moved one or more heavy atoms by >0.5 Å; inspect the prepared receptor/poses."
        )
    return result | {"receptor_label": f"{entry['pdb_id']} · chains {', '.join(chains)}"}

"""SARscope in the browser. Run locally with ``streamlit run streamlit_app.py``.

Only what is implemented is live: target lookup and the raw-record breakdown.
Analysis sections appear as the modules behind them are implemented.
"""

from __future__ import annotations

import collections
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

from sarscope import __version__
from sarscope.__main__ import FETCH_SUMMARY_FIELDS, default_cache_dir
from sarscope.sources.chembl import ChemblClient, ChemblError, normalise_target_id

#: Categorical slot 1 of the reference palette; one series, so one hue.
BAR_COLOR = "#2a78d6"

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

ACTIVITY_TYPES = ["IC50", "Ki", "Kd", "EC50"]

UPCOMING = [
    "Curation log: every record removed, and by which step",
    "Descriptor profile by activity group (Table 2) and PCA (Table 3, Fig. 5)",
    "Murcko scaffold diversity (Table 4) and enrichment factors",
    "Structure-activity similarity maps and activity cliffs (Figs. 8, 9)",
    "QSAR bake-off with scaffold split and leakage audit (Table 6)",
]


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


def breakdown(records: list[dict[str, Any]], field: str) -> pd.DataFrame:
    counts = collections.Counter(
        "(none)" if r.get(field) is None else str(r.get(field)) for r in records
    )
    frame = pd.DataFrame(counts.most_common(), columns=["value", "records"])
    frame["share"] = frame["records"] / len(records)
    return frame


def bar_chart(frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(frame)
        .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X("records:Q", title="Records"),
            y=alt.Y("value:N", sort="-x", title=None),
            tooltip=[
                alt.Tooltip("value:N", title="Value"),
                alt.Tooltip("records:Q", title="Records", format=","),
                alt.Tooltip("share:Q", title="Share", format=".1%"),
            ],
        )
        .properties(height=26 * len(frame) + 30)
    )


def main() -> None:
    st.set_page_config(page_title="SARscope", layout="wide")
    st.title("SARscope")
    st.caption(
        f"v{__version__} · Target ID in, structure-activity report out · "
        "[source](https://github.com/yboulaamane/sarscope)"
    )

    with st.form("target"):
        left, right = st.columns([2, 3])
        raw_id = left.text_input(
            "ChEMBL target ID", value="CHEMBL5145", help="e.g. CHEMBL5145 or 5145"
        )
        types = right.multiselect("Activity types", ACTIVITY_TYPES, default=["IC50"])
        submitted = st.form_submit_button("Look up target")

    if submitted:
        st.session_state.pop("fetched", None)
    if not types:
        st.info("Choose at least one activity type.")
        return

    try:
        target_id = normalise_target_id(raw_id)
    except ValueError as exc:
        st.error(str(exc))
        return

    try:
        with st.spinner("Asking ChEMBL..."):
            target, release, n_records = lookup(target_id, tuple(types))
    except ChemblError as exc:
        st.error(str(exc))
        return

    st.subheader(target["pref_name"])
    cols = st.columns(4)
    cols[0].metric("Target", target["target_chembl_id"])
    cols[1].metric("Organism", target["organism"])
    cols[2].metric("Type", str(target["target_type"]).title())
    cols[3].metric(f"{'/'.join(types)} records", f"{n_records:,}")
    st.caption(f"Data: {release}. Check the name above: a wrong ID fetches a different protein.")

    if n_records == 0:
        st.warning("No records of these types for this target.")
        return

    key = (target_id, tuple(types))
    if st.session_state.get("fetched") != key:
        minutes = max(1, round(n_records / 7000))
        if not st.button(f"Fetch all {n_records:,} records (about {minutes} min the first time)"):
            return
        st.session_state["fetched"] = key

    try:
        with st.spinner(f"Downloading {n_records:,} records from ChEMBL..."):
            records = fetch(target_id, tuple(types))
    except ChemblError as exc:
        st.error(str(exc))
        return

    molecules = len({r.get("molecule_chembl_id") for r in records})
    st.subheader("Raw records")
    cols = st.columns(2)
    cols[0].metric("Records", f"{len(records):,}")
    cols[1].metric("Molecules", f"{molecules:,}")
    st.write("Each chart is one field that curation filters on. Hover a bar for exact counts.")

    fields = list(FETCH_SUMMARY_FIELDS)
    for row in range(0, len(fields), 2):
        for col, field in zip(st.columns(2), fields[row : row + 2], strict=False):
            title, note = FIELD_NOTES[field]
            frame = breakdown(records, field)
            with col:
                st.markdown(f"**{title}**")
                st.caption(note)
                st.altair_chart(bar_chart(frame), width="stretch")
                with st.expander("Table"):
                    st.dataframe(
                        frame,
                        hide_index=True,
                        column_config={"share": st.column_config.NumberColumn(format="percent")},
                    )

    st.subheader("Analysis")
    st.info(
        "Not implemented yet. These sections appear here as the modules land:\n\n"
        + "\n".join(f"- {item}" for item in UPCOMING)
    )


main()

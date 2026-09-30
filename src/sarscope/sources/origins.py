"""ChEMBL activity source IDs: origin, not a claim of independent data."""

from __future__ import annotations

from typing import Any

import pandas as pd

SOURCE_NAMES = {
    1: "ChEMBL literature",
    7: "PubChem BioAssay (integrated)",
    37: "BindingDB (integrated)",
    38: "Patent bioactivity",
}


def source_id(value: Any) -> int | None:
    try:
        return int(value) if value is not None and not pd.isna(value) else None
    except (TypeError, ValueError):
        return None


def source_name(value: Any) -> str:
    origin = source_id(value)
    if origin is None:
        return "Unknown ChEMBL source"
    return SOURCE_NAMES.get(origin, f"ChEMBL source {origin}")


def source_summary(evidence: pd.DataFrame) -> pd.DataFrame:
    """Retained record counts and compound overlap by original ChEMBL source."""
    columns = ["src_id", "source", "records", "molecules", "assays", "shared_molecules"]
    if evidence.empty or "src_id" not in evidence:
        return pd.DataFrame(columns=columns)
    frame = evidence.copy()
    frame["src_id"] = frame["src_id"].map(source_id)
    frame["source"] = frame["src_id"].map(source_name)
    cross_source = frame.groupby("molecule_id")["source"].nunique().gt(1)
    frame["shared"] = frame["molecule_id"].map(cross_source)
    frame["shared_molecule_id"] = frame["molecule_id"].where(frame["shared"])
    summary = (
        frame.groupby(["src_id", "source"], dropna=False)
        .agg(
            records=("record_id", "size"),
            molecules=("molecule_id", "nunique"),
            assays=("assay_chembl_id", "nunique"),
            shared_molecules=("shared_molecule_id", "nunique"),
        )
        .reset_index()
        .sort_values(["records", "source"], ascending=[False, True])
    )
    return summary[columns]


def source_disagreement(evidence: pd.DataFrame) -> pd.DataFrame:
    """Cross-origin median potency spread; a diagnostic, not an assay correction."""
    columns = [
        "molecule_id",
        "origins",
        "origin_count",
        "measurements",
        "endpoint_types",
        "source_medians",
        "median_pactivity_spread",
    ]
    needed = {"molecule_id", "src_id", "pactivity"}
    if evidence.empty or not needed <= set(evidence):
        return pd.DataFrame(columns=columns)
    frame = evidence.copy()
    frame["source"] = frame["src_id"].map(source_name)
    grouped = (
        frame.groupby(["molecule_id", "source"], dropna=False)["pactivity"].median().reset_index()
    )
    origin_counts = grouped.groupby("molecule_id")["source"].size()
    shared_ids = origin_counts[origin_counts > 1].index
    if shared_ids.empty:
        return pd.DataFrame(columns=columns)
    grouped = grouped[grouped["molecule_id"].isin(shared_ids)]
    shared_records = frame[frame["molecule_id"].isin(shared_ids)]
    record_counts = shared_records.groupby("molecule_id").size()
    if "standard_type" in shared_records:
        endpoint_types_by_molecule = shared_records.groupby("molecule_id")["standard_type"].agg(
            lambda values: "; ".join(sorted(values.dropna().astype(str).unique()))
        )
    else:
        endpoint_types_by_molecule = pd.Series(dtype=str)
    rows = []
    for molecule_id, origins in grouped.groupby("molecule_id"):
        medians = dict(zip(origins["source"], origins["pactivity"], strict=True))
        rows.append(
            {
                "molecule_id": molecule_id,
                "origins": "; ".join(sorted(medians)),
                "origin_count": len(medians),
                "measurements": int(record_counts[molecule_id]),
                "endpoint_types": endpoint_types_by_molecule.get(molecule_id, ""),
                "source_medians": "; ".join(
                    f"{name}: {value:.2f}" for name, value in sorted(medians.items())
                ),
                "median_pactivity_spread": float(
                    origins["pactivity"].max() - origins["pactivity"].min()
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(
        "median_pactivity_spread", ascending=False
    )

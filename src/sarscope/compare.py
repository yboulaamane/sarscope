"""Medicinal-chemistry comparison of two ChEMBL targets."""

from __future__ import annotations

import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sarscope import provenance
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.curate import curate_chembl
from sarscope.params import RunParams
from sarscope.sources.chembl import ChemblClient


@dataclass
class ComparisonResult:
    target_a: dict[str, Any]
    target_b: dict[str, Any]
    compounds: pd.DataFrame
    scaffolds: pd.DataFrame
    provenance: dict[str, Any]


def _key(table: pd.DataFrame) -> pd.Series:
    return table["inchikey"].fillna(table["smiles"]).astype(str)


def _target_table(records: list[dict[str, Any]], params: RunParams, suffix: str) -> pd.DataFrame:
    table = add_scaffolds(curate_chembl(records, params).table).copy()
    table["structure_key"] = _key(table)
    return table.rename(
        columns={
            "molecule_id": f"molecule_id_{suffix}",
            "pactivity": f"pactivity_{suffix}",
            "activity_class": f"activity_class_{suffix}",
            "group": f"group_{suffix}",
        }
    )


def compare_tables(
    table_a: pd.DataFrame, table_b: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return shared-compound selectivity and shared-scaffold summaries."""
    a_cols = [
        "structure_key",
        "molecule_id_a",
        "smiles",
        "murcko",
        "pactivity_a",
        "activity_class_a",
        "group_a",
    ]
    b_cols = [
        "structure_key",
        "molecule_id_b",
        "pactivity_b",
        "activity_class_b",
        "group_b",
    ]
    compounds = table_a[a_cols].merge(table_b[b_cols], on="structure_key", how="inner")
    compounds["delta_pactivity_a_minus_b"] = compounds["pactivity_a"] - compounds["pactivity_b"]
    compounds["selectivity_ratio_a_over_b"] = 10.0 ** compounds["delta_pactivity_a_minus_b"]
    compounds["preferred_target"] = compounds["delta_pactivity_a_minus_b"].map(
        lambda value: "A" if value > 0 else "B" if value < 0 else "tie"
    )
    compounds = compounds.sort_values(
        "delta_pactivity_a_minus_b", key=lambda values: values.abs(), ascending=False
    ).reset_index(drop=True)

    def scaffold_stats(table: pd.DataFrame, suffix: str) -> pd.DataFrame:
        scaffold = table.dropna(subset=["murcko"])
        return (
            scaffold.groupby("murcko", sort=False)
            .agg(
                **{
                    f"n_{suffix}": (f"molecule_id_{suffix}", "size"),
                    f"median_pactivity_{suffix}": (f"pactivity_{suffix}", "median"),
                    f"active_fraction_{suffix}": (
                        f"group_{suffix}",
                        lambda values: float((values == 1).mean()),
                    ),
                }
            )
            .reset_index(names="scaffold")
        )

    scaffolds = scaffold_stats(table_a, "a").merge(
        scaffold_stats(table_b, "b"), on="scaffold", how="inner"
    )
    shared_selectivity = (
        compounds.dropna(subset=["murcko"])
        .groupby("murcko")["delta_pactivity_a_minus_b"]
        .agg(shared_compounds="size", median_selectivity_delta="median")
        .reset_index(names="scaffold")
    )
    scaffolds = scaffolds.merge(shared_selectivity, on="scaffold", how="left")
    scaffolds["active_fraction_delta_a_minus_b"] = (
        scaffolds["active_fraction_a"] - scaffolds["active_fraction_b"]
    )
    baseline_a = float((table_a["group_a"] == 1).mean())
    baseline_b = float((table_b["group_b"] == 1).mean())
    scaffolds["enrichment_a"] = (
        scaffolds["active_fraction_a"] / baseline_a if baseline_a else np.nan
    )
    scaffolds["enrichment_b"] = (
        scaffolds["active_fraction_b"] / baseline_b if baseline_b else np.nan
    )
    scaffolds["enrichment_delta_a_minus_b"] = scaffolds["enrichment_a"] - scaffolds["enrichment_b"]
    scaffolds["enriched_target"] = [
        "A"
        if a > 1 and b <= 1
        else "B"
        if b > 1 and a <= 1
        else "both"
        if a > 1 and b > 1
        else "neither"
        for a, b in zip(scaffolds["enrichment_a"], scaffolds["enrichment_b"], strict=True)
    ]
    scaffolds = scaffolds.sort_values(
        ["enrichment_delta_a_minus_b", "n_a", "n_b"],
        key=lambda values: values.abs() if values.name == "enrichment_delta_a_minus_b" else values,
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return compounds, scaffolds


def run_compare(
    target_a: str, target_b: str, params: RunParams, client: ChemblClient
) -> ComparisonResult:
    metadata_a = client.target(target_a)
    metadata_b = client.target(target_b)
    raw_a = client.activities(target_a, params.curation.standard_types)
    raw_b = client.activities(target_b, params.curation.standard_types)
    table_a = _target_table(raw_a, params, "a")
    table_b = _target_table(raw_b, params, "b")
    compounds, scaffolds = compare_tables(table_a, table_b)
    record = provenance.collect(
        params,
        {
            "kind": "chembl_comparison",
            "release": client.release,
            "targets": [
                provenance.chembl_source(metadata_a, client.release, len(raw_a)),
                provenance.chembl_source(metadata_b, client.release, len(raw_b)),
            ],
        },
    )
    return ComparisonResult(metadata_a, metadata_b, compounds, scaffolds, record)


def write_comparison(result: ComparisonResult, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()) and not (out_dir / "provenance.json").exists():
        raise FileExistsError(
            f"{out_dir} is not empty and is not a SARscope output folder; refusing to write"
        )
    tables = out_dir / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    result.compounds.to_csv(tables / "shared_compound_selectivity.csv", index=False)
    result.scaffolds.to_csv(tables / "shared_scaffold_selectivity.csv", index=False)
    (out_dir / "provenance.json").write_text(json.dumps(result.provenance, indent=2))

    name_a = str(result.target_a.get("pref_name") or result.target_a.get("target_chembl_id"))
    name_b = str(result.target_b.get("pref_name") or result.target_b.get("target_chembl_id"))
    table = result.compounds.head(50).to_html(index=False, escape=True, float_format="%.3g")
    scaffold_table = result.scaffolds.head(50).to_html(
        index=False, escape=True, float_format="%.3g"
    )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>SARscope comparison</title>"
        "<style>body{max-width:1200px;margin:32px auto;font:15px/1.5 sans-serif;padding:0 16px}"
        "table{border-collapse:collapse;width:100%;font-size:12px}th,td{padding:5px 7px;"
        "border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{"
        "text-align:left}"
        "code{font-size:12px}</style></head><body>"
        f"<h1>{html.escape(name_a)} vs {html.escape(name_b)}</h1>"
        f"<p>{len(result.compounds):,} shared structures and {len(result.scaffolds):,} shared "
        "Murcko scaffolds. Positive selectivity deltas favour target A; the potency ratio is "
        "10<sup>pA-pB</sup>.</p>"
        "<h2>Shared compounds</h2>"
        f"{table}<h2>Shared scaffolds</h2>{scaffold_table}"
        "<p>Complete CSV tables and all settings are in <code>tables/</code> and "
        "<code>provenance.json</code>.</p></body></html>"
    )
    path = out_dir / "compare.html"
    path.write_text(page, encoding="utf-8")
    return path

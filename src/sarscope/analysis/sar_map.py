"""Traceable position-level SAR from compounds differing at one R-group only.

Assay-matched deltas use measurement medians within the same assay, endpoint
and variant annotation. Pooled molecule values are a separate evidence tier.
Pair counts are never called independent replication: compounds can be reused.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import combinations, product
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope.analysis.rgroups import ScaffoldSar, _substituent_stats
from sarscope.variants import normalize_variant

PAIR_COLUMNS = [
    "position",
    "fragment_from",
    "fragment_to",
    "id_from",
    "id_to",
    "context",
    "assay_id",
    "endpoint",
    "variant",
    "evidence_tier",
    "pactivity_from",
    "pactivity_to",
    "delta_pactivity",
    "record_ids_from",
    "record_ids_to",
]
SUMMARY_COLUMNS = [
    "position",
    "fragment_from",
    "fragment_to",
    "evidence_tier",
    "endpoint",
    "variant",
    "supporting_pairs",
    "distinct_contexts",
    "disjoint_pairs",
    "assays",
    "median_delta",
    "q25_delta",
    "q75_delta",
    "positive_pairs",
    "negative_pairs",
    "direction_agreement",
    "evidence_strength",
    "interpretation",
]


@dataclass
class SarMap:
    series: ScaffoldSar
    pairs: pd.DataFrame
    transformations: pd.DataFrame
    annotations: pd.DataFrame
    pair_limit_reached: bool
    effect_threshold: float
    record_ids: tuple[str, ...]


def filter_series(
    series: ScaffoldSar, evidence: pd.DataFrame, filters: dict[str, list[Any]]
) -> tuple[ScaffoldSar, pd.DataFrame]:
    """Reaggregate retained measurements; keep original atom/position mapping.

    Keys absent from ``filters`` impose no restriction. Empty selections are
    deliberately empty, allowing the UI to honour a cleared multiselect.
    """
    members = series.members.copy()
    if evidence.empty:
        return series, evidence.copy()
    records = evidence[evidence["molecule_id"].isin(members["molecule_id"])].copy()
    for field, selected in filters.items():
        if field in records:
            records = records[records[field].fillna("Unreported").isin(selected)]
    values = pd.to_numeric(records["pactivity"], errors="coerce")
    records = records[np.isfinite(values)].copy()
    records["pactivity"] = values.loc[records.index]
    medians = records.groupby("molecule_id")["pactivity"].median()
    members = members[members["molecule_id"].isin(medians.index)].copy()
    members["pactivity"] = members["molecule_id"].map(medians)
    # Positions are intentionally preserved after filtering: disappearing
    # alternatives should be visible as insufficient evidence, not renumbered.
    filtered = ScaffoldSar(
        series.scaffold,
        series.core,
        len(members),
        series.n_unmatched,
        members.reset_index(drop=True),
        _substituent_stats(members, series.positions, 1),
    )
    return filtered, records


def fragment_label(fragment: str) -> str:
    """Readable labels for common terminal fragments, SMILES for other chemistry."""
    if fragment == "H":
        return "H"
    molecule = Chem.MolFromSmiles(fragment)
    if molecule is None:
        return fragment
    for atom in molecule.GetAtoms():
        atom.SetAtomMapNum(0)
    clean = Chem.MolToSmiles(molecule)
    names = {
        "*F": "F",
        "*Cl": "Cl",
        "*Br": "Br",
        "*I": "I",
        "*C": "Me",
        "*OC": "OMe",
        "*O": "OH",
        "*N": "NH₂",
        "*C#N": "CN",
        "*C(F)(F)F": "CF₃",
        "*CC": "Et",
        "*CCC": "nPr",
        "*CC(C)": "iPr",
        "*CCCC": "nBu",
        "*c1ccccc1": "Ph",
        "*OCC": "OEt",
    }
    return names.get(clean, clean)


def _measurements(evidence: pd.DataFrame) -> dict[str, dict[tuple[str, str, str], dict[str, Any]]]:
    result: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = {}
    if evidence.empty or not {
        "molecule_id",
        "assay_chembl_id",
        "standard_type",
        "pactivity",
    } <= set(evidence):
        return result
    records = evidence.dropna(subset=["assay_chembl_id", "standard_type", "pactivity"]).copy()
    if "standard_relation" in records:
        records = records[records["standard_relation"] == "="]
    records = records.assign(
        _variant=[
            ";".join(annotation.normalized_mutations) or annotation.raw_annotation or "Unreported"
            for annotation in (
                normalize_variant(value, source="chembl_structured")
                for value in records.get(
                    "assay_variant_mutation", pd.Series(None, index=records.index)
                )
            )
        ]
    )
    fields = ["molecule_id", "assay_chembl_id", "standard_type", "_variant"]
    for (identifier, assay, endpoint, variant), block in records.groupby(fields, sort=True):
        result.setdefault(str(identifier), {})[(str(assay), str(endpoint), str(variant))] = {
            "value": float(block["pactivity"].median()),
            "records": ";".join(
                sorted(set(block.get("record_id", pd.Series(dtype=str)).astype(str)))
            ),
        }
    return result


def _disjoint_count(frame: pd.DataFrame) -> int:
    """Conservative deterministic set of pairs sharing no molecule IDs."""
    used: set[str] = set()
    count = 0
    for left, right in sorted(set(zip(frame["id_from"], frame["id_to"], strict=True))):
        if left not in used and right not in used:
            count += 1
            used.update((left, right))
    return count


def rank_transformations(frame: pd.DataFrame) -> pd.DataFrame:
    """Give the graphic and the evidence selector the same supported default."""
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["_priority"] = ranked["evidence_strength"].map(
        {"strong": 4, "moderate": 3, "mixed": 0, "limited": 1}
    )
    ranked.loc[
        (ranked["evidence_strength"] == "mixed") & (ranked["disjoint_pairs"] >= 2), "_priority"
    ] = 2
    ranked["_magnitude"] = ranked["median_delta"].abs()
    return ranked.sort_values(
        ["_priority", "disjoint_pairs", "supporting_pairs", "_magnitude"], ascending=False
    ).drop(columns=["_priority", "_magnitude"])


def build_sar_map(
    series: ScaffoldSar,
    evidence: pd.DataFrame,
    *,
    effect_threshold: float = 0.5,
    max_pairs: int = 20_000,
) -> SarMap:
    """Build bounded comparisons and calibrated annotations for an existing core."""
    if not np.isfinite(effect_threshold) or effect_threshold <= 0 or max_pairs < 1:
        raise ValueError("Use a positive effect threshold and pair limit")
    measured = _measurements(evidence)
    rows: list[dict[str, Any]] = []
    limited = False
    members = series.members.sort_values("molecule_id")
    for position in series.positions:
        others = [name for name in series.positions if name != position]
        blocks = members.groupby(others, sort=True, dropna=False) if others else [((), members)]
        for context, block in blocks:
            context_text = json.dumps(
                dict(
                    zip(others, context if isinstance(context, tuple) else (context,), strict=True)
                ),
                sort_keys=True,
            )
            alternatives = [
                part.to_dict("records") for _, part in block.groupby(position, sort=True)
            ]
            comparisons_by_fragment = (
                pair
                for left_group, right_group in combinations(alternatives, 2)
                for pair in product(left_group, right_group)
            )
            for left, right in comparisons_by_fragment:
                a, b = str(left["molecule_id"]), str(right["molecule_id"])
                shared = sorted(measured.get(a, {}).keys() & measured.get(b, {}).keys())
                comparisons = [
                    (assay, endpoint, variant, "shared assay", measured[a][key], measured[b][key])
                    for key in shared
                    for assay, endpoint, variant in [key]
                ] or [
                    (
                        "",
                        "",
                        "",
                        "pooled / unmatched",
                        {"value": float(left["pactivity"]), "records": ""},
                        {"value": float(right["pactivity"]), "records": ""},
                    )
                ]
                for assay, endpoint, variant, tier, source, destination in comparisons:
                    if len(rows) >= max_pairs:
                        limited = True
                        break
                    rows.append(
                        {
                            "position": position,
                            "fragment_from": str(left[position]),
                            "fragment_to": str(right[position]),
                            "id_from": a,
                            "id_to": b,
                            "context": context_text,
                            "assay_id": assay,
                            "endpoint": endpoint,
                            "variant": variant,
                            "evidence_tier": tier,
                            "pactivity_from": source["value"],
                            "pactivity_to": destination["value"],
                            "delta_pactivity": destination["value"] - source["value"],
                            "record_ids_from": source["records"],
                            "record_ids_to": destination["records"],
                        }
                    )
                if limited:
                    break
            if limited:
                break
        if limited:
            break
    pairs = pd.DataFrame(rows, columns=PAIR_COLUMNS)
    summaries = []
    if not pairs.empty:
        keys = ["position", "fragment_from", "fragment_to", "evidence_tier", "endpoint", "variant"]
        for (position, before, after, tier, endpoint, variant), block in pairs.groupby(
            keys, sort=True
        ):
            # One delta per molecular pair, even when measured in several assays.
            effects = block.groupby(["id_from", "id_to"])["delta_pactivity"].median()
            delta = float(effects.median())
            positive, negative = int((effects > 0).sum()), int((effects < 0).sum())
            agreement = max(positive, negative) / len(effects)
            disjoint = _disjoint_count(block)
            contexts = block["context"].nunique()
            # Opposing assay measurements are evidence of context dependence,
            # even if their pooled pair median has one sign.
            signs = block["delta_pactivity"]
            assay_conflict = bool(
                (signs > effect_threshold).any() and (signs < -effect_threshold).any()
            )
            if abs(delta) < effect_threshold or agreement < 0.8 or assay_conflict:
                interpretation = "mixed / small change"
            else:
                interpretation = "increase" if delta > 0 else "decrease"
            strength = "limited"
            if tier == "shared assay" and not limited and interpretation != "mixed / small change":
                if disjoint >= 3 and contexts >= 2:
                    strength = "strong"
                elif disjoint >= 2:
                    strength = "moderate"
            if interpretation == "mixed / small change":
                strength = "mixed"
            summaries.append(
                {
                    "position": position,
                    "fragment_from": before,
                    "fragment_to": after,
                    "evidence_tier": tier,
                    "endpoint": endpoint,
                    "variant": variant,
                    "supporting_pairs": len(effects),
                    "distinct_contexts": contexts,
                    "disjoint_pairs": disjoint,
                    "assays": block.loc[block["assay_id"] != "", "assay_id"].nunique(),
                    "median_delta": delta,
                    "q25_delta": float(effects.quantile(0.25)),
                    "q75_delta": float(effects.quantile(0.75)),
                    "positive_pairs": positive,
                    "negative_pairs": negative,
                    "direction_agreement": agreement,
                    "evidence_strength": strength,
                    "interpretation": interpretation,
                }
            )
    summary = pd.DataFrame(summaries, columns=SUMMARY_COLUMNS)
    notes = []
    for position in series.positions:
        candidates = summary[summary["position"] == position].copy()
        if candidates.empty:
            notes.append(
                {
                    "position": position,
                    "label": "No single-position comparisons",
                    "detail": "Insufficient evidence",
                    "color": "#9ca3af",
                }
            )
            continue
        best = rank_transformations(candidates).iloc[0]
        before, after = fragment_label(best["fragment_from"]), fragment_label(best["fragment_to"])
        color = "#9ca3af"
        if best["evidence_strength"] == "mixed":
            color = "#f59e0b"
        elif best["evidence_strength"] in {"strong", "moderate"}:
            color = "#10b981" if best["median_delta"] > 0 else "#ef4444"
        notes.append(
            {
                "position": position,
                "label": f"{before} {'↔' if best['evidence_strength'] == 'mixed' else '→'} {after}",
                "detail": f"Δ {best['median_delta']:+.2f} · {best['supporting_pairs']} pairs · "
                f"{best['evidence_strength']}",
                "color": color,
                "endpoint": best["endpoint"],
                "variant": best["variant"],
            }
        )
    return SarMap(
        series,
        pairs,
        summary,
        pd.DataFrame(
            notes, columns=["position", "label", "detail", "color", "endpoint", "variant"]
        ),
        limited,
        effect_threshold,
        tuple(sorted(set(evidence.get("record_id", pd.Series(dtype=str)).astype(str)))),
    )

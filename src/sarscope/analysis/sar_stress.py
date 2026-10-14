"""Bounded, descriptive challenges to an observed positional SAR claim.

No hypothesis is assigned a probability and no proposed analogue is assigned
a potency. Measurement omissions are recomputed from retained raw records;
compound omissions remove every incident comparison, not just one edge.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope.analysis.sar_map import SarMap, _disjoint_count, _measurements

CLAIM_FIELDS = ("position", "fragment_from", "fragment_to", "evidence_tier", "endpoint", "variant")
CHECK_COLUMNS = [
    "test",
    "omitted",
    "pairs",
    "disjoint_pairs",
    "assays",
    "median_delta",
    "direction_agreement",
    "interpretation",
    "retains_direction",
]
CONTEXT_COLUMNS = ["assay_id", "context", *CHECK_COLUMNS[2:-1]]
CYCLE_COLUMNS = [
    "other_position",
    "context_from",
    "context_to",
    "assay_id",
    "endpoint",
    "variant",
    "id_a",
    "id_b",
    "id_c",
    "id_d",
    "delta_first",
    "delta_second",
    "interaction_delta",
]
SUGGESTION_COLUMNS = [
    "action",
    "reason",
    "assay_id",
    "endpoint",
    "variant",
    "existing_ids",
    "proposed_smiles",
    "proposed_pattern",
    "supporting_ids",
    "record_ids",
]


@dataclass
class SarStressTest:
    claim: dict[str, Any]
    checks: pd.DataFrame
    contexts: pd.DataFrame
    thresholds: pd.DataFrame
    cycles: pd.DataFrame
    suggestions: pd.DataFrame
    verdict: str
    limitations: tuple[str, ...]
    settings: dict[str, Any]


def claim_pairs(result: SarMap, claim: pd.Series | dict[str, Any]) -> pd.DataFrame:
    selected = pd.Series(True, index=result.pairs.index)
    for field in CLAIM_FIELDS:
        selected &= result.pairs[field] == claim[field]
    return result.pairs[selected].copy()


def _statistics(pairs: pd.DataFrame, threshold: float) -> dict[str, Any]:
    if pairs.empty:
        return dict(
            pairs=0,
            disjoint_pairs=0,
            assays=0,
            median_delta=np.nan,
            direction_agreement=np.nan,
            interpretation="insufficient evidence",
        )
    effects = pairs.groupby(["id_from", "id_to"])["delta_pactivity"].median()
    delta = float(effects.median())
    agreement = max(int((effects > 0).sum()), int((effects < 0).sum())) / len(effects)
    signs = pairs["delta_pactivity"]
    conflict = bool((signs > threshold).any() and (signs < -threshold).any())
    interpretation = (
        "mixed / small change"
        if abs(delta) < threshold or agreement < 0.8 or conflict
        else "increase"
        if delta > 0
        else "decrease"
    )
    return dict(
        pairs=len(effects),
        disjoint_pairs=_disjoint_count(pairs),
        assays=pairs.loc[pairs.assay_id != "", "assay_id"].nunique(),
        median_delta=delta,
        direction_agreement=agreement,
        interpretation=interpretation,
    )


def _remeasure(pairs: pd.DataFrame, evidence: pd.DataFrame) -> pd.DataFrame:
    """Recompute only the existing shared-assay edges; never fall back to pooling."""
    measured = _measurements(evidence)
    rows = []
    for pair in pairs.to_dict("records"):
        key = (pair["assay_id"], pair["endpoint"], pair["variant"])
        left = measured.get(str(pair["id_from"]), {}).get(key)
        right = measured.get(str(pair["id_to"]), {}).get(key)
        if left is not None and right is not None:
            rows.append(
                {
                    **pair,
                    "pactivity_from": left["value"],
                    "pactivity_to": right["value"],
                    "delta_pactivity": right["value"] - left["value"],
                    "record_ids_from": left["records"],
                    "record_ids_to": right["records"],
                }
            )
    return pd.DataFrame(rows, columns=pairs.columns)


def assemble_analogue(member: pd.Series, position: str, fragment: str) -> str:
    """Join labelled fragments only if the original molecule round-trips exactly.

    No 3D generation, unspecified-stereo enumeration or stereochemistry erasure.
    Older decomposition tables may lack reconstruction metadata; then a pattern,
    not an invented structure, is returned by the suggestion workflow.
    """
    required = {"core_smiles", "rgroup_fragments", "molecule_smiles"}
    if not required <= set(member.index):
        return ""

    def join(fragments: dict[str, str]) -> str:
        molecule = Chem.MolFromSmiles(str(member["core_smiles"]))
        if molecule is None:
            return ""
        for label, value in fragments.items():
            if value == "H":
                value = f"[H][*:{int(label[1:])}]"
            part = Chem.MolFromSmiles(value)
            if part is None:
                return ""
            molecule = Chem.CombineMols(molecule, part)
        molecule = Chem.molzip(molecule)
        Chem.SanitizeMol(molecule)
        molecule = Chem.RemoveHs(molecule)
        if any(atom.GetAtomicNum() == 0 for atom in molecule.GetAtoms()):
            return ""
        return Chem.MolToSmiles(molecule, isomericSmiles=True)

    try:
        fragments = json.loads(str(member["rgroup_fragments"]))
        original = Chem.MolFromSmiles(str(member["molecule_smiles"]))
        if original is None or join(fragments) != Chem.MolToSmiles(original, isomericSmiles=True):
            return ""
        if position not in fragments:
            return ""
        return join({**fragments, position: fragment})
    except (ValueError, RuntimeError, TypeError, KeyError):
        return ""


def _cycles_and_suggestions(
    result: SarMap,
    pairs: pd.DataFrame,
    evidence: pd.DataFrame,
    threshold: float,
    *,
    max_visits: int,
    max_suggestions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    cycles: list[dict[str, Any]] = []
    suggestions: list[dict[str, Any]] = []
    if pairs.empty:
        return pd.DataFrame(columns=CYCLE_COLUMNS), pd.DataFrame(columns=SUGGESTION_COLUMNS), False
    measured = _measurements(evidence)
    members = result.series.members.drop_duplicates("molecule_id").set_index("molecule_id")
    positions = result.series.positions
    position = str(pairs.iloc[0].position)
    others = [p for p in positions if p != position]
    patterns: dict[tuple[str, ...], list[str]] = {}
    for identifier, member in members.iterrows():
        patterns.setdefault(tuple(str(member[p]) for p in positions), []).append(str(identifier))
    seen: set[tuple[Any, ...]] = set()
    visits = 0
    truncated = False
    for pair in pairs.itertuples():
        key = (pair.assay_id, pair.endpoint, pair.variant)
        a, b = members.loc[pair.id_from], members.loc[pair.id_to]
        a_pattern = tuple(str(a[p]) for p in positions)
        b_pattern = tuple(str(b[p]) for p in positions)
        # Multiple IDs at a cell can indicate duplicate structures or incomplete
        # mapping. Do not silently average them into a four-compound experiment.
        if len(patterns[a_pattern]) != 1 or len(patterns[b_pattern]) != 1:
            continue
        for other in others:
            unchanged = [p for p in positions if p not in {position, other}]
            mask = members[position] == pair.fragment_from
            mask &= members[other] != a[other]
            for p in unchanged:
                mask &= members[p] == a[p]
            for c_id, c in members[mask].iterrows():
                visits += 1
                if visits > max_visits:
                    truncated = True
                    break
                c_pattern = tuple(str(c[p]) for p in positions)
                if len(patterns[c_pattern]) != 1:
                    continue
                c_measurement = measured.get(str(c_id), {}).get(key)
                if c_measurement is None:
                    continue
                d_pattern = tuple(
                    pair.fragment_to if p == position else str(c[p]) for p in positions
                )
                d_ids = patterns.get(d_pattern, [])
                if len(d_ids) > 1:
                    continue
                d_id = d_ids[0] if d_ids else ""
                d_measurement = measured.get(d_id, {}).get(key)
                if d_measurement is not None:
                    identity = (
                        "cycle",
                        tuple(sorted([pair.id_from, pair.id_to, str(c_id), d_id])),
                        key,
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    first, second = (
                        float(pair.delta_pactivity),
                        d_measurement["value"] - c_measurement["value"],
                    )
                    rows = [str(a[other]), str(c[other])]
                    ids = [pair.id_from, pair.id_to, str(c_id), d_id]
                    if rows[0] > rows[1]:
                        rows.reverse()
                        first, second = second, first
                        ids = ids[2:] + ids[:2]
                    cycles.append(
                        dict(
                            other_position=other,
                            context_from=rows[0],
                            context_to=rows[1],
                            assay_id=key[0],
                            endpoint=key[1],
                            variant=key[2],
                            id_a=ids[0],
                            id_b=ids[1],
                            id_c=ids[2],
                            id_d=ids[3],
                            delta_first=first,
                            delta_second=second,
                            interaction_delta=second - first,
                        )
                    )
                elif len(suggestions) < max_suggestions:
                    identity = ("missing", d_pattern, key)
                    if identity in seen:
                        continue
                    seen.add(identity)
                    suggestions.append(
                        dict(
                            action="Measure existing analogue"
                            if d_id
                            else "Complete missing analogue",
                            reason=f"Complete a same-assay cycle at {position} / {other} "
                            "to test whether the substitution effect changes with its neighbour.",
                            assay_id=key[0],
                            endpoint=key[1],
                            variant=key[2],
                            existing_ids=d_id,
                            proposed_smiles=""
                            if d_id
                            else assemble_analogue(c, position, pair.fragment_to),
                            proposed_pattern=json.dumps(
                                dict(zip(positions, d_pattern, strict=True)), sort_keys=True
                            ),
                            supporting_ids=";".join([pair.id_from, pair.id_to, str(c_id)]),
                            record_ids=";".join(
                                sorted(
                                    set(
                                        (
                                            str(pair.record_ids_from)
                                            + ";"
                                            + str(pair.record_ids_to)
                                            + ";"
                                            + c_measurement["records"]
                                        ).split(";")
                                    )
                                    - {""}
                                )
                            ),
                        )
                    )
            if truncated:
                break
        if truncated:
            break
    # Prioritize a direct check of the strongest within-pair assay contradiction.
    conflicts = pairs.groupby(["id_from", "id_to"])["delta_pactivity"].agg(["min", "max"])
    conflicts = conflicts[(conflicts["min"] < -threshold) & (conflicts["max"] > threshold)]
    if not conflicts.empty:
        ids = (conflicts["max"] - conflicts["min"]).idxmax()
        block = pairs[(pairs.id_from == ids[0]) & (pairs.id_to == ids[1])]
        suggestions.insert(
            0,
            dict(
                action="Retest conflicting pair",
                reason="The same molecular pair changes direction "
                "between assays. Remeasure both together in a defined protocol; "
                "inspect source conditions first.",
                assay_id=";".join(sorted(set(block.assay_id))),
                endpoint=str(block.iloc[0].endpoint),
                variant=str(block.iloc[0].variant),
                existing_ids=";".join(ids),
                proposed_smiles="",
                proposed_pattern="",
                supporting_ids=";".join(ids),
                record_ids=";".join(
                    sorted(
                        set(
                            ";".join(
                                block.record_ids_from.tolist() + block.record_ids_to.tolist()
                            ).split(";")
                        )
                        - {""}
                    )
                ),
            ),
        )
    if not suggestions and not pairs.empty and _disjoint_count(pairs) < 2:
        pair = pairs.iloc[0]
        suggestions.append(
            dict(
                action="Remeasure sparse pair",
                reason="One non-overlapping pair cannot establish a "
                "transferable substitution rule. Check reproducibility and "
                "obtain another matched context.",
                assay_id=pair.assay_id,
                endpoint=pair.endpoint,
                variant=pair.variant,
                existing_ids=f"{pair.id_from};{pair.id_to}",
                proposed_smiles="",
                proposed_pattern="",
                supporting_ids=f"{pair.id_from};{pair.id_to}",
                record_ids=f"{pair.record_ids_from};{pair.record_ids_to}",
            )
        )
    return (
        pd.DataFrame(cycles, columns=CYCLE_COLUMNS),
        pd.DataFrame(suggestions[:max_suggestions], columns=SUGGESTION_COLUMNS),
        truncated,
    )


def stress_test_sar(
    result: SarMap,
    claim: pd.Series | dict[str, Any],
    evidence: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    max_omissions_per_axis: int = 100,
    max_cycle_visits: int = 5000,
    max_suggestions: int = 12,
) -> SarStressTest:
    """Stress-test one endpoint/variant claim without network calls or ML fits."""
    if max_omissions_per_axis < 1 or max_cycle_visits < 1 or max_suggestions < 1:
        raise ValueError("Stress-test limits must be positive")
    if not thresholds or any(not np.isfinite(t) or t <= 0 for t in thresholds):
        raise ValueError("Effect thresholds must be finite and positive")
    pairs = claim_pairs(result, claim)
    shared = pairs[pairs.evidence_tier == "shared assay"].copy()
    notes = [
        "Descriptive sensitivity analysis, not statistical significance or calibrated confidence.",
        "Shared assay IDs do not guarantee identical conditions or independent replication; "
        "unreported variants are not confirmed wild type.",
    ]
    settings = dict(
        effect_threshold=result.effect_threshold,
        thresholds=list(thresholds),
        max_omissions_per_axis=max_omissions_per_axis,
        max_cycle_visits=max_cycle_visits,
        max_suggestions=max_suggestions,
        core=result.series.core,
        pair_limit_reached=result.pair_limit_reached,
        duplicate_policy="Repeated copies of the same record ID/value removed. "
        "Distinct record IDs retained; equal rounded values do not prove duplication. "
        "One vote per molecular pair, not per record or assay.",
    )
    rows = []
    baseline = _statistics(shared, result.effect_threshold)
    direction = baseline["interpretation"]

    def add(test: str, omitted: str, subset: pd.DataFrame) -> None:
        stats = _statistics(subset, result.effect_threshold)
        rows.append(
            dict(
                test=test,
                omitted=omitted,
                **stats,
                retains_direction=direction in {"increase", "decrease"}
                and stats["interpretation"] == direction,
            )
        )

    add("Baseline", "None", shared)
    checks_truncated = False
    if not shared.empty:
        ids = sorted(set(shared.id_from) | set(shared.id_to))
        checks_truncated |= len(ids) > max_omissions_per_axis
        for identifier in ids[:max_omissions_per_axis]:
            add(
                "Leave one compound out",
                identifier,
                shared[(shared.id_from != identifier) & (shared.id_to != identifier)],
            )
        contexts = sorted(shared.context.unique())
        checks_truncated |= len(contexts) > max_omissions_per_axis
        for context in contexts[:max_omissions_per_axis]:
            add("Leave one chemical context out", context, shared[shared.context != context])
        used_records = set(
            ";".join(shared.record_ids_from.tolist() + shared.record_ids_to.tolist()).split(";")
        ) - {""}
        scoped = evidence.copy()
        if "record_id" in scoped:
            scoped = scoped[scoped.record_id.astype(str).isin(used_records)]
        for field, test in [
            ("assay_chembl_id", "Leave one assay out"),
            ("document_chembl_id", "Leave one publication out"),
        ]:
            if field not in scoped:
                notes.append(f"{test}: metadata unavailable; no independence can be inferred.")
                continue
            labels = scoped[field].fillna("").astype(str)
            options = sorted(set(labels) - {"", "Unreported", "None", "nan"})
            if (labels == "").any() or not options:
                notes.append(f"{test}: some or all retained records lack identifiers.")
            checks_truncated |= len(options) > max_omissions_per_axis
            for label in options[:max_omissions_per_axis]:
                add(test, label, _remeasure(shared, scoped[labels != label]))
    context_rows = [
        dict(assay_id=assay, context=context, **_statistics(block, result.effect_threshold))
        for (assay, context), block in shared.groupby(["assay_id", "context"], sort=True)
    ]
    contexts_frame = pd.DataFrame(context_rows, columns=CONTEXT_COLUMNS)
    threshold_rows = [
        dict(threshold=float(t), **_statistics(shared, float(t))) for t in sorted(set(thresholds))
    ]
    cycles, suggestions, cycles_truncated = _cycles_and_suggestions(
        result,
        shared,
        evidence,
        result.effect_threshold,
        max_visits=max_cycle_visits,
        max_suggestions=max_suggestions,
    )
    checks = pd.DataFrame(rows, columns=CHECK_COLUMNS)
    omissions = checks[checks.test != "Baseline"]
    evaluable = omissions[omissions.pairs > 0]
    if shared.empty:
        verdict = "Insufficient shared-assay evidence"
    elif direction == "mixed / small change":
        verdict = "Mixed or below the selected effect threshold"
    elif baseline["disjoint_pairs"] < 2:
        verdict = "Too sparse to assess transferability"
    elif not evaluable.empty and not evaluable.retains_direction.all():
        verdict = "Conclusion depends on the retained evidence"
    else:
        verdict = "Direction retained in the tested subsets"
    if (omissions.pairs == 0).any():
        notes.append(
            "Some omissions leave no matched pairs: these do not show that the effect persists."
        )
    if checks_truncated or cycles_truncated or result.pair_limit_reached:
        notes.append("A computational limit was reached; this is a partial sensitivity analysis.")
    if not result.series.members.empty and "core_smiles" not in result.series.members:
        notes.append(
            "Legacy series lacks per-molecule core metadata; stereo identity was not audited here."
        )
    notes.append(
        "Proposed analogues are unmeasured hypotheses, not potency predictions or "
        "synthetic feasibility assessments. Cycle interaction deltas lack experimental "
        "error calibration and do not establish a binding mechanism."
    )
    settings.update(
        omissions_truncated=checks_truncated,
        cycle_search_truncated=cycles_truncated,
        tested_omissions=len(omissions),
        evaluable_omissions=len(evaluable),
        available_compounds=len(set(shared.id_from) | set(shared.id_to)),
    )
    return SarStressTest(
        {field: claim[field] for field in CLAIM_FIELDS},
        checks,
        contexts_frame,
        pd.DataFrame(threshold_rows),
        cycles,
        suggestions,
        verdict,
        tuple(notes),
        settings,
    )

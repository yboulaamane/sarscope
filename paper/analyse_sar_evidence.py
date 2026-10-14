"""Exploratory robustness analysis of frozen, single-assay SAR examples.

No acquisition, training, docking, expert review or parameter optimization.
This analysis was specified retrospectively; potency permutations are a
diagnostic null and are not evidence of experimental reproducibility.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from itertools import combinations, product
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdDepictor, rdFingerprintGenerator, rdRGroupDecomposition
from rdkit.Chem.Draw import rdMolDraw2D

from sarscope.analysis.rgroups import ScaffoldSar, _label, _substituent_stats
from sarscope.analysis.sar_map import build_sar_map
from sarscope.provenance import file_sha256, package_versions

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

KEYS = ["position", "fragment_from", "fragment_to", "evidence_tier", "endpoint", "variant"]


def bh_adjust(values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjustment over the complete transformation family."""
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Need a nonempty vector of finite p values between zero and one")
    order = np.argsort(values, kind="stable")
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def directional(deltas: np.ndarray, threshold: float) -> np.ndarray:
    """Exactly reproduce the existing map's sign/threshold/conflict rule."""
    values = np.atleast_2d(deltas)
    median = np.median(values, axis=1)
    agreement = np.maximum((values > 0).sum(axis=1), (values < 0).sum(axis=1)) / values.shape[1]
    conflict = (values > threshold).any(axis=1) & (values < -threshold).any(axis=1)
    return (np.abs(median) >= threshold) & (agreement >= 0.8) & ~conflict


def leave_one_compound_out(block: pd.DataFrame) -> dict[str, Any]:
    """Remove all edges incident on the omitted compound, never one edge only."""
    ids = sorted(set(block.id_from) | set(block.id_to))
    medians = []
    for identifier in ids:
        retained = block[(block.id_from != identifier) & (block.id_to != identifier)]
        if not retained.empty:
            medians.append(float(retained.delta_pactivity.median()))
    original = float(block.delta_pactivity.median())
    return {
        "loco_valid_omissions": len(medians),
        "loco_median_min": min(medians) if medians else np.nan,
        "loco_median_max": max(medians) if medians else np.nan,
        "loco_sign_retention": float(np.mean(np.sign(medians) == np.sign(original)))
        if medians
        else np.nan,
    }


def double_changes(members: pd.DataFrame) -> pd.DataFrame:
    """Enumerate complete four-compound cycles; ambiguous cells are excluded.

    Defined on this RDKit R-group mapping, not an exhaustive MMP-cycle search.
    Values are IC50 differences, never binding free energies.
    """
    positions = [c for c in members if c.startswith("R")]
    rows = []
    for first, second in combinations(positions, 2):
        others = [p for p in positions if p not in {first, second}]
        groups = members.groupby(others, sort=True, dropna=False) if others else [((), members)]
        for context, block in groups:
            cells: dict[tuple[str, str], list[Any]] = {}
            for member in block.to_dict("records"):
                cells.setdefault((member[first], member[second]), []).append(member)
            for a, b in product(
                combinations(sorted(block[first].unique()), 2),
                combinations(sorted(block[second].unique()), 2),
            ):
                corners = [(a[0], b[0]), (a[1], b[0]), (a[0], b[1]), (a[1], b[1])]
                if any(len(cells.get(key, [])) != 1 for key in corners):
                    continue
                selected = [cells[key][0] for key in corners]
                values = [float(row["pactivity"]) for row in selected]
                rows.append(
                    {
                        "position_a": first,
                        "position_b": second,
                        "a0": a[0],
                        "a1": a[1],
                        "b0": b[0],
                        "b1": b[1],
                        "context": str(context),
                        **{
                            f"id_{label}": row["molecule_id"]
                            for label, row in zip(["00", "10", "01", "11"], selected, strict=True)
                        },
                        **{
                            f"pIC50_{label}": value
                            for label, value in zip(["00", "10", "01", "11"], values, strict=True)
                        },
                        "delta_a_b0": values[1] - values[0],
                        "delta_a_b1": values[3] - values[2],
                        "delta_b_a0": values[2] - values[0],
                        "delta_b_a1": values[3] - values[1],
                        "delta_delta_pIC50": values[3] - values[1] - values[2] + values[0],
                    }
                )
    return pd.DataFrame(rows)


def stereo_scope(
    members: pd.DataFrame, molecules: pd.DataFrame, scope: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Audit the per-molecule R-group core, including assigned stereochemistry.

    Largest annotated core state is selected by count, not potency. R/S alone
    is not compared: substituent priorities can change CIP labels. Instead use
    the same mapped core graph with its preserved stereo annotations.
    """
    selected = molecules[molecules.molecule_id.isin(members.molecule_id)]
    rows, unmatched = rdRGroupDecomposition.RGroupDecompose(
        [Chem.MolFromSmiles(scope["scaffold"])],
        [Chem.MolFromSmiles(smi) for smi in selected.smiles],
        asSmiles=True,
        asRows=True,
    )
    if unmatched or len(rows) != len(selected):
        raise ValueError("Stereochemical audit failed to recover the source decomposition")
    positions = [column for column in members if column.startswith("R")]
    saved = members.set_index("molecule_id")
    audit = []
    for identifier, row in zip(selected.molecule_id, rows, strict=True):
        if any(saved.loc[identifier, p] != _label(row.get(p, "")) for p in positions):
            raise ValueError("Recomputed atom mapping differs from the frozen source mapping")
        audit.append({"molecule_id": identifier, "isomeric_core": row["Core"]})
    table = pd.DataFrame(audit)
    counts = table.isomeric_core.value_counts().rename_axis("core").reset_index(name="n")
    core = str(counts.sort_values(["n", "core"], ascending=[False, True]).iloc[0].core)
    table["retained"] = table.isomeric_core == core
    return members[members.molecule_id.isin(table[table.retained].molecule_id)].copy(), table, core


def analyse(results: Path, out: Path, *, repeats: int = 5000, seed: int = 20261013) -> Path:
    if repeats < 100:
        raise ValueError("Use at least 100 diagnostic permutations")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new, empty output directory")
    out.mkdir(parents=True, exist_ok=True)
    statistics, threshold_rows, null_rows, cycles, paired_records, hashes = [], [], [], [], [], {}
    coverage = []
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048, includeChirality=False
    )
    for case_index, case in enumerate(["braf", "drd2", "esr1"]):
        directory = results / case
        for name in [
            "pairs.csv",
            "measurements.csv",
            "series_members.csv",
            "transformations.csv",
            "molecules.csv",
            "scope.json",
        ]:
            hashes[f"{case}/{name}"] = file_sha256(directory / name)
        evidence = pd.read_csv(directory / "measurements.csv")
        members = pd.read_csv(directory / "series_members.csv").sort_values("molecule_id")
        source_molecules = pd.read_csv(directory / "molecules.csv")
        molecules = source_molecules.set_index("molecule_id")
        scope = json.loads((directory / "scope.json").read_text())
        original_members = len(members)
        members, stereo_audit, isomeric_core = stereo_scope(members, source_molecules, scope)
        stereo_audit.to_csv(out / f"{case}-stereo-audit.csv", index=False)
        evidence = evidence[evidence.molecule_id.isin(members.molecule_id)]
        positions = [column for column in members if column.startswith("R")]
        series = ScaffoldSar(
            scope["scaffold"],
            isomeric_core,
            len(members),
            0,
            members,
            _substituent_stats(members, positions, 1),
        )
        sar = build_sar_map(series, evidence)
        pairs, transformations = sar.pairs, sar.transformations
        coverage.append(
            {
                "case_id": case,
                "curated_molecules": len(molecules),
                "original_series_molecules": original_members,
                "stereo_core_states": stereo_audit.isomeric_core.nunique(),
                "selected_series_molecules": len(members),
                "excluded_core_state": original_members - len(members),
                "selected_measurements": len(evidence),
                "position_pair_rows": len(pairs),
                "transformations": len(transformations),
                "selected_core": isomeric_core,
            }
        )
        members.to_csv(out / f"{case}-selected-members.csv", index=False)
        evidence.to_csv(out / f"{case}-selected-measurements.csv", index=False)
        pairs.to_csv(out / f"{case}-selected-pairs.csv", index=False)
        if evidence.assay_chembl_id.nunique() != 1 or set(evidence.standard_type) != {"IC50"}:
            raise ValueError("This diagnostic requires one IC50 assay per cohort")
        if (
            set(pairs.evidence_tier) != {"shared assay"}
            or evidence.assay_variant_mutation.notna().any()
        ):
            raise ValueError("Only assay-matched, structurally unannotated examples are supported")
        ids = members.molecule_id.tolist()
        lookup = {name: i for i, name in enumerate(ids)}
        values = evidence.groupby("molecule_id").pactivity.median().reindex(ids).to_numpy()
        if not np.isfinite(values).all() or not np.allclose(values, members.pactivity):
            raise ValueError("Member potencies do not match the selected assay evidence")
        rng = np.random.default_rng(seed + case_index)
        permuted = np.array([rng.permutation(values) for _ in range(repeats)])
        null_supported = np.zeros(repeats, dtype=int)
        case_thresholds = {
            t: {"strong": 0, "moderate": 0, "mixed": 0, "limited": 0}
            for t in [0.25, 0.5, 0.75, 1.0]
        }
        for transformation in transformations.to_dict("records"):
            block = pairs.copy()
            for key in KEYS:
                block = block[block[key] == transformation[key]]
            block = block.drop_duplicates(["id_from", "id_to"])
            if block.empty:
                raise ValueError("Transformation has no supporting pair records")
            left = np.array([lookup[name] for name in block.id_from])
            right = np.array([lookup[name] for name in block.id_to])
            observed = values[right] - values[left]
            if not np.allclose(observed, block.delta_pactivity):
                raise ValueError("A pair delta disagrees with its assay measurements")
            delta = float(np.median(observed))
            null_delta = np.median(permuted[:, right] - permuted[:, left], axis=1)
            pvalue = (int((np.abs(null_delta) >= abs(delta) - 1e-12).sum()) + 1) / (repeats + 1)
            support_eligible = transformation["disjoint_pairs"] >= 2
            if support_eligible:
                null_supported += directional(permuted[:, right] - permuted[:, left], 0.5)
            for threshold, tally in case_thresholds.items():
                grade = "mixed"
                if directional(observed, threshold)[0]:
                    grade = "limited"
                    if (
                        transformation["disjoint_pairs"] >= 3
                        and transformation["distinct_contexts"] >= 2
                    ):
                        grade = "strong"
                    elif support_eligible:
                        grade = "moderate"
                tally[grade] += 1
                if threshold == 0.5 and grade != transformation["evidence_strength"]:
                    raise ValueError("Diagnostic grade calculation differs from the existing map")
            statistics.append(
                {
                    "case_id": case,
                    **transformation,
                    **leave_one_compound_out(block),
                    "permutation_p_two_sided": pvalue,
                }
            )
        for threshold, tally in case_thresholds.items():
            threshold_rows.append({"case_id": case, "threshold": threshold, **tally})
        observed_count = int(transformations.evidence_strength.isin(["strong", "moderate"]).sum())
        null_rows.append(
            {
                "case_id": case,
                "observed_supported": observed_count,
                "null_mean_supported": float(null_supported.mean()),
                "null_q025": float(np.quantile(null_supported, 0.025)),
                "null_q975": float(np.quantile(null_supported, 0.975)),
                "permutation_p_supported_count": (int((null_supported >= observed_count).sum()) + 1)
                / (repeats + 1),
            }
        )
        pd.DataFrame(
            {"permutation": np.arange(repeats), "supported_transformations": null_supported}
        ).to_csv(out / f"{case}-null-counts.csv", index=False)
        cycle_frame = double_changes(members)
        if not cycle_frame.empty:
            cycles.append(cycle_frame.assign(case_id=case))
        fingerprints = {
            name: generator.GetFingerprint(Chem.MolFromSmiles(molecules.loc[name, "smiles"]))
            for name in ids
        }
        for row in pairs.to_dict("records"):
            sim = DataStructs.TanimotoSimilarity(
                fingerprints[row["id_from"]], fingerprints[row["id_to"]]
            )
            paired_records.append(
                {
                    "case_id": case,
                    **row,
                    "ecfp4_tanimoto": sim,
                    "cliff_0_9_2_0": sim > 0.9 and abs(row["delta_pactivity"]) > 2.0,
                }
            )
    frame = pd.DataFrame(statistics)
    frame["permutation_q_BH"] = bh_adjust(frame.permutation_p_two_sided.to_numpy())
    frame.to_csv(out / "transformation-robustness.csv", index=False)
    pd.DataFrame(threshold_rows).to_csv(out / "threshold-sensitivity.csv", index=False)
    pd.DataFrame(null_rows).to_csv(out / "null-support-summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(out / "stereochemistry-scoped-coverage.csv", index=False)
    pd.concat(cycles, ignore_index=True).to_csv(out / "complete-cycles.csv", index=False)
    pd.DataFrame(paired_records).to_csv(out / "pair-evidence-with-similarity.csv", index=False)
    cliff_grid = []
    paired_frame = pd.DataFrame(paired_records)
    for case, group in paired_frame.groupby("case_id"):
        for similarity, gap in product([0.85, 0.9, 0.95], [1.5, 2.0, 2.5]):
            cliff_grid.append(
                {
                    "case_id": case,
                    "similarity_cutoff": similarity,
                    "gap_cutoff": gap,
                    "mapped_pair_cliffs": int(
                        (
                            (group.ecfp4_tanimoto > similarity)
                            & (group.delta_pactivity.abs() > gap)
                        ).sum()
                    ),
                }
            )
    pd.DataFrame(cliff_grid).to_csv(out / "cliff-cutoff-sensitivity.csv", index=False)
    (out / "provenance.json").write_text(
        json.dumps(
            {
                "analysis": (
                    "Retrospective, exploratory sensitivity, graph-aware label permutations "
                    "and compound omission"
                ),
                "seed": seed,
                "permutations_per_case": repeats,
                "effect_thresholds": [0.25, 0.5, 0.75, 1.0],
                "stereochemistry_scope": (
                    "Largest identical, stereo-annotated mapped core per scaffold; "
                    "unassigned and oppositely assigned states are not pooled"
                ),
                "multiple_testing_family": len(frame),
                "input_sha256": hashes,
                "analysis_script_sha256": file_sha256(Path(__file__)),
                "packages": package_versions(),
                "interpretation": (
                    "A permutation compatibility diagnostic, not proof of causality, calibrated "
                    "support, experimental precision or external validation"
                ),
                "cliff_definition": (
                    "Morgan radius 2, 2048 bits, chirality off; Tanimoto >0.9 and "
                    "abs(delta pIC50)>2.0; only mapped pairs, not all cohort pairs"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    plot_effects(frame, pd.DataFrame(paired_records), out)
    plot_cycle(pd.concat(cycles, ignore_index=True), out)
    return out / "transformation-robustness.csv"


def plot_effects(statistics: pd.DataFrame, pairs: pd.DataFrame, out: Path) -> None:
    selected = [
        ("braf", "R3", "H", "O[*:3]", "BRAF: H → OH, para"),
        ("braf", "R4", "H", "O[*:4]", "BRAF: H → OH, meta"),
        ("braf", "R4", "CO[*:4]", "O[*:4]", "BRAF: OMe → OH, meta"),
        ("drd2", "R2", "Cl[*:2]", "H", "DRD2: Cl → H, para"),
        ("drd2", "R4", "Cl[*:4]", "H", "DRD2: Cl → H, ortho"),
        ("esr1", "R3", "Br[*:3]", "C=C[*:3]", "ERα: Br → vinyl, R3"),
        ("esr1", "R3", "Br[*:3]", "CC=C[*:3]", "ERα: Br → allyl, R3"),
    ]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.axvspan(-0.5, 0.5, color="#e8edf0", zorder=0)
    ax.axvline(0, color="#718096", lw=1)
    exports = []
    for index, (case, position, before, after, label) in enumerate(selected):
        selector = (
            (pairs.case_id == case)
            & (pairs.position == position)
            & (pairs.fragment_from == before)
            & (pairs.fragment_to == after)
        )
        block = pairs[selector].copy()
        row = statistics[
            (statistics.case_id == case)
            & (statistics.position == position)
            & (statistics.fragment_from == before)
            & (statistics.fragment_to == after)
        ].iloc[0]
        color = {"braf": "#278d81", "drd2": "#ab669e", "esr1": "#c48a32"}[case]
        y = len(selected) - 1 - index
        ax.plot([row.loco_median_min, row.loco_median_max], [y, y], color=color, lw=5, alpha=0.35)
        offsets = np.linspace(-0.13, 0.13, len(block))
        ax.scatter(block.delta_pactivity, y + offsets, color=color, s=30, alpha=0.75, zorder=3)
        ax.scatter(
            [row.median_delta], [y], marker="D", color=color, s=65, edgecolor="white", zorder=4
        )
        ax.text(2.65, y, f"n = {len(block)}", va="center", fontsize=10)
        for pair in block.to_dict("records"):
            exports.append({"display_label": label, **pair})
    ax.set_yticks(np.arange(len(selected)), [item[-1] for item in selected][::-1])
    ax.set_xlim(-2.6, 3.3)
    ax.set_xlabel("Measured ΔpIC50 (positive values indicate increased potency)")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out / "position-effects.svg")
    plt.close(fig)
    pd.DataFrame(exports).to_csv(out / "figure-position-effects-data.csv", index=False)


def plot_cycle(cycles: pd.DataFrame, out: Path) -> None:
    selected = cycles[
        (cycles.case_id == "drd2")
        & (cycles.position_a == "R2")
        & (cycles.position_b == "R4")
        & (cycles.a0.isin(["H", "Cl[*:2]"]))
        & (cycles.a1.isin(["H", "Cl[*:2]"]))
        & (cycles.b0.isin(["H", "Cl[*:4]"]))
        & (cycles.b1.isin(["H", "Cl[*:4]"]))
    ]
    if len(selected) != 1:
        raise ValueError("Expected one complete DRD2 para/ortho chlorine cycle")
    members = pd.read_csv(out / "drd2-selected-members.csv")
    ids = [selected.iloc[0][f"id_{key}"] for key in ["00", "10", "01", "11"]]
    quartet = members[members.molecule_id.isin(ids)].copy()
    quartet["IC50_nM"] = 10 ** (9 - quartet.pactivity)
    quartet.to_csv(out / "drd2-chlorine-quartet.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(-0.4, 1.4)
    ax.set_ylim(-0.45, 1.45)
    for row in quartet.itertuples():
        x, y = int(row.R4 != "H"), 1 - int(row.R2 != "H")
        ax.text(
            x,
            y,
            f"{row.molecule_id}\npIC50 {row.pactivity:.3f}\nIC50 {row.IC50_nM:.2f} nM",
            ha="center",
            va="center",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.7", "fc": "#edf5f3", "ec": "#278d81"},
        )
    ax.text(0, 1.35, "ortho H", ha="center", fontsize=12)
    ax.text(1, 1.35, "ortho Cl", ha="center", fontsize=12)
    ax.text(-0.33, 1, "para H", va="center", rotation=90, fontsize=12)
    ax.text(-0.33, 0, "para Cl", va="center", rotation=90, fontsize=12)
    ax.text(
        0.5,
        -0.32,
        f"ΔΔpIC50 = {selected.iloc[0].delta_delta_pIC50:+.3f}\n"
        "Same assay and unchanged substitution at other mapped sites",
        ha="center",
        fontsize=11,
        color="#425564",
    )
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "drd2-chlorine-quartet.svg")
    plt.close(fig)


def plot_chemical_cycle(results: Path, out: Path) -> None:
    """Four aligned molecular drawings with explicit reported concentrations."""
    molecules = pd.read_csv(results / "drd2/molecules.csv").set_index("molecule_id")
    measurements = pd.read_csv(out / "drd2-selected-measurements.csv")
    cells = [
        ("CHEMBL341497", 0, 0),
        ("CHEMBL126633", 1, 0),
        ("CHEMBL419610", 0, 1),
        ("CHEMBL302075", 1, 1),
    ]
    quartet = pd.read_csv(out / "drd2-chlorine-quartet.csv")
    if set(quartet.molecule_id) != {cell[0] for cell in cells}:
        raise ValueError("Illustrative structures do not match the computed quartet")
    reference = Chem.MolFromSmiles(molecules.loc["CHEMBL341497", "smiles"])
    rdDepictor.Compute2DCoords(reference)
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1120 820" '
        'width="1120" height="820" role="img" '
        'aria-label="Four stereochemically matched DRD2 analogues with para and ortho chlorine">',
        '<rect width="1120" height="820" fill="white"/>',
        '<g font-family="sans-serif" fill="#253747">',
        '<text x="320" y="35" text-anchor="middle" font-size="23">ortho H</text>',
        '<text x="880" y="35" text-anchor="middle" font-size="23">ortho Cl</text>',
        '<text x="28" y="192" transform="rotate(-90 28 192)" '
        'text-anchor="middle" font-size="23">para H</text>',
        '<text x="28" y="560" transform="rotate(-90 28 560)" '
        'text-anchor="middle" font-size="23">para Cl</text>',
        "</g>",
    ]
    for identifier, column, row in cells:
        recorded = measurements[measurements.molecule_id == identifier]
        concentration = (
            " / ".join(f"{value:g}" for value in sorted(recorded.standard_value)) + " nM"
        )
        molecule = Chem.MolFromSmiles(molecules.loc[identifier, "smiles"])
        rdDepictor.GenerateDepictionMatching2DStructure(molecule, reference)
        drawer = rdMolDraw2D.MolDraw2DSVG(430, 250)
        drawer.drawOptions().padding = 0.08
        drawer.DrawMolecule(molecule)
        drawer.FinishDrawing()
        svg = re.sub(r"^.*?<svg\b[^>]*>", "", drawer.GetDrawingText(), flags=re.S)
        svg = re.sub(r"</svg>\s*$", "", svg)
        x, y = 70 + column * 560, 55 + row * 370
        value = molecules.loc[identifier, "pactivity"]
        parts.extend(
            [
                f'<g transform="translate({x},{y})">{svg}</g>',
                f'<text x="{x + 215}" y="{y + 278}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="19" fill="#253747">{identifier}</text>',
                f'<text x="{x + 215}" y="{y + 306}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="19" fill="#253747">'
                f"pIC50 {value:.3f} · {html.escape(concentration)}</text>",
            ]
        )
    potency = {cell[0]: float(molecules.loc[cell[0], "pactivity"]) for cell in cells}
    for x, y, delta in [
        (500, 188, potency["CHEMBL126633"] - potency["CHEMBL341497"]),
        (500, 558, potency["CHEMBL302075"] - potency["CHEMBL419610"]),
    ]:
        parts.extend(
            [
                f'<path d="M {x} {y} h105 m-12 -8 12 8-12 8" fill="none" '
                'stroke="#278d81" stroke-width="3"/>',
                f'<text x="{x + 48}" y="{y - 14}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="20" fill="#278d81">{delta:+.3f}</text>',
            ]
        )
    for x, y, delta in [
        (290, 380, potency["CHEMBL419610"] - potency["CHEMBL341497"]),
        (850, 380, potency["CHEMBL302075"] - potency["CHEMBL126633"]),
    ]:
        parts.extend(
            [
                f'<path d="M {x} {y - 13} v45 m-8 -12 8 12 8-12" fill="none" '
                'stroke="#a65674" stroke-width="3"/>',
                f'<text x="{x + 18}" y="{y + 14}" font-family="sans-serif" '
                f'font-size="20" fill="#a65674">{delta:+.3f}</text>',
            ]
        )
    contrast = (
        potency["CHEMBL302075"]
        - potency["CHEMBL419610"]
        - potency["CHEMBL126633"]
        + potency["CHEMBL341497"]
    )
    parts.extend(
        [
            '<text x="560" y="785" text-anchor="middle" font-family="sans-serif" '
            'font-size="21" fill="#425564">Same assay · same encoded (S,S) core · '
            "other mapped sites unchanged · ΔΔpIC50 = "
            f"{contrast:+.3f}</text>",
            "</svg>",
        ]
    )
    (out / "drd2-chlorine-structures.svg").write_text("".join(parts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("paper/results"))
    parser.add_argument("--out", type=Path, default=Path("paper/scientific-results"))
    parser.add_argument("--permutations", type=int, default=5000)
    args = parser.parse_args()
    print(analyse(args.results, args.out, repeats=args.permutations))
    plot_chemical_cycle(args.results, args.out)

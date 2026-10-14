"""Offline publication evidence: frozen examples, numerical checks and review tasks.

This never trains a model, docks, fetches data, or declares expert validation.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from importlib import metadata
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sarscope.analysis.rgroups import ScaffoldSar, decompose_scaffold
from sarscope.analysis.sar_map import build_sar_map
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.benchmark import TARGETS, load_frozen
from sarscope.curate import curate_chembl
from sarscope.params import CurationParams, RunParams
from sarscope.provenance import file_sha256, package_versions
from sarscope.sar_depict import sar_map_svg

DECISIONS = {"increase", "decrease", "mixed", "insufficient"}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Refusing to overwrite non-empty output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def freeze_cases(frozen: Path, protocol: Path, out: Path) -> Path:
    """Extract complete assay cohorts, not handpicked successful transformations."""
    manifest, snapshots = load_frozen(frozen)
    specification = json.loads(protocol.read_text())
    if specification.get("schema_version") != 1:
        raise ValueError("Unsupported case protocol")
    cases = specification["cases"]
    names = [case["case_id"] for case in cases]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[a-z0-9_-]+", n) for n in names)
    ):
        raise ValueError("Case IDs must be unique, safe directory names")
    datasets = {entry["target_id"]: rows for entry, rows in snapshots}
    cohorts = []
    for case in cases:
        records = [
            r for r in datasets[case["target_id"]] if r["assay_chembl_id"] == case["assay_id"]
        ]
        if not records or any(r.get("document_chembl_id") != case["document_id"] for r in records):
            raise ValueError(f"Missing assay or mismatched document for {case['case_id']}")
        cohorts.append((case, records))
    _new_directory(out)
    entries = []
    for case, records in cohorts:
        name = f"{case['case_id']}.json"
        _write_json(out / name, records)
        entries.append(
            {**case, "file": name, "sha256": file_sha256(out / name), "records": len(records)}
        )
    _write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "chembl_release": manifest["chembl_release"],
            "parent_manifest_sha256": file_sha256(frozen / "manifest.json"),
            "protocol_sha256": file_sha256(protocol),
            "protocol": specification,
            "license": "CC-BY-SA-3.0",
            "attribution": "ChEMBL, EMBL-EBI; original publications identified in each case",
            "cases": entries,
        },
    )
    return out / "manifest.json"


def load_cases(
    path: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported case manifest")
    result = []
    for case in manifest["cases"]:
        if (
            not re.fullmatch(r"[a-z0-9_-]+", case["case_id"])
            or case["file"] != f"{case['case_id']}.json"
        ):
            raise ValueError("Invalid case filename")
        source = path / case["file"]
        if file_sha256(source) != case["sha256"]:
            raise ValueError(f"Case snapshot hash mismatch: {case['case_id']}")
        records = json.loads(source.read_text())
        if len(records) != case["records"] or any(
            r.get("target_chembl_id") != case["target_id"]
            or r.get("assay_chembl_id") != case["assay_id"]
            or r.get("document_chembl_id") != case["document_id"]
            for r in records
        ):
            raise ValueError("Case identity or record count mismatch")
        result.append((case, records))
    return manifest, result


def direct_position_pairs(series: ScaffoldSar, evidence: pd.DataFrame) -> pd.DataFrame:
    """Small-cohort exhaustive reference, independent of SAR-map pair grouping.

    Restricted to ONE assay, ONE endpoint, UNANNOTATED variants. This checks
    the RDKit-derived decomposition numerically; it is not another SAR tool.
    """
    if evidence.assay_chembl_id.nunique() != 1 or evidence.standard_type.nunique() != 1:
        raise ValueError("Reference enumeration requires one assay and endpoint")
    if evidence.assay_variant_mutation.notna().any():
        raise ValueError("Reference enumeration requires unannotated variants")
    values = evidence.groupby("molecule_id").pactivity.median()
    result = []
    for left, right in combinations(series.members.to_dict("records"), 2):
        differing = [p for p in series.positions if left[p] != right[p]]
        if len(differing) != 1:
            continue
        position = differing[0]
        if left[position] > right[position]:
            left, right = right, left
        a, b = left["molecule_id"], right["molecule_id"]
        result.append(
            {
                "position": position,
                "id_from": a,
                "id_to": b,
                "delta_pactivity": float(values[b] - values[a]),
            }
        )
    return pd.DataFrame(result, columns=["position", "id_from", "id_to", "delta_pactivity"])


def evaluate_cases(cases: Path, out: Path) -> Path:
    manifest, cohorts = load_cases(cases)
    _new_directory(out)
    params = RunParams(curation=CurationParams(source_ids=(1,), variant=None))
    overview: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    keys: list[dict[str, Any]] = []
    review_pairs: list[dict[str, Any]] = []
    for case, records in cohorts:
        start = time.perf_counter()
        curated = curate_chembl(records, params)
        table = add_scaffolds(curated.table)
        counts = table.murcko.dropna().value_counts().rename_axis("scaffold").reset_index(name="n")
        counts = counts.sort_values(["n", "scaffold"], ascending=[False, True])
        if counts.empty:
            raise ValueError(f"No scaffold in case {case['case_id']}")
        series = decompose_scaffold(table, counts.iloc[0].scaffold, min_substituent_count=1)
        if series is None:
            raise ValueError(f"Decomposition failed for case {case['case_id']}")
        evidence = curated.evidence[curated.evidence.molecule_id.isin(series.members.molecule_id)]
        result = build_sar_map(
            series, evidence, effect_threshold=manifest["protocol"]["effect_threshold"]
        )
        # Audit the same stereochemically scoped cohort used by the map.
        series = result.series
        evidence = evidence[evidence.molecule_id.isin(series.members.molecule_id)]
        reference = direct_position_pairs(series, evidence)
        columns = ["position", "id_from", "id_to"]
        observed = (
            result.pairs[columns + ["delta_pactivity"]].sort_values(columns).reset_index(drop=True)
        )
        expected = reference.sort_values(columns).reset_index(drop=True)
        agrees = observed[columns].equals(expected[columns]) and np.allclose(
            observed.delta_pactivity, expected.delta_pactivity, atol=1e-10, rtol=0
        )
        if not agrees:
            raise ValueError(f"Direct enumeration disagrees with SAR map: {case['case_id']}")
        directory = out / case["case_id"]
        directory.mkdir()
        for name, frame in (
            ("molecules", table),
            ("measurements", curated.evidence),
            ("scaffold_counts", counts),
            ("series_members", series.members),
            ("pairs", result.pairs),
            ("transformations", result.transformations),
            ("reference_pairs", reference),
        ):
            frame.to_csv(directory / f"{name}.csv", index=False)
        svg = sar_map_svg(result)
        if svg:
            (directory / "sar-map.svg").write_text(svg)
        _write_json(
            directory / "scope.json",
            {
                **case,
                "core": series.core,
                "scaffold": series.scaffold,
                "curation": asdict(params.curation),
                "curation_steps": [asdict(s) for s in curated.steps],
                "grade_definition": "descriptive heuristic, not calibrated confidence",
                "pair_limit_reached": result.pair_limit_reached,
                "excluded_core_molecules": result.excluded_core_molecules,
            },
        )
        overview.append(
            {
                "case_id": case["case_id"],
                "target_id": case["target_id"],
                "assay_id": case["assay_id"],
                "raw_records": len(records),
                "retained_measurements": len(curated.evidence),
                "curated_molecules": len(table),
                "scaffolds": len(counts),
                "series_molecules": series.n_molecules,
                "unmatched": series.n_unmatched,
                "positions": len(series.positions),
                "pair_rows": len(result.pairs),
                "transformations": len(result.transformations),
                "reference_agreement": agrees,
                "pair_limit_reached": result.pair_limit_reached,
                "elapsed_seconds": time.perf_counter() - start,
            }
        )
        # Deterministic grade-stratified sample. Show every stratum, including
        # mixed/limited and empty cases; never present it as a prevalence sample.
        for grade in ("strong", "moderate", "mixed", "limited"):
            candidates = result.transformations.query("evidence_strength == @grade")
            for _, row in candidates.head(5).iterrows():
                identifier = f"T{len(tasks) + 1:03d}"
                block = result.pairs[
                    (result.pairs.position == row.position)
                    & (result.pairs.fragment_from == row.fragment_from)
                    & (result.pairs.fragment_to == row.fragment_to)
                    & (result.pairs.endpoint == row.endpoint)
                    & (result.pairs.variant == row.variant)
                    & (result.pairs.evidence_tier == row.evidence_tier)
                ]
                tasks.append(
                    {
                        "task_id": identifier,
                        "case_id": case["case_id"],
                        "doi": case["doi"],
                        "core": series.core,
                        "position": row.position,
                        "fragment_from": row.fragment_from,
                        "fragment_to": row.fragment_to,
                        "effect_threshold": result.effect_threshold,
                    }
                )
                keys.append(
                    {
                        "task_id": identifier,
                        "grade": grade,
                        "delta": row.median_delta,
                        "tool_decision": row.interpretation,
                    }
                )
                for pair in block.to_dict("records"):
                    review_pairs.append({"task_id": identifier, **pair})
    pd.DataFrame(overview).to_csv(out / "case-summary.csv", index=False)
    pd.DataFrame(tasks).to_csv(out / "review-tasks.csv", index=False)
    pd.DataFrame(keys).to_csv(out / "review-answer-key.csv", index=False)
    pd.DataFrame(review_pairs).to_csv(out / "review-pairs.csv", index=False)
    pd.DataFrame(
        [
            {
                "task_id": task["task_id"],
                "reviewer_id": "",
                "decision": "",
                "source_issue": "",
                "rationale": "",
                "literature_table_or_figure": "",
            }
            for task in tasks
        ]
    ).to_csv(out / "review-template.csv", index=False)
    _write_json(
        out / "evaluation.json",
        {
            "case_manifest_sha256": file_sha256(cases / "manifest.json"),
            "python": platform.python_version(),
            "packages": package_versions(),
            "scope": "Descriptive published-data examples plus exhaustive numerical verification",
            "expert_validation": "not performed",
            "literature_trend_recovery": "not evaluated",
            "user_workflow_comparison": "not performed",
            "timing": "one local run, not a speed comparison",
        },
    )
    return out / "case-summary.csv"


def benchmark_audit(predictions: Path, out: Path, *, repeats: int = 1000, seed: int = 42) -> Path:
    """Scaffold-cluster bootstrap of SAVED predictions, never retraining models."""
    frame = pd.read_csv(predictions)
    required = {
        "target_id",
        "molecule_id",
        "murcko",
        "pactivity",
        "predicted_pactivity",
        "baseline_pactivity",
    }
    if not required <= set(frame) or repeats < 100:
        raise ValueError("Need saved predictions with scaffold IDs and >=100 bootstrap replicates")
    if set(frame.target_id) != {t[0] for t in TARGETS}:
        raise ValueError("Saved predictions must contain the fixed three-target panel")
    if (
        frame.duplicated(["target_id", "molecule_id"]).any()
        or not np.isfinite(
            frame[["pactivity", "predicted_pactivity", "baseline_pactivity"]].to_numpy()
        ).all()
    ):
        raise ValueError("Duplicate or non-finite held-out predictions")
    _new_directory(out)
    rng = np.random.default_rng(seed)
    rows = []
    for target, block in frame.groupby("target_id", sort=True):
        y = block.pactivity.to_numpy()
        err = (block.predicted_pactivity.to_numpy() - y) ** 2
        base = (block.baseline_pactivity.to_numpy() - y) ** 2
        groups = block.murcko.fillna("").astype(str)
        groups = groups.mask(groups.eq(""), "acyclic:" + block.molecule_id.astype(str))
        codes, labels = pd.factorize(groups, sort=True)
        counts = np.bincount(codes)
        errors = np.bincount(codes, weights=err)
        baselines = np.bincount(codes, weights=base)
        sampled = rng.multinomial(len(labels), np.full(len(labels), 1 / len(labels)), size=repeats)
        model_rmse = np.sqrt((sampled @ errors) / (sampled @ counts))
        base_rmse = np.sqrt((sampled @ baselines) / (sampled @ counts))
        denominator = ((y - y.mean()) ** 2).sum()
        rows.append(
            {
                "target_id": target,
                "n_test": len(block),
                "scaffold_clusters": len(labels),
                "r2": 1 - err.sum() / denominator if denominator else None,
                "rmse": np.sqrt(err.mean()),
                "spearman": spearmanr(y, block.predicted_pactivity).statistic,
                "baseline_rmse": np.sqrt(base.mean()),
                "rmse_ci_low": np.quantile(model_rmse, 0.025),
                "rmse_ci_high": np.quantile(model_rmse, 0.975),
                "baseline_minus_model_ci_low": np.quantile(base_rmse - model_rmse, 0.025),
                "baseline_minus_model_ci_high": np.quantile(base_rmse - model_rmse, 0.975),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "bootstrap-summary.csv", index=False)
    _write_json(
        out / "scope.json",
        {
            "predictions_sha256": file_sha256(predictions),
            "bootstrap_repeats": repeats,
            "seed": seed,
            "scope": (
                "Conditional held-out sampling uncertainty for one fitted model per target; "
                "not repeated training, external validation or prospective uncertainty"
            ),
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 3.8))
    x = np.arange(len(metrics))
    ax.bar(x - 0.18, metrics.rmse, 0.36, label="Random forest", color="#158f83")
    ax.bar(x + 0.18, metrics.baseline_rmse, 0.36, label="Training-median baseline", color="#d89740")
    # Percentile intervals need not contain a point estimate; draw absolute
    # bounds rather than passing potentially negative errors to matplotlib.
    ax.vlines(x - 0.18, metrics.rmse_ci_low, metrics.rmse_ci_high, color="#253747")
    labels = {"CHEMBL5145": "BRAF", "CHEMBL217": "DRD2", "CHEMBL206": "ER alpha"}
    ax.set_xticks(x, [labels[t] for t in metrics.target_id])
    ax.set_ylabel("Held-out RMSE (log10 molar units)")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "benchmark.svg")
    plt.close(fig)
    return out / "bootstrap-summary.csv"


def score_reviews(key: Path, reviews: Path, out: Path) -> Path:
    answers = pd.read_csv(key)
    decisions = pd.read_csv(reviews, keep_default_na=False, dtype=str)
    if not {"task_id", "reviewer_id", "decision"} <= set(decisions):
        raise ValueError("Review CSV needs task_id, reviewer_id and decision")
    if answers.empty or answers.task_id.duplicated().any():
        raise ValueError("Answer key is empty or has duplicate tasks")
    if (
        decisions.empty
        or (decisions.reviewer_id.str.strip() == "").any()
        or not set(decisions.decision) <= DECISIONS
    ):
        raise ValueError("Reviews are incomplete; never count blank template rows as validation")
    if decisions.duplicated(["task_id", "reviewer_id"]).any() or not set(decisions.task_id) <= set(
        answers.task_id
    ):
        raise ValueError("Unknown task or duplicate reviewer decision")
    rows = []
    for task in answers.itertuples():
        block = decisions[decisions.task_id == task.task_id]
        consensus = block.decision.nunique() == 1 and len(block) >= 2
        expert = block.decision.iloc[0] if consensus else "unresolved"
        tool = "mixed" if task.tool_decision == "mixed / small change" else task.tool_decision
        rows.append(
            {
                "task_id": task.task_id,
                "grade": task.grade,
                "reviewers": len(block),
                "consensus": expert,
                "tool_decision": tool,
                "agreement": expert == tool if consensus else None,
                "unsupported_directional_claim": expert != tool
                if consensus and task.grade in {"strong", "moderate"}
                else None,
            }
        )
    _new_directory(out)
    pd.DataFrame(rows).to_csv(out / "review-agreement.csv", index=False)
    _write_json(
        out / "scope.json",
        {
            "key_sha256": file_sha256(key),
            "reviews_sha256": file_sha256(reviews),
            "tasks": len(answers),
            "consensus_tasks": sum(r["consensus"] != "unresolved" for r in rows),
            "scope": (
                "Grade-stratified descriptive expert agreement; unresolved/missing tasks excluded, "
                "not a calibrated probability or prevalence estimate"
            ),
        },
    )
    return out / "review-agreement.csv"


def compare_mmpdb(results: Path, out: Path) -> Path:
    """Actual optional comparator run on identical curated single-assay inputs.

    Compare unordered compound-pair membership, not fragment numbering or UI
    task times. Hydrogen treatment, stereo and context choices can differ.
    """
    from sarscope.analysis.mmp import matched_molecular_pairs

    try:
        version = metadata.version("mmpdb")
    except metadata.PackageNotFoundError as exc:
        raise ValueError("Install mmpdb in a separate evaluation environment first") from exc
    _new_directory(out)
    summaries = []
    for directory in sorted(results.iterdir()):
        if not directory.is_dir() or not (directory / "molecules.csv").is_file():
            continue
        table = pd.read_csv(directory / "molecules.csv")
        case = out / directory.name
        case.mkdir()
        smiles = case / "input.smi"
        smiles.write_text("".join(f"{r.smiles}\t{r.molecule_id}\n" for r in table.itertuples()))
        fragments, pairs = case / "fragments.fragdb", case / "mmpdb-pairs.tsv"
        commands = [
            [
                sys.executable,
                "-m",
                "mmpdblib",
                "fragment",
                str(smiles),
                "--num-cuts",
                "1",
                "--cut-smarts",
                "[#6+0;!$(*=,#[!#6])]!@!=!#[!#0;!#1]",
                "--salt-remover",
                "<none>",
                "--max-heavies",
                "none",
                "--max-rotatable-bonds",
                "999",
                "-j",
                "1",
                "-o",
                str(fragments),
            ],
            [
                sys.executable,
                "-m",
                "mmpdblib",
                "index",
                str(fragments),
                "--max-variable-heavies",
                "10",
                "--max-variable-ratio",
                "0.5",
                "--out",
                "csv",
                "-o",
                str(pairs),
            ],
        ]
        logs = []
        start = time.perf_counter()
        for command in commands:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=120, check=False
            )
            logs.append(completed.stdout + completed.stderr)
            if completed.returncode:
                (case / "execution.log").write_text("\n".join(logs))
                raise ValueError(f"mmpdb failed in {directory.name}; see execution.log")
        elapsed = time.perf_counter() - start
        (case / "execution.log").write_text("\n".join(logs))
        _write_json(case / "commands.json", commands)
        baseline = (
            pd.read_csv(pairs, sep="\t", header=None) if pairs.stat().st_size else pd.DataFrame()
        )
        start = time.perf_counter()
        own = matched_molecular_pairs(table)
        own_elapsed = time.perf_counter() - start
        own.to_csv(case / "sarscope-pairs.csv", index=False)
        own_keys = {tuple(sorted((str(r.id_a), str(r.id_b)))) for r in own.itertuples()}
        other_keys = (
            {tuple(sorted((str(a), str(b)))) for a, b in zip(baseline[2], baseline[3], strict=True)}
            if not baseline.empty
            else set()
        )
        pd.DataFrame(
            [
                {
                    "id_a": a,
                    "id_b": b,
                    "sarscope": (a, b) in own_keys,
                    "mmpdb": (a, b) in other_keys,
                }
                for a, b in sorted(own_keys | other_keys)
            ]
        ).to_csv(case / "pair-membership.csv", index=False)
        summaries.append(
            {
                "case_id": directory.name,
                "molecules": len(table),
                "sarscope_pair_rows": len(own),
                "mmpdb_pair_rows": len(baseline),
                "sarscope_unique_pairs": len(own_keys),
                "mmpdb_unique_pairs": len(other_keys),
                "shared_unique_pairs": len(own_keys & other_keys),
                "sarscope_only": len(own_keys - other_keys),
                "mmpdb_only": len(other_keys - own_keys),
                "mmpdb_process_seconds": elapsed,
                "sarscope_function_seconds": own_elapsed,
                "input_sha256": file_sha256(smiles),
            }
        )
    if not summaries:
        raise ValueError("No evaluated case cohorts found")
    pd.DataFrame(summaries).to_csv(out / "comparison.csv", index=False)
    _write_json(
        out / "scope.json",
        {
            "mmpdb_version": version,
            "packages": package_versions(),
            "scope": (
                "Single-cut compound-pair coverage on identical curated inputs, not superiority"
            ),
            "timing": (
                "mmpdb subprocess startup plus disk IO vs an in-process function; "
                "NOT a fair speed comparison"
            ),
            "known_differences": (
                "Hydrogen substitutions, stereochemical enumeration and constant-fragment "
                "orientation; inspect mismatches before accuracy claims"
            ),
        },
    )
    return out / "comparison.csv"


def bundle_repository(root: Path, out: Path, *, benchmark: Path | None = None) -> Path:
    """Archive nonignored source/evidence without venvs, secrets or run caches."""
    root, out = root.resolve(), out.resolve()
    if out.exists() or out.is_relative_to(root):
        raise ValueError("Use a new archive path outside the repository")
    if benchmark is not None:
        load_frozen(benchmark / "frozen")
    files = (
        subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    entries = []
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(set(filter(None, files))):
            path = root / name
            if not path.is_file() or path.is_symlink():
                continue
            archive.write(path, f"sarscope/{name}")
            entries.append({"file": name, "sha256": file_sha256(path)})
        if benchmark is not None:
            for path in sorted(benchmark.rglob("*")):
                if path.is_file() and not path.is_symlink() and path.suffix in {".json", ".csv"}:
                    name = f"reproducibility/benchmark/{path.relative_to(benchmark)}"
                    archive.write(path, f"sarscope/{name}")
                    entries.append({"file": name, "sha256": file_sha256(path)})
        archive.writestr(
            "sarscope/ARCHIVE-MANIFEST.json",
            json.dumps(
                {
                    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root)
                    .decode()
                    .strip(),
                    "includes_uncommitted_files": bool(
                        subprocess.check_output(["git", "status", "--porcelain"], cwd=root)
                    ),
                    "files": entries,
                    "status": "local archive, not a published release or DOI deposit",
                },
                indent=2,
            ),
        )
    return out


def lock_environment(report: Path, out: Path) -> Path:
    """Turn a pip --report resolution into a platform-specific hashed lock.

    Resolve with --ignore-installed so transitive requirements are included.
    No private index URL or credential from the report is copied to the lock.
    """
    if out.exists():
        raise ValueError("Refusing to overwrite an existing environment lock")
    resolution = json.loads(report.read_text())
    packages = []
    for item in resolution.get("install", []):
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        digest = item["download_info"]["archive_info"].get("hashes", {}).get("sha256", "")
        if (
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+_!-]*", version)
            or not re.fullmatch(r"[a-f0-9]{64}", digest)
        ):
            raise ValueError("Resolution needs valid names/versions and SHA-256 archive hashes")
        packages.append(f"{name}=={version} --hash=sha256:{digest}")
    if not packages:
        raise ValueError("Empty pip resolution; use --ignore-installed")
    environment = resolution["environment"]
    header = (
        "# Platform-specific pip resolution, not a universal lock.\n"
        f"# Python {environment['python_full_version']}; {environment['sys_platform']}; "
        f"{environment['platform_machine']}\n"
        f"# Source pip report SHA-256: {file_sha256(report)}\n"
        "# Install SARscope source separately with --no-deps.\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header + "\n".join(sorted(packages, key=str.lower)) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze-cases")
    freeze.add_argument("--frozen", type=Path, required=True)
    freeze.add_argument("--protocol", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--cases", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    audit = sub.add_parser("benchmark-audit")
    audit.add_argument("--predictions", type=Path, required=True)
    audit.add_argument("--out", type=Path, required=True)
    audit.add_argument("--repeats", type=int, default=1000)
    review = sub.add_parser("score-reviews")
    review.add_argument("--key", type=Path, required=True)
    review.add_argument("--reviews", type=Path, required=True)
    review.add_argument("--out", type=Path, required=True)
    compare = sub.add_parser("compare-mmpdb")
    compare.add_argument("--results", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    bundle = sub.add_parser("bundle")
    bundle.add_argument("--root", type=Path, default=Path.cwd())
    bundle.add_argument("--out", type=Path, required=True)
    bundle.add_argument(
        "--benchmark", type=Path, help="include verified full frozen benchmark and saved results"
    )
    lock = sub.add_parser("lock-environment")
    lock.add_argument("--report", type=Path, required=True)
    lock.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "freeze-cases":
            path = freeze_cases(args.frozen, args.protocol, args.out)
        elif args.command == "evaluate":
            path = evaluate_cases(args.cases, args.out)
        elif args.command == "benchmark-audit":
            path = benchmark_audit(args.predictions, args.out, repeats=args.repeats)
        elif args.command == "score-reviews":
            path = score_reviews(args.key, args.reviews, args.out)
        elif args.command == "compare-mmpdb":
            path = compare_mmpdb(args.results, args.out)
        elif args.command == "lock-environment":
            path = lock_environment(args.report, args.out)
        else:
            path = bundle_repository(args.root, args.out, benchmark=args.benchmark)
    except (ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

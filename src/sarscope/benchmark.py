"""Predeclared, frozen ChEMBL multi-family pIC50 regression benchmark.

The acquisition step is separate from evaluation. The runner accepts only
hashed local snapshots and never fetches or silently refreshes target data.
This is a within-target, scaffold-held-out benchmark across target families,
not a claim of cross-target transfer or prospective validation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sarscope.analysis.features import fingerprint_matrix
from sarscope.analysis.regression import evaluate_regression, regression_metrics
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.curate import curate_chembl, source_safe_table
from sarscope.params import CurationParams, ModelParams, RunParams
from sarscope.provenance import package_versions
from sarscope.sources.chembl import ChemblClient

BENCHMARK_VERSION = 1
TARGETS = (
    ("CHEMBL5145", "protein_kinase", "Serine/threonine-protein kinase B-raf"),
    ("CHEMBL217", "class_a_gpcr", "D(2) dopamine receptor"),
    ("CHEMBL206", "nuclear_receptor", "Estrogen receptor"),
)


@dataclass(frozen=True)
class BenchmarkProtocol:
    endpoint: str = "IC50"
    units: str = "nM"
    assay_type: str = "B"
    variant: str = "wild_type_or_unannotated"
    aggregation: str = "median"
    fingerprint: str = "ecfp4"
    algorithm: str = "random_forest"
    split: str = "scaffold"
    cv_folds: int = 3
    test_fraction: float = 0.2
    seed: int = 42


PROTOCOL = BenchmarkProtocol()
ORIGIN_HOLDOUT_ID = 37  # BindingDB records integrated in ChEMBL


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def freeze_benchmark(out: Path, client: ChemblClient) -> Path:
    """Download the fixed target panel once; refuse replacement of existing data."""
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"freeze destination is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    release = client.release
    for target_id, family, expected_name in TARGETS:
        target = client.target(target_id)
        if (
            target.get("target_chembl_id") != target_id
            or target.get("organism") != "Homo sapiens"
            or target.get("target_type") != "SINGLE PROTEIN"
            or target.get("pref_name") != expected_name
        ):
            raise ValueError(
                f"target identity changed for {target_id}; review panel before freezing"
            )
        records = client.activities(target_id, (PROTOCOL.endpoint,))
        data = _json_bytes(records)
        filename = f"{target_id}.activities.json"
        (out / filename).write_bytes(data)
        entries.append(
            {
                "target_id": target_id,
                "target_name": expected_name,
                "family": family,
                "file": filename,
                "sha256": _sha256(data),
                "raw_activity_count": len(records),
            }
        )
    manifest = {
        "schema_version": BENCHMARK_VERSION,
        "chembl_release": release,
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol": asdict(PROTOCOL),
        "targets": entries,
        "scope": "within-target scaffold-held-out pIC50 regression across three protein families",
        "packages_at_freeze": package_versions(),
    }
    path = out / "manifest.json"
    path.write_bytes(_json_bytes(manifest))
    return path


def load_frozen(
    out: Path,
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[dict[str, Any]]]]]:
    """Validate panel, protocol, paths and hashes before opening any model data."""
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest.get("schema_version") != BENCHMARK_VERSION:
        raise ValueError("unsupported benchmark manifest version")
    if not isinstance(manifest.get("chembl_release"), str) or not manifest["chembl_release"]:
        raise ValueError("benchmark manifest is missing the ChEMBL release")
    if manifest.get("protocol") != asdict(PROTOCOL):
        raise ValueError("benchmark protocol differs from the predeclared version")
    entries = manifest.get("targets", [])
    expected = [(tid, family, name) for tid, family, name in TARGETS]
    actual = [(e.get("target_id"), e.get("family"), e.get("target_name")) for e in entries]
    if actual != expected:
        raise ValueError("benchmark panel changed or target families are not independent")
    frozen = []
    for entry in entries:
        filename = entry.get("file", "")
        if filename != f"{entry['target_id']}.activities.json":
            raise ValueError("invalid snapshot filename")
        data = (out / filename).read_bytes()
        if _sha256(data) != entry.get("sha256"):
            raise ValueError(f"snapshot hash mismatch: {filename}")
        records = json.loads(data)
        if not isinstance(records, list) or len(records) != entry.get("raw_activity_count"):
            raise ValueError(f"invalid or incomplete activity snapshot: {filename}")
        if any(r.get("target_chembl_id") != entry["target_id"] for r in records):
            raise ValueError(f"snapshot contains another target: {filename}")
        frozen.append((entry, records))
    return manifest, frozen


def run_benchmark(frozen_dir: Path, out: Path, *, validation: str = "scaffold") -> Path:
    if validation not in ("scaffold", "origin"):
        raise ValueError("validation must be scaffold or origin")
    manifest, frozen = load_frozen(frozen_dir)
    if out.resolve().is_relative_to(frozen_dir.resolve()):
        raise ValueError("benchmark results must not overwrite frozen inputs")
    params = RunParams(
        curation=CurationParams(
            standard_types=(PROTOCOL.endpoint,),
            units=(PROTOCOL.units,),
            assay_types=(PROTOCOL.assay_type,),
            variant=None,
            aggregate="median",
        ),
        model=ModelParams(
            regression_algorithms=(PROTOCOL.algorithm,),
            split="source" if validation == "origin" else "scaffold",
            source_test_id=ORIGIN_HOLDOUT_ID if validation == "origin" else None,
            cv_folds=PROTOCOL.cv_folds,
            test_fraction=PROTOCOL.test_fraction,
            leakage_audit=False,
            seed=PROTOCOL.seed,
        ),
    )
    summaries = []
    held_out = []
    for entry, records in frozen:
        print(f"Benchmarking {entry['target_id']} ({entry['family']}, {validation})…", flush=True)
        curated = curate_chembl(records, params)
        table = add_scaffolds(
            source_safe_table(curated, params) if validation == "origin" else curated.table
        )
        if len(table) < 30 or table["murcko"].nunique() < 5:
            raise ValueError(f"{entry['target_id']} lacks enough curated molecules/scaffolds")
        X = fingerprint_matrix(table["smiles"].tolist(), "ecfp4")
        fitted = evaluate_regression(
            X,
            table["pactivity"].to_numpy(dtype=float),
            table["murcko"].tolist(),
            table["activity_class"].tolist(),
            params.model,
            source_test=table["source_test"].tolist() if validation == "origin" else None,
        )
        baseline = np.repeat(
            float(table.iloc[fitted.train_index]["pactivity"].median()), len(fitted.test_index)
        )
        baseline_r2, baseline_rmse, baseline_spearman = regression_metrics(
            fitted.test_truth, baseline
        )
        score = fitted.scores.iloc[0]
        train_scaffolds = set(table.iloc[fitted.train_index]["murcko"].dropna())
        test_scaffolds = set(table.iloc[fitted.test_index]["murcko"].dropna())
        train_rows = table.iloc[fitted.train_index]
        test_rows = table.iloc[fitted.test_index]
        train_keys = set(train_rows["inchikey"].fillna(train_rows["smiles"]))
        test_keys = set(test_rows["inchikey"].fillna(test_rows["smiles"]))
        if train_keys & test_keys:
            raise ValueError(f"structure leakage detected for {entry['target_id']}")
        shared_scaffolds = len(train_scaffolds & test_scaffolds)
        if validation == "scaffold" and shared_scaffolds:
            raise ValueError(f"scaffold leakage detected for {entry['target_id']}")
        summaries.append(
            {
                "target_id": entry["target_id"],
                "family": entry["family"],
                "validation": validation,
                "held_out_origin": ORIGIN_HOLDOUT_ID if validation == "origin" else None,
                "chembl_release": manifest["chembl_release"],
                "raw_activities": entry["raw_activity_count"],
                "curated_compounds": len(table),
                "n_train": len(fitted.train_index),
                "n_test": len(fitted.test_index),
                "n_train_scaffolds": len(train_scaffolds),
                "n_test_scaffolds": len(test_scaffolds),
                "scaffold_overlap": shared_scaffolds,
                "structure_overlap": 0,
                "cv_rmse": score["cv_rmse"],
                "test_r2": score["test_r2"],
                "test_rmse": score["test_rmse"],
                "test_spearman": score["test_spearman"],
                "median_baseline_r2": baseline_r2,
                "median_baseline_rmse": baseline_rmse,
                "median_baseline_spearman": baseline_spearman,
                "n_features": score["n_features"],
            }
        )
        predictions = table.iloc[fitted.test_index][
            ["molecule_id", "smiles", "inchikey", "murcko", "pactivity"]
        ].copy()
        predictions["target_id"] = entry["target_id"]
        predictions["validation"] = validation
        predictions["predicted_pactivity"] = fitted.test_predictions
        predictions["baseline_pactivity"] = baseline
        held_out.append(predictions)
        print(
            f"  {len(table)} curated, {len(fitted.test_index)} held out, "
            f"test RMSE {score['test_rmse']:.3f}",
            flush=True,
        )
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(out / "target_metrics.csv", index=False)
    pd.concat(held_out, ignore_index=True).to_csv(out / "held_out_predictions.csv", index=False)
    evaluated_protocol = asdict(PROTOCOL)
    evaluated_protocol["split"] = "source" if validation == "origin" else "scaffold"
    evaluated_protocol["source_test_id"] = ORIGIN_HOLDOUT_ID if validation == "origin" else None
    result = {
        "manifest_sha256": _sha256((frozen_dir / "manifest.json").read_bytes()),
        "protocol": evaluated_protocol,
        "validation": validation,
        "held_out_origin": ORIGIN_HOLDOUT_ID if validation == "origin" else None,
        "packages_at_evaluation": package_versions(),
        "target_count": len(summaries),
        "mean_test_r2": float(np.mean([row["test_r2"] for row in summaries])),
        "mean_test_rmse": float(np.mean([row["test_rmse"] for row in summaries])),
        "mean_test_spearman": float(np.mean([row["test_spearman"] for row in summaries])),
        "note": "Descriptive three-target panel; no external prospective or cross-target claim.",
    }
    path = out / "summary.json"
    path.write_bytes(_json_bytes(result))
    return path

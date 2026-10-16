"""Frozen native multi-family evaluation; reuse SARscope's actual app pipeline.

Haga et al. (2025), doi:10.3390/ph18030440 supplies pre-aligned source data.
This is NOT the historical reproduction profile. No external QSAR program,
joint train/test scaling or test-label correction participates in prediction.
Failures are evidence, not a reason to change the frozen cohort or settings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.diagnostics import bootstrap_regression
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_qsar import FieldDataset, prepare_field_series, run_field_qsar
from sarscope.field_reproduction import dataset_from_sources
from sarscope.params import FieldQsarParams, ModelParams

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks/field_qsar/native-validation-protocol.json"
)


def audit_partition(dataset: FieldDataset, n_train: int) -> pd.DataFrame:
    """Refuse graph duplicates before random CV; never delete or relabel a source row."""
    if not 1 < n_train < len(dataset.table):
        raise ValueError("The frozen partition needs training and test rows.")
    rows = []
    for index, mol in enumerate(dataset.molecules):
        graph = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
        rows.append(
            {
                "row": index,
                "molecule_id": dataset.table.iloc[index].molecule_id,
                "split": "train" if index < n_train else "held_out",
                "graph_sha256": hashlib.sha256(graph.encode()).hexdigest(),
            }
        )
    result = pd.DataFrame(rows)
    if result.graph_sha256.duplicated().any():
        raise ValueError(
            "Duplicate encoded structures within/across the frozen split: grouped validation "
            "or source adjudication is needed; no compounds were silently removed."
        )
    if result.molecule_id.duplicated().any():
        raise ValueError(
            "Source identifiers are not unique; adjudicate rather than rename silently."
        )
    return result


def _json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _manifest(out: Path) -> None:
    _json(
        out / "manifest.sha256.json",
        {
            str(path.relative_to(out)): provenance.file_sha256(path)
            for path in sorted(out.rglob("*"))
            if path.is_file() and path.name != "manifest.sha256.json"
        },
    )


def run_campaign(
    protocol: dict[str, Any], payloads: dict[str, dict[str, bytes]], out: Path
) -> dict[str, Any]:
    """All configurations run once; test results never choose a replacement protocol."""
    if protocol.get("schema") != 1:
        raise ValueError("Unknown native field-validation campaign schema.")
    if protocol["model"]["regression_cv"] not in ("default", "loo"):
        raise ValueError("Use a ModelParams regression_cv value: default or loo.")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty output directory; retain previous evidence.")
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "protocol.json", protocol)
    _json(out / "environment.json", provenance.package_versions())
    rows: list[dict[str, Any]] = []
    for source in protocol["datasets"]:
        name = source["name"]
        folder = out / name
        folder.mkdir()
        try:
            dataset, source_audit = dataset_from_sources({"data": source}, payloads[name])
            source_audit.to_csv(folder / "source_audit.csv", index=False)
            n_train = source["train"]["valid"]
            audit_partition(dataset, n_train).to_csv(folder / "partition.csv", index=False)
            prepared = prepare_field_series(dataset, FieldQsarParams(**protocol["fields"]))
            prepared.alignment.report.to_csv(folder / "alignment.csv", index=False)
        except Exception as exc:
            for method in protocol["methods"]:
                rows.append(
                    {
                        "dataset": name,
                        "family": source["family"],
                        "method": method["name"],
                        "completed": False,
                        "stage": "input/preparation",
                        "error": str(exc),
                    }
                )
            _json(folder / "failure.json", {"stage": "input/preparation", "error": str(exc)})
            print(f"FAILED {name} input/preparation: {exc}", flush=True)
            continue
        for method in protocol["methods"]:
            spec = {key: value for key, value in method.items() if key != "name"}
            settings = FieldQsarParams(**{**protocol["fields"], **spec})
            case = folder / method["name"]
            case.mkdir()
            print(f"Running native {name}/{method['name']}…", flush=True)
            start = time.monotonic()
            try:
                with threadpool_limits(limits=1):
                    result = run_field_qsar(
                        replace(prepared, params=settings),
                        ModelParams(**protocol["model"]),
                        split_indices=(
                            list(range(n_train)),
                            list(range(n_train, len(dataset.table))),
                        ),
                    )
                for label, table in (
                    ("predictions", result.predictions),
                    ("statistics", result.statistics),
                    ("checks", result.checks),
                    ("permutations", result.permutations),
                    ("contributions", result.contours.summary),
                    ("component_search", result.regression.fitted_model.selection_table_),
                    ("charges", pd.DataFrame(result.fields.charges)),
                ):
                    table.to_csv(case / f"{label}.csv", index=False)
                test = result.regression.test_index
                groups = add_scaffolds(dataset.table).iloc[test].murcko.tolist()
                bootstrap_regression(
                    result.regression,
                    groups,
                    repeats=protocol["bootstrap_repeats"],
                    seed=settings.seed,
                ).to_csv(case / "bootstrap.csv", index=False)
                _json(case / "provenance.json", result.provenance)
                stats = result.statistics[result.statistics.algorithm == "field_pls"].iloc[0]
                base = result.statistics[result.statistics.algorithm == "random_forest"].iloc[0]
                mean_predictions = result.regression.baseline_test_predictions
                if mean_predictions is None:
                    raise RuntimeError("Training-mean baseline predictions missing.")
                rows.append(
                    {
                        "dataset": name,
                        "family": source["family"],
                        "method": method["name"],
                        "completed": True,
                        "train": n_train,
                        "test": len(test),
                        "q2": float(stats.cv_pooled_q2),
                        "test_r2": float(stats.test_r2),
                        "test_rmse": float(stats.test_rmse),
                        "test_spearman": float(stats.test_spearman),
                        "rf_test_rmse": float(base.test_rmse),
                        "mean_test_rmse": float(
                            np.sqrt(np.mean((result.regression.test_truth - mean_predictions) ** 2))
                        ),
                        "components": int(stats.selected_components),
                        "retained_columns": int(result.regression.fitted_filter.support_.sum()),
                        "permutation_pvalue": float(stats.permutation_pvalue),
                        "model_status": result.status,
                        "seconds": time.monotonic() - start,
                    }
                )
                print(f"Completed {name}/{method['name']}: test R²={stats.test_r2:.4f}", flush=True)
            except Exception as exc:
                error = {"stage": "calculation/evaluation", "error": f"{type(exc).__name__}: {exc}"}
                _json(case / "failure.json", error)
                rows.append(
                    {
                        "dataset": name,
                        "family": source["family"],
                        "method": method["name"],
                        "completed": False,
                        **error,
                        "seconds": time.monotonic() - start,
                    }
                )
                print(f"FAILED {name}/{method['name']}: {exc}", flush=True)
            pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    assessment = {
        "expected_cases": len(protocol["datasets"]) * len(protocol["methods"]),
        "completed_cases": sum(row["completed"] for row in rows),
        "failed_cases": sum(not row["completed"] for row in rows),
        "families": sorted({source["family"] for source in protocol["datasets"]}),
        "all_failures_retained": True,
        "historical_preprocessing_or_test_mean_correction": False,
        "prospective_or_blinded": False,
        "package_certified": False,
        "interpretation": protocol["interpretation"],
        "runner_sha256": provenance.file_sha256(Path(__file__)),
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): provenance.file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("analysis/*.py"))
        },
    }
    _json(out / "assessment.json", assessment)
    _manifest(out)
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline-data", type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payloads: dict[str, dict[str, bytes]] = {}
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for source in protocol["datasets"]:
            payloads[source["name"]] = {}
            for split in ("train", "test"):
                filename = f"{source['name']}_{split}.sdf"
                if args.offline_data:
                    content = (args.offline_data / filename).read_bytes()
                else:
                    response = client.get(protocol["data_base_url"] + filename)
                    response.raise_for_status()
                    content = response.content
                payloads[source["name"]][split] = content
    result = run_campaign(protocol, payloads, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

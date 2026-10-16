"""Frozen steroid/CBG comparison from primary-paper data, never GPL source.

Haga et al. Pharmaceuticals 2025, 18:440, doi:10.3390/ph18030440.
Run explicitly: python -m sarscope.field_benchmark --out NEW_DIRECTORY.
The pinned upstream SDF is fetched for local use, not redistributed here.
No protocol is selected by the resulting test scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.field_qsar import (
    FieldDataset,
    prepare_field_series,
    run_field_qsar,
    upload_table,
)
from sarscope.params import FieldQsarParams, ModelParams, RunParams

PROTOCOL_PATH = Path(__file__).resolve().parents[2] / "benchmarks/field_qsar/protocol.json"


def load_protocol(path: Path = PROTOCOL_PATH) -> dict[str, Any]:
    protocol = json.loads(path.read_text())
    if protocol.get("schema") != 1:
        raise ValueError("Unknown frozen field-benchmark protocol schema.")
    return protocol


def verify_source(payload: bytes, metadata: dict[str, Any]) -> None:
    if hashlib.sha256(payload).hexdigest() != metadata["sha256"]:
        raise ValueError(
            "Pinned benchmark source hash differs; re-audit, do not silently update it."
        )
    if "bytes" in metadata and len(payload) != metadata["bytes"]:
        raise ValueError("Pinned benchmark source byte count differs.")


def article_tables(payload: bytes, protocol: dict[str, Any]) -> dict[str, Any]:
    root = ET.fromstring(payload)
    license_text = " ".join("".join(n.itertext()) for n in root.findall(".//permissions"))
    if "creativecommons.org/licenses/by/4.0/" not in license_text:
        raise ValueError("Primary-paper CC BY licence statement was not found.")
    by_id = {table.attrib["id"]: table for table in root.findall(".//table-wrap")}
    activity_rows = []
    split = None
    for row in by_id[protocol["article"]["activity_table"]].findall(".//tbody/tr"):
        cells = [" ".join("".join(cell.itertext()).split()) for cell in row]
        if len(cells) == 1:
            split = {"Training Set": "train", "Test Set": "held_out"}.get(cells[0])
            continue
        if cells[0] in ("Mean", "Standard Deviation"):
            continue
        if split is None or len(cells) != 5:
            raise ValueError("Unrecognised primary-paper activity table structure.")
        activity_rows.append(
            {"article_name": cells[0], "pactivity": float(cells[1]), "split": split}
        )
    metric_rows = []
    for row in by_id[protocol["article"]["metrics_table"]].findall(".//tr"):
        metric_rows.append([" ".join("".join(cell.itertext()).split()) for cell in row])
    q2 = next(row for row in metric_rows if row[0] == "q2")
    expected = protocol["published_reference"]
    if [float(x) for x in q2[1:]] != [
        expected["seh_classic_as_reported_in_table1"]["q2"],
        expected["seh_pycomsia_table1"]["q2"],
        expected["sehad_pycomsia_table1"]["q2"],
    ]:
        raise ValueError("Published reference q² values do not match the frozen primary table.")
    return {
        "citation": "Haga et al. Pharmaceuticals 2025, 18, 440; doi:10.3390/ph18030440",
        "license": license_text,
        "table1": metric_rows,
        "table2": activity_rows,
    }


def audit_dataset(
    payload: bytes, reference: dict[str, Any], protocol: dict[str, Any]
) -> tuple[list[Any], pd.DataFrame, dict[str, Any]]:
    _, molecules = upload_table(payload, "sdf")
    if len(molecules) != protocol["dataset"]["records"] or len(reference["table2"]) != 31:
        raise ValueError("Benchmark must have 32 deposited / 31 published compounds.")
    excluded = molecules[31]
    if excluded.GetProp("_Name") != protocol["dataset"]["excluded_name"]:
        raise ValueError("The excluded extra compound differs from the frozen protocol.")
    rows = []
    for index, (mol, paper) in enumerate(zip(molecules[:31], reference["table2"], strict=True)):
        name = mol.GetProp("_Name")
        if protocol["dataset"]["name_aliases"].get(name, name) != paper["article_name"]:
            raise ValueError(f"Identity/order mismatch at deposited compound {index + 1}: {name}.")
        stored = mol.GetDoubleProp(protocol["dataset"]["activity_property"])
        if not np.isfinite(stored) or stored >= 0:
            raise ValueError("Stored CBG sign convention differs from the audited dataset.")
        heavy = Chem.RemoveHs(mol)
        metadata = Chem.MolFromSmiles(mol.GetProp("isomericSMILES"))
        a, b = Chem.Mol(heavy), Chem.Mol(metadata)
        Chem.RemoveStereochemistry(a)
        Chem.RemoveStereochemistry(b)
        graph_agrees = Chem.MolToSmiles(a) == Chem.MolToSmiles(b)
        stereo_agrees = heavy.HasSubstructMatch(metadata, useChirality=True)
        if not graph_agrees or not stereo_agrees:
            raise ValueError(f"Structure metadata mismatch for {name}; do not repair silently.")
        expected_split = "train" if index < 21 else "held_out"
        if paper["split"] != expected_split:
            raise ValueError("Primary-paper training/test order differs from the frozen partition.")
        coordinates = heavy.GetConformer().GetPositions()
        diff = -stored - paper["pactivity"]
        rows.append(
            {
                "record_1based": index + 1,
                "molecule_id": name,
                "article_name": paper["article_name"],
                "split": expected_split,
                "stored_pKi_CBG": stored,
                "sdf_pactivity": -stored,
                "article_pactivity": paper["pactivity"],
                "sdf_minus_article": diff,
                "beyond_two_decimal_rounding": abs(diff) > 0.0051,
                "graph_metadata_agrees": graph_agrees,
                "encoded_stereo_agrees": stereo_agrees,
                "structure_sha256": hashlib.sha256(Chem.MolToSmiles(heavy).encode()).hexdigest(),
                "heavy_coordinates_sha256": hashlib.sha256(
                    coordinates.astype("<f8").tobytes()
                ).hexdigest(),
            }
        )
    extra_key = Chem.MolToSmiles(Chem.RemoveHs(excluded))
    duplicate_of = [
        mol.GetProp("_Name")
        for mol in molecules[:31]
        if Chem.MolToSmiles(Chem.RemoveHs(mol)) == extra_key
    ]
    return (
        molecules[:31],
        pd.DataFrame(rows),
        {
            "excluded_record": 32,
            "excluded_name": excluded.GetProp("_Name"),
            "same_encoded_graph_as": duplicate_of,
            "reason": protocol["dataset"]["exclusion_reason"],
            "independent_structure_adjudication": "SDF graph versus its separately encoded SMILES; "
            "not independent experimental verification of structures or activity measurements",
        },
    )


def _dataset(
    molecules: list[Any], audit: pd.DataFrame, source: str, protocol: dict[str, Any]
) -> FieldDataset:
    activity_column = "sdf_pactivity" if source == "sdf_sign_corrected" else "article_pactivity"
    table = pd.DataFrame(
        {
            "molecule_id": audit["molecule_id"],
            "smiles": [Chem.MolToSmiles(Chem.RemoveHs(mol)) for mol in molecules],
            "pactivity": audit[activity_column],
            "n_measurements": 1,
            "pactivity_range": 0.0,
        }
    )
    measurements = table[["molecule_id", "smiles", "pactivity"]].copy()
    measurements["record_id"] = [f"Coats_CBG:{i + 1}" for i in range(len(table))]
    return FieldDataset(
        table,
        molecules,
        measurements,
        pd.DataFrame(
            [
                {
                    "stage": "fixed literature cohort",
                    "deposited": 32,
                    "included": 31,
                    "train": 21,
                    "held_out": 10,
                }
            ]
        ),
        {
            "kind": "frozen_literature_comparison",
            "activity_source": source,
            "dataset_sha256": protocol["dataset"]["sha256"],
            "article_sha256": protocol["article"]["sha256"],
            "endpoint": "CBG historical binding activity; tabulated point convention",
            "activity_transform": protocol["dataset"]["activity_transformation"]
            if source == "sdf_sign_corrected"
            else "positive Table 2 values, no transformation",
        },
    )


def run_benchmark(
    protocol: dict[str, Any], sdf: bytes, article: bytes, out: Path
) -> dict[str, Any]:
    verify_source(sdf, protocol["dataset"])
    verify_source(article, protocol["article"])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(
            "Use a new empty benchmark output directory; prior evidence is preserved."
        )
    out.mkdir(parents=True, exist_ok=True)
    reference = article_tables(article, protocol)
    molecules, audit, extra = audit_dataset(sdf, reference, protocol)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (out / "primary_reference.json").write_text(json.dumps(reference, indent=2) + "\n")
    (out / "extra_compound.json").write_text(json.dumps(extra, indent=2) + "\n")
    audit.to_csv(out / "input_audit.csv", index=False)
    (out / "ATTRIBUTION.txt").write_text(
        reference["citation"] + "\nPublished Table 1/2 reference data: CC BY 4.0.\n"
        "Upstream Coats/exeResearch SDF is fetched by pinned URL/hash for local execution; "
        "its redistribution permission was not established and it is NOT bundled here.\n"
    )
    partition = (list(range(21)), list(range(21, 31)))
    summaries, failures = [], []
    for source in protocol["activity_sensitivity"]:
        dataset = _dataset(molecules, audit, source, protocol)
        # The fixed alignment is reviewed once; geometry does not depend on y.
        try:
            prepared = prepare_field_series(
                dataset, FieldQsarParams(**protocol["geometry_quality_control"])
            )
        except Exception as exc:
            failures.append(
                {"case": f"alignment__{source}", "error": f"{type(exc).__name__}: {exc}"}
            )
            print(f"FAILED alignment__{source}: {exc}", flush=True)
            continue
        for spec in protocol["presets"]:
            case = f"{spec['name']}__{source}"
            print(f"Running {case} (21 training / 10 test)…", flush=True)
            start = time.monotonic()
            settings = {
                key: value
                for key, value in spec.items()
                if key not in ("name", "regression_cv", "cv_folds")
            }
            if "field_subset" in settings:
                settings["field_subset"] = tuple(settings["field_subset"])
            n_permutations = (
                protocol["negative_controls"]["permutations_native"]
                if spec["name"] == "native_default"
                else 0
            )
            params = FieldQsarParams(
                **settings,
                permutations=n_permutations,
                mcs_timeout=protocol["geometry_quality_control"]["mcs_timeout"],
                max_mappings=protocol["geometry_quality_control"]["max_mappings"],
                assess_prealigned_core=protocol["geometry_quality_control"][
                    "assess_prealigned_core"
                ],
                max_grid_points=20000,
                max_memory_mb=512,
                permutation_timeout=300,
            )
            model = ModelParams(
                split="random",
                cv_folds=spec["cv_folds"],
                regression_cv=spec["regression_cv"],
                seed=42,
            )
            try:
                with threadpool_limits(limits=1):
                    result = run_field_qsar(
                        replace(prepared, params=params), model, split_indices=partition
                    )
                folder = out / case
                folder.mkdir()
                result.predictions.drop(columns="smiles").to_csv(
                    folder / "predictions.csv", index=False
                )
                result.statistics.to_csv(folder / "statistics.csv", index=False)
                result.permutations.to_csv(folder / "permutations.csv", index=False)
                result.checks.to_csv(folder / "checks.csv", index=False)
                result.regression.fitted_model.selection_table_.to_csv(
                    folder / "component_selection.csv", index=False
                )
                result.contours.summary.to_csv(folder / "contributions.csv", index=False)
                (folder / "provenance.json").write_text(
                    json.dumps(result.provenance, indent=2, allow_nan=False) + "\n"
                )
                row = result.statistics.iloc[0]
                ref_key = (
                    "seh_pycomsia_table1"
                    if len(params.field_subset) == 3
                    else "sehad_pycomsia_table1"
                )
                published_q2 = protocol["published_reference"][ref_key]["q2"]
                selection = result.regression.fitted_model.selection_table_
                summary = {
                    "case": case,
                    "activity_source": source,
                    "preset": spec["name"],
                    "training": 21,
                    "test": 10,
                    "cv": spec["regression_cv"],
                    "components": int(row.selected_components),
                    "n_features": int(row.n_features),
                    "q2": float(row.cv_pooled_q2),
                    "published_q2": published_q2,
                    "published_reference_key": ref_key,
                    "classic_seh_q2_as_reported": protocol["published_reference"][
                        "seh_classic_as_reported_in_table1"
                    ]["q2"]
                    if len(params.field_subset) == 3
                    else None,
                    "q2_difference": float(row.cv_pooled_q2) - published_q2,
                    "q2_comparable_protocol": spec["regression_cv"] == "loo",
                    "train_r2": float(row.train_r2),
                    "test_r2": float(row.test_r2),
                    "test_rmse": float(row.test_rmse),
                    "test_spearman": float(row.test_spearman),
                    "permutation_pvalue": float(row.permutation_pvalue) if n_permutations else None,
                    "selection_q2_max": float(selection.selection_q2.max())
                    if selection.selection_q2.notna().any()
                    else None,
                    "grid_points": result.fields.grid.n_points,
                    "baseline_rf_test_r2": float(
                        result.statistics.query("algorithm == 'random_forest'").iloc[0].test_r2
                    ),
                    "elapsed_seconds": time.monotonic() - start,
                    "exact_reproduction": False,
                }
                summaries.append(summary)
                print(json.dumps(summary, allow_nan=False), flush=True)
            except Exception as exc:
                failures.append({"case": case, "error": f"{type(exc).__name__}: {exc}"})
                print(f"FAILED {case}: {exc}", flush=True)
    summary_table = pd.DataFrame(summaries)
    summary_table.to_csv(out / "summary.csv", index=False)
    run = {
        "schema": 1,
        "created_utc": provenance.collect(RunParams(), {})["created_utc"],
        "packages": provenance.package_versions(),
        "protocol_sha256": hashlib.sha256(
            json.dumps(protocol, sort_keys=True).encode()
        ).hexdigest(),
        "benchmark_runner_sha256": provenance.file_sha256(Path(__file__)),
        "completed_cases": len(summaries),
        "failed_cases": failures,
        "activity_discrepancies": audit.loc[
            audit.beyond_two_decimal_rounding,
            ["molecule_id", "sdf_pactivity", "article_pactivity", "sdf_minus_article"],
        ].to_dict("records"),
        "exact_historical_reproduction": False,
        "conclusion": "Executed field-method comparison, not exact historical reproduction, "
        "external multi-target validation, or evidence that every user model is valid."
        if summaries
        else "No model scores produced; preparation failed. See failed_cases.",
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    if summaries:
        from sarscope.field_benchmark_report import write_benchmark_report

        write_benchmark_report(out)
    manifest = {
        str(path.relative_to(out)): provenance.file_sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    (out / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        sources = []
        for name in ("dataset", "article"):
            response = client.get(protocol[name]["url"])
            response.raise_for_status()
            sources.append(response.content)
    result = run_benchmark(protocol, sources[0], sources[1], args.out)
    print(json.dumps(result, indent=2), flush=True)
    if result["failed_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

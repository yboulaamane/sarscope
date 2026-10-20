"""Frozen steroid-paper repetition plus generated-alignment sensitivity.

Only upstream chemical data are fetched, never external algorithm source.
Historical reproduction and ordinary training-only modelling remain separate.
The already inspected test set is not a fresh validation challenge.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import httpx
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.field_qsar import prepare_field_series, run_field_qsar
from sarscope.field_reproduction import dataset_from_sources, reproduce
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "benchmarks/field_qsar/independent-protocol.json"
ROUTES = ("prealigned", "rigid_mcs", "constrained")
SOURCE_FILES = (
    "benchmarks/reproduce_steroid_alignment.py",
    "src/sarscope/field_reproduction.py",
    "src/sarscope/analysis/align3d.py",
    "src/sarscope/analysis/fields.py",
    "src/sarscope/analysis/mmff_fields.py",
    "src/sarscope/analysis/regression.py",
    "src/sarscope/analysis/field_filter.py",
    "src/sarscope/conformers.py",
    "src/sarscope/field_qsar.py",
    "src/sarscope/params.py",
)


def save(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda x: x.item()) + "\n")


def prepare(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    historical = json.loads(HISTORICAL.read_text())
    payloads = {}
    data = out / "data"
    data.mkdir()
    with httpx.Client(follow_redirects=True, timeout=40) as client:
        for split in ("train", "test"):
            info = historical["data"][split]
            response = client.get(historical["data"]["base_url"] + info["file"])
            response.raise_for_status()
            payloads[split] = response.content
            (data / info["file"]).write_bytes(response.content)
    dataset, audit = dataset_from_sources(historical, payloads)
    if dataset.table.smiles.duplicated().any():
        raise ValueError("Duplicate chemical identities require grouped validation.")
    audit.to_csv(out / "source_accounting.csv", index=False)
    dataset.table.drop(columns="smiles").assign(split=["train"] * 21 + ["test"] * 10).to_csv(
        out / "membership.csv", index=False
    )
    common = FieldQsarParams(
        spacing=1.0,
        field_scaling="block_std",
        component_selection="max_q2",
        assess_prealigned_core=False,
        preserve_input_hydrogens=False,
        max_grid_points=40000,
        max_memory_mb=512,
        permutations=0,
        seed=42,
    )
    profiles = {
        "gaussian_SEH": replace(common, field_subset=("steric", "electrostatic", "hydrophobic")),
        "mmff94_SE": replace(
            common,
            method="comfa",
            field_definition="mmff94_mif",
            charge_model="mmff94",
            minimum_distance=0.0001,
            electrostatic_cutoff=30.0,
        ),
    }
    save(
        out / "protocol.json",
        {
            "purpose": "Repeat published SEH internal metrics and native alignment sensitivity",
            "dataset": "Steroid/CBG: 21 training and 10 original test compounds; Haga et al. 2025",
            "doi": "10.3390/ph18030440",
            "selection": (
                "Measured endpoint, source alignment, fixed split and numeric targets; "
                "not selected on new scores"
            ),
            "historical_protocol": historical,
            "historical_protocol_sha256": provenance.file_sha256(HISTORICAL),
            "routes": list(ROUTES),
            "alignment_settings": asdict(common),
            "reference": {
                "index": 0,
                "molecule_id": dataset.table.iloc[0].molecule_id,
                "rule": "First source training molecule; fixed before fitting",
            },
            "profiles": {name: asdict(value) for name, value in profiles.items()},
            "model": asdict(
                ModelParams(
                    split="random", cv_folds=3, seed=42, oversample=False, leakage_audit=False
                )
            ),
            "native_hydrogen_control": (
                "Remove original H in every route; same compute_fields "
                "AddHs(addCoords=True) for all native cases"
            ),
            "native_training_only": (
                "Training grid, fold-local filtering/scaling/component selection; n/5 cap retained"
            ),
            "alignment_comparison_scope": (
                "Generation/conformer + alignment pipeline effect, "
                "not rigid-transform-only or pose validation"
            ),
            "test_set_status": (
                "Historical, already inspected; consumed reproduction/sensitivity "
                "fixture, not blinded"
            ),
            "all_failures_retained": True,
            "no_test_driven_selection_or_retry": True,
            "permutations": 0,
            "predictive_qualification": False,
            "source_sha256": {name: provenance.file_sha256(ROOT / name) for name in SOURCE_FILES},
            "packages": provenance.package_versions(),
        },
    )


def settings(raw: dict) -> FieldQsarParams:
    values = raw.copy()
    for key in ("field_subset", "grid_phase"):
        values[key] = tuple(values[key])
    return FieldQsarParams(**values)


def run(out: Path) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        if provenance.file_sha256(ROOT / name) != expected:
            raise ValueError("Source changed after freeze: " + name)
    historical = protocol["historical_protocol"]
    payloads = {
        split: (out / "data" / historical["data"][split]["file"]).read_bytes()
        for split in ("train", "test")
    }
    dataset, _ = dataset_from_sources(historical, payloads)
    print("Repeat own historical SEH calculation", flush=True)
    reproduce(historical, payloads, out / "historical_repeat")
    rows = []
    model = ModelParams(**protocol["model"])
    for route in protocol["routes"]:
        folder = out / route
        folder.mkdir(exist_ok=False)
        print("Prepare alignment:", route, flush=True)
        start = time.monotonic()
        try:
            aligned = prepare_field_series(
                dataset, replace(settings(protocol["alignment_settings"]), alignment=route)
            )
            elapsed = time.monotonic() - start
            aligned.alignment.report.to_csv(folder / "alignment.csv", index=False)
            writer = Chem.SDWriter(str(folder / "aligned.sdf"))
            for original, actual in zip(
                dataset.molecules, aligned.alignment.molecules, strict=True
            ):
                if Chem.MolToSmiles(Chem.RemoveHs(original)) != Chem.MolToSmiles(
                    Chem.RemoveHs(actual)
                ):
                    raise ValueError("Alignment changed chemical identity; no score accepted")
                writer.write(actual)
            writer.close()
        except Exception as exc:
            save(
                folder / "alignment_failure.json",
                {"error": str(exc), "elapsed_s": time.monotonic() - start},
            )
            for profile in protocol["profiles"]:
                rows.append(
                    {
                        "route": route,
                        "profile": profile,
                        "status": "alignment_failed",
                        "error": str(exc),
                    }
                )
            print(route, "ALIGNMENT FAILED:", exc, flush=True)
            continue
        for name, raw in protocol["profiles"].items():
            print("Native model:", route, name, flush=True)
            case = folder / name
            case.mkdir(exist_ok=False)
            began = time.monotonic()
            row = {"route": route, "profile": name, "alignment_wall_s": elapsed}
            try:
                with threadpool_limits(limits=1):
                    result = run_field_qsar(
                        replace(aligned, params=settings(raw)),
                        model,
                        split_indices=(list(range(21)), list(range(21, 31))),
                    )
                for filename, frame in (
                    ("statistics.csv", result.statistics),
                    ("predictions.csv", result.predictions.drop(columns="smiles")),
                    ("checks.csv", result.checks),
                    ("contributions.csv", result.contours.summary),
                ):
                    frame.to_csv(case / filename, index=False)
                save(case / "provenance.json", result.provenance)
                stat = result.statistics.set_index("algorithm").loc["field_pls"]
                row.update(
                    status="ok",
                    nested_q2=float(stat.cv_pooled_q2),
                    test_r2=float(stat.test_r2),
                    test_rmse=float(stat.test_rmse),
                    components=int(stat.selected_components),
                    failed_checks=result.checks.loc[~result.checks.passed, "check"].tolist(),
                )
            except Exception as exc:
                row.update(status="model_failed", error=str(exc))
            row["model_wall_s"] = time.monotonic() - began
            save(case / "assessment.json", row)
            rows.append(row)
            print(row, flush=True)
            pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "summary.csv", index=False)
    save(
        out / "manifest.sha256.json",
        {
            str(path.relative_to(out)): provenance.file_sha256(path)
            for path in sorted(out.rglob("*"))
            if path.is_file()
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    (prepare if args.action == "prepare" else run)(args.out.resolve())

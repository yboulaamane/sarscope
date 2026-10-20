"""Check independently implemented MMFF MIFs against frozen black-box outputs.

No GPL algorithm source or external executable is needed by this runner.
Existing native Open3DQSAR outputs are comparison evidence only. Differences
are retained; no constants or molecules are fitted to those outputs.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from threadpoolctl import threadpool_limits

from benchmarks.compare_o3a_alignment import sha, write_json
from sarscope.field_qsar import FieldDataset, prepare_field_series, run_field_qsar
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "runs/open3dqsar-native-20261019"
SOURCES = [
    "src/sarscope/analysis/mmff_fields.py",
    "src/sarscope/analysis/fields.py",
    "src/sarscope/params.py",
    "src/sarscope/field_qsar.py",
    "src/sarscope/analysis/field_filter.py",
    "src/sarscope/analysis/regression.py",
    "tests/test_mmff_fields.py",
    "benchmarks/validate_mmff_mif.py",
]


def prepare(out: Path) -> None:
    out.mkdir(exist_ok=False, parents=True)
    previous = json.loads((EXTERNAL / "protocol.json").read_text())
    cases = []
    for info in previous["cases"]:
        name = f"{info['cohort']}_{info['route']}"
        path = EXTERNAL / name
        settings = info["common_engine_params"] | {
            "field_definition": "mmff94_mif",
            "charge_model": "mmff94",
            "allow_charge_fallback": False,
            "minimum_distance": 0.0001,
            "steric_cutoff": 30.0,
            "electrostatic_cutoff": 30.0,
            "missing_field_policy": "refuse",
        }
        for key in ("grid_phase", "field_subset"):
            settings[key] = tuple(settings[key])
        params = FieldQsarParams(**settings)
        cases.append(
            {
                "name": name,
                "train": info["train"],
                "test": info["test"],
                "params": asdict(params),
                "external_grid": info["grid"],
                "input_sha256": {
                    f: sha(path / f)
                    for f in ("input.sdf", "membership.csv", "native_fields.npy", "summary.json")
                },
            }
        )
    write_json(
        out / "protocol.json",
        {
            "scope": "Independent public-equation implementation versus frozen native "
            "grids and predictions",
            "rdkit": rdBase.rdkitVersion,
            "cases": cases,
            "source_sha256": {f: sha(ROOT / f) for f in SOURCES},
            "prior_protocol_sha256": sha(EXTERNAL / "protocol.json"),
            "parameters": "Public Halgren buffered 14-7 + RDKit MMFF94 pair API; +1e point "
            "probe, MMFF Coulomb 332.0716 with published 0.05 A buffer, independent +/-30 cutoffs",
            "grid_agreement_diagnostics": {
                "steric_atol_kcal": 0.001,
                "electrostatic_atol_kcal": 0.01,
                "all_maxima_and_molecule_outliers_reported": True,
            },
            "no_parameter_fitting": True,
            "permutations": 0,
            "oracle_note": "Installed rdMIF MMFFVdWaals docstring says kJ/mol; actual values "
            "match MMFF kcal/mol pair epsilon. Removed an erroneous test-only /4.184 "
            "conversion after analytic unit check; field equations were not rescaled.",
            "license_scope": "Own code and BSD RDKit API only; no Open3DQSAR/OpenBabel source port",
        },
    )


def run(out: Path) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    if any(sha(ROOT / f) != expected for f, expected in protocol["source_sha256"].items()):
        raise ValueError("Implementation changed after protocol freeze")
    summaries, molecule_errors = [], []
    for info in protocol["cases"]:
        name = info["name"]
        source = EXTERNAL / name
        folder = out / name
        folder.mkdir(exist_ok=False)
        summary = {"case": name}
        try:
            if any(sha(source / f) != h for f, h in info["input_sha256"].items()):
                raise ValueError("Frozen external inputs changed")
            table = pd.read_csv(source / "membership.csv", dtype={"molecule_id": str})
            molecules = list(Chem.SDMolSupplier(str(source / "input.sdf"), removeHs=False))
            if len(molecules) != len(table) or any(m is None for m in molecules):
                raise ValueError("No missing molecules may be excluded")
            settings = info["params"].copy()
            for key in ("grid_phase", "field_subset"):
                settings[key] = tuple(settings[key])
            params = FieldQsarParams(**settings)
            frame = table[["molecule_id", "smiles", "pactivity"]].copy()
            data = FieldDataset(
                frame,
                molecules,
                frame.copy(),
                pd.DataFrame(),
                {"scope": "Historical comparison", "license": "CC-BY-NC-4.0"},
            )
            start = time.monotonic()
            with threadpool_limits(limits=1):
                result = run_field_qsar(
                    prepare_field_series(data, params),
                    ModelParams(
                        split="random",
                        cv_folds=3,
                        test_fraction=0.33,
                        seed=42,
                        oversample=False,
                        leakage_audit=False,
                    ),
                    split_indices=(info["train"], info["test"]),
                )
            summary["full_pipeline_wall_s"] = time.monotonic() - start
            if json.loads(json.dumps(asdict(result.fields.grid))) != info["external_grid"]:
                raise ValueError("Grid differs from frozen external comparison")
            external = np.load(source / "native_fields.npy")
            own = result.fields.matrix
            if own.shape != external.shape:
                raise ValueError("External matrix membership/column shape differs")
            for index, (field, tolerance) in enumerate(
                (("steric", 0.001), ("electrostatic", 0.01))
            ):
                a = own.reshape(len(table), 2, -1)[:, index]
                b = external.reshape(len(table), 2, -1)[:, index]
                differences = np.abs(a - b)
                summary.update(
                    {
                        f"{field}_max_abs_difference": float(differences.max()),
                        f"{field}_mean_abs_difference": float(differences.mean()),
                        f"{field}_p99_abs_difference": float(np.percentile(differences, 99)),
                        f"{field}_fraction_within_atol": float((differences <= tolerance).mean()),
                    }
                )
                for i, identifier in enumerate(table.molecule_id):
                    molecule_errors.append(
                        {
                            "case": name,
                            "molecule_id": identifier,
                            "field": field,
                            "max_abs_difference": float(differences[i].max()),
                            "mean_abs_difference": float(differences[i].mean()),
                        }
                    )
            row = result.statistics.set_index("algorithm").loc["field_pls"]
            baseline = json.loads((source / "summary.json").read_text())
            summary.update(
                status="ok",
                n=len(table),
                nested_q2=float(row.cv_pooled_q2),
                test_r2=float(row.test_r2),
                test_rmse=float(row.test_rmse),
                components=int(row.selected_components),
                external_test_r2=baseline["common_test_r2"],
                external_nested_q2=baseline["common_nested_q2"],
                external_test_rmse=baseline["common_test_rmse"],
            )
            for filename, frame in (
                ("statistics", result.statistics),
                ("predictions", result.predictions),
                ("checks", result.checks),
                ("charges", pd.DataFrame(result.fields.charges)),
            ):
                frame.to_csv(folder / f"{filename}.csv", index=False)
            write_json(folder / "provenance.json", result.provenance)
        except Exception as exc:
            summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_json(folder / "summary.json", summary)
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    pd.DataFrame(summaries).to_csv(out / "results.csv", index=False)
    pd.DataFrame(molecule_errors).to_csv(out / "molecule_grid_differences.csv", index=False)
    write_json(
        out / "manifest.sha256.json",
        {
            str(p.relative_to(out)): sha(p)
            for p in out.rglob("*")
            if p.is_file() and p.name != "manifest.sha256.json"
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args.out.resolve())
    else:
        run(args.out.resolve())

"""Compare raw versus block-scaled SEH inputs through existing SARscope regression.

No external source code is consulted. Both matrices come from the preceding
isolated field comparison. Settings are frozen before this diagnostic run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.regression import evaluate_regression
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parent


def pooled_q2(truth: Any, prediction: Any) -> float:
    truth, prediction = np.asarray(truth, dtype=float), np.asarray(prediction, dtype=float)
    if truth.ndim != 1 or prediction.shape != truth.shape or not np.isfinite(prediction).all():
        raise ValueError("One finite CV prediction is required for each training row.")
    denominator = float(np.sum((truth - truth.mean()) ** 2))
    if not np.isfinite(truth).all() or denominator <= 0:
        raise ValueError("q2 needs finite, varying measured activities.")
    return 1 - float(np.sum((prediction - truth) ** 2)) / denominator


def verify_manifest(folder: Path) -> None:
    manifest = json.loads((folder / "manifest.sha256.json").read_text())
    for name, expected in manifest.items():
        if provenance.file_sha256(folder / name) != expected:
            raise ValueError(f"Frozen input manifest differs: {folder.name}/{name}")


def run(out: Path) -> dict[str, Any]:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory.")
    protocol_path = ROOT / "model-parity-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    fields_folder = ROOT / protocol["field_arrays"]
    audit_path = ROOT / protocol["labels_and_identity_audit"]
    verify_manifest(fields_folder)
    verify_manifest(audit_path.parent)
    audit = json.loads(audit_path.read_text())
    records = audit["train"]["records"] + audit["test"]["records"]
    if len(records) != 31:
        raise ValueError("Expected all 31 original valid molecules.")
    target = np.array([row["activity"] for row in records])
    settings = protocol["settings"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    summaries, differences = [], []
    fitted = {}
    for scaling in settings["scaling_modes"]:
        for side in ("reference", "sarscope", "reference_unit_converted"):
            source = "reference" if side == "reference_unit_converted" else side
            with np.load(fields_folder / f"{source}.npz") as fields:
                blocks = [fields[name] for name in protocol["fields"]]
            if side == "reference_unit_converted":
                # Post-hoc diagnosis based on the preceding field experiment, NOT y.
                # No such conversion is inserted into SARscope production.
                parity = json.loads((fields_folder / "assessment.json").read_text())
                blocks = [
                    block / parity["fields"][name]["posthoc_reference_over_sarscope_factor"]
                    for block, name in zip(blocks, protocol["fields"], strict=True)
                ]
            matrix = np.hstack(blocks)
            params = FieldQsarParams(
                method="comsia",
                field_subset=tuple(protocol["fields"]),
                field_scaling=scaling,
                keep_fraction=settings["filter_keep_fraction"],
                max_features=settings["max_features"],
                fixed_components=settings["fixed_components"],
                component_cap=settings["component_cap"],
                permutations=0,
            )
            model = ModelParams(
                split="random",
                regression_cv="loo",
                regression_algorithms=("field_pls",),
                field_qsar=params,
                seed=settings["seed"],
            )
            with threadpool_limits(limits=1):
                result = evaluate_regression(
                    matrix,
                    target,
                    ["series"] * 31,
                    ["series"] * 31,
                    model,
                    split_indices=(list(range(21)), list(range(21, 31))),
                )
            fitted[scaling, side] = result
            name = f"{scaling}__{side}"
            row = result.scores.query("algorithm == 'field_pls'").iloc[0].to_dict()
            row.update(scaling=scaling, field_generator=side)
            row["pooled_q2"] = pooled_q2(target[:21], result.cv_predictions)
            row["posthoc_unit_conversion_diagnostic"] = side == "reference_unit_converted"
            summaries.append(row)
            result.scores.to_csv(out / f"{name}__scores.csv", index=False)
            pd.DataFrame(
                {
                    "name": [r["name"] for r in records[21:]],
                    "observed": target[21:],
                    "prediction": result.test_predictions,
                }
            ).to_csv(out / f"{name}__external_predictions.csv", index=False)
            pd.DataFrame(
                {
                    "name": [r["name"] for r in records[:21]],
                    "observed": target[:21],
                    "loo_prediction": result.cv_predictions,
                }
            ).to_csv(out / f"{name}__loo_predictions.csv", index=False)
            print(name, row, flush=True)
        for variant in ("reference", "reference_unit_converted"):
            a, b = fitted[scaling, variant], fitted[scaling, "sarscope"]
            external_difference = float(np.max(np.abs(a.test_predictions - b.test_predictions)))
            loo_difference = float(np.max(np.abs(a.cv_predictions - b.cv_predictions)))
            q2_a = pooled_q2(target[:21], a.cv_predictions)
            q2_b = pooled_q2(target[:21], b.cv_predictions)
            counts_equal = int(a.fitted_filter.support_.sum()) == int(
                b.fitted_filter.support_.sum()
            )
            tolerance = protocol["comparison_tolerances"]
            differences.append(
                {
                    "reference_variant": variant,
                    "scaling": scaling,
                    "external_prediction_max_abs_difference": external_difference,
                    "loo_prediction_max_abs_difference": loo_difference,
                    "q2_absolute_difference": abs(q2_a - q2_b),
                    "same_final_feature_count": counts_equal,
                    "comparison_passes_declared_tolerances": bool(
                        counts_equal
                        and external_difference <= tolerance["prediction_max_absolute_difference"]
                        and loo_difference <= tolerance["prediction_max_absolute_difference"]
                        and abs(q2_a - q2_b) <= tolerance["q2_absolute_difference"]
                    ),
                }
            )
    pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
    assessment = {
        "probe_sha256": provenance.file_sha256(Path(__file__)),
        "protocol_sha256": provenance.file_sha256(protocol_path),
        "field_manifest_sha256": provenance.file_sha256(fields_folder / "manifest.sha256.json"),
        "packages": provenance.package_versions(),
        "source_sha256": {
            name: provenance.file_sha256(ROOT.parents[1] / "src/sarscope" / name)
            for name in ("analysis/regression.py", "analysis/field_filter.py", "params.py")
        },
        "comparisons": differences,
        "posthoc_diagnostic_note": "The reference_unit_converted cases were added after "
        "the initial raw/block comparison exposed unequal variance-mask counts. Their fixed "
        "conversion factors come from the preceding geometry-only field comparison, not "
        "activity fitting or test-performance selection. They are a unit-convention diagnostic, "
        "not a prespecified reproduction success or a change to production methods.",
        "exact_historical_reproduction": False,
        "independent_predictive_validation": False,
        "note": "Controlled matrix-to-model comparison under a shared SARscope protocol, "
        "not reproduction of every stage of the external program or paper.",
    }
    (out / "assessment.json").write_text(json.dumps(assessment, indent=2) + "\n")
    manifest = {
        str(path.relative_to(out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    (out / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return assessment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.out), indent=2))


if __name__ == "__main__":
    main()

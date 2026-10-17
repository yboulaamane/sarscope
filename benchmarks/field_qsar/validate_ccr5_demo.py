"""Retrospective audit of the real native CCR5 tutorial; never certifies fresh validation.

Run with the installed SARscope environment and a NEW evidence directory:
    python benchmarks/field_qsar/validate_ccr5_demo.py --out NEW_DIRECTORY

No source downloads, external QSAR solver, test-label shift, preset selection,
or automatic change to the app is performed. All outcomes are retained.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.diagnostics import bootstrap_regression
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_campaign import _json, _manifest
from sarscope.field_example import field_example_dataset, field_example_files
from sarscope.field_qsar import FieldQsarResult, prepare_field_series, run_field_qsar
from sarscope.field_sensitivity import run_sensitivity, training_dataset
from sarscope.params import FieldQsarParams, ModelParams

PROTOCOL = Path(__file__).with_name("ccr5-demo-validation-protocol.json")


def save_result(result: FieldQsarResult, out: Path) -> dict[str, Any]:
    out.mkdir()
    for name, table in {
        "statistics": result.statistics,
        "predictions": result.predictions,
        "checks": result.checks,
        "permutations": result.permutations,
        "alignment": result.prepared.alignment.report,
        "contributions": result.contours.summary,
        "charges": pd.DataFrame(result.fields.charges),
        "component_search": result.regression.fitted_model.selection_table_,
    }.items():
        pd.DataFrame(table).to_csv(out / f"{name}.csv", index=False)
    record = dict(result.provenance)
    # The common runner's generic explicit-split wording must not imply a new,
    # untouched external set for this already inspected historical cohort.
    record["validation_scope"] = "Retrospective, previously inspected public cohort"
    if record["split_interpretation"] == "prespecified external benchmark partition":
        record["split_interpretation"] = "Re-evaluation of previously inspected historical split"
    _json(out / "provenance.json", record)
    return json.loads(
        result.statistics.query("algorithm == 'field_pls'").to_json(orient="records")
    )[0]


def invariance_checks(original: FieldQsarResult, altered: FieldQsarResult, tol: float) -> dict:
    train = original.regression.train_index
    left, right = original.regression, altered.regression
    comparisons = {
        "grid": asdict(original.fields.grid) == asdict(altered.fields.grid),
        "training_fields": np.allclose(
            original.fields.matrix[train],
            altered.fields.matrix[train],
            atol=tol,
            rtol=0,
            equal_nan=True,
        ),
        "training_support": np.array_equal(
            left.fitted_filter.support_, right.fitted_filter.support_
        ),
        "training_column_means": np.allclose(
            left.fitted_filter.column_mean_,
            right.fitted_filter.column_mean_,
            atol=tol,
            rtol=0,
            equal_nan=True,
        ),
        "training_column_scales": np.allclose(
            left.fitted_filter.column_scale_,
            right.fitted_filter.column_scale_,
            atol=tol,
            rtol=0,
        ),
        "selected_components": left.fitted_model.selected_components_
        == right.fitted_model.selected_components_,
        "coefficients": np.allclose(
            left.fitted_model.coef_, right.fitted_model.coef_, atol=tol, rtol=0
        ),
        "training_oof": np.allclose(left.cv_predictions, right.cv_predictions, atol=tol, rtol=0),
    }
    return {name: bool(value) for name, value in comparisons.items()}


def run(protocol_path: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a NEW empty evidence directory; never overwrite a previous run.")
    protocol = json.loads(protocol_path.read_text())
    if protocol["schema"] != 1:
        raise ValueError("Unknown validation protocol schema.")
    contents = field_example_files("ccr5")  # Verifies every bundled data hash.
    dataset = field_example_dataset("ccr5")
    reference = min(
        range(len(dataset.molecules)),
        key=lambda i: (
            dataset.molecules[i].GetNumHeavyAtoms(),
            str(dataset.table.iloc[i].molecule_id),
        ),
    )
    settings = FieldQsarParams(**{**protocol["fields"], "reference_index": reference})
    model = ModelParams(**protocol["model"])
    out.mkdir(parents=True, exist_ok=True)
    _json(out / "protocol.json", protocol)
    root = Path(__file__).resolve().parents[2] / "src" / "sarscope"
    frozen = {
        "protocol_sha256": provenance.file_sha256(protocol_path),
        "runner_sha256": provenance.file_sha256(Path(__file__)),
        "input_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        "source_sha256": {
            str(p.relative_to(root)): provenance.file_sha256(p) for p in sorted(root.rglob("*.py"))
        },
        "resolved_fields": asdict(settings),
        "resolved_model": asdict(model),
        "environment": provenance.package_versions(),
        "independent_external_validation": False,
        "frozen_before_this_run": True,
        "previously_observed_demo_metrics": True,
    }
    _json(out / "freeze.json", frozen)  # Written before preparing or fitting any model.
    print("Frozen inputs, settings and code before execution.", flush=True)

    table = add_scaffolds(dataset.table)
    graphs = [Chem.MolToSmiles(Chem.RemoveHs(m)) for m in dataset.molecules]
    if len(set(graphs)) != len(table) or not table.molecule_id.is_unique:
        raise ValueError("Duplicate encoded graphs/identifiers; no rows are silently removed.")
    audit = pd.read_csv(io.BytesIO(contents["preparation.csv"]), dtype={"molecule_id": str})
    audit = table[["molecule_id"]].merge(audit, on="molecule_id", validate="one_to_one")
    if len(audit) != 75 or audit.source_activity.isna().any():
        raise ValueError("Cohort accounting differs from all 75 declared records.")
    np.testing.assert_array_equal(audit.source_activity, table.pactivity)
    audit.to_csv(out / "preparation-audit.csv", index=False)

    with threadpool_limits(limits=1):
        prepared = prepare_field_series(dataset, settings)
        primary = run_field_qsar(prepared, model)
        primary_stats = save_result(primary, out / "primary")
        print("Primary:", primary_stats, flush=True)
        train, test = primary.regression.train_index, primary.regression.test_index
        partition = (train.tolist(), test.tolist())

        # Conditional intervals, not uncertainty of the entire model-building process.
        groups = table.iloc[test].murcko.tolist()
        bootstrap_tables = []
        for label, result in [
            ("3d_vs_mean", primary.regression),
            ("2d_rf_vs_mean", primary.baseline),
            (
                "3d_vs_2d_rf",
                replace(
                    primary.regression, baseline_test_predictions=primary.baseline.test_predictions
                ),
            ),
        ]:
            try:
                interval = bootstrap_regression(
                    result, groups, **{k: protocol["bootstrap"][k] for k in ("repeats", "seed")}
                )
                bootstrap_tables.append(interval.assign(comparison=label))
            except ValueError as exc:
                _json(out / f"bootstrap-{label}-failure.json", {"error": str(exc)})
        intervals = (
            pd.concat(bootstrap_tables, ignore_index=True) if bootstrap_tables else pd.DataFrame()
        )
        intervals.to_csv(out / "bootstrap.csv", index=False)

        # Measure structural interpolation rather than infer domain quality from a PCA box.
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fingerprints = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table.smiles]
        novelty = primary.predictions.iloc[test].copy()
        novelty["nearest_training_ecfp4_tanimoto"] = [
            max(
                DataStructs.BulkTanimotoSimilarity(
                    fingerprints[i], [fingerprints[j] for j in train]
                )
            )
            for i in test
        ]
        novelty["scaffold_in_training"] = (
            table.iloc[test].murcko.isin(set(table.iloc[train].murcko)).to_numpy()
        )
        novelty.to_csv(out / "held-out-novelty.csv", index=False)

        control_settings = replace(settings, permutations=0)
        control = replace(prepared, params=control_settings)
        changed_table = dataset.table.copy()
        changed_table.loc[test, "pactivity"] += protocol["leakage_controls"][
            "held_out_label_offset"
        ]
        changed_measurements = dataset.measurements.copy()
        mask = changed_measurements.molecule_id.isin(changed_table.iloc[test].molecule_id)
        changed_measurements.loc[mask, "pactivity"] += protocol["leakage_controls"][
            "held_out_label_offset"
        ]
        changed_dataset = replace(dataset, table=changed_table, measurements=changed_measurements)
        label_result = run_field_qsar(
            replace(control, dataset=changed_dataset), model, split_indices=partition
        )
        leakage = {
            "held_out_labels": invariance_checks(
                primary, label_result, protocol["leakage_controls"]["tolerance"]
            )
        }
        leakage["held_out_labels"]["held_out_predictions"] = bool(
            np.allclose(
                primary.regression.test_predictions,
                label_result.regression.test_predictions,
                atol=1e-10,
                rtol=0,
            )
        )
        shifted = [Chem.Mol(m) for m in prepared.alignment.molecules]
        shift = np.array(protocol["leakage_controls"]["held_out_translation_A"])
        for i in test:
            conf = shifted[i].GetConformer()
            for atom, xyz in enumerate(conf.GetPositions()):
                conf.SetAtomPosition(atom, xyz + shift)
        geometry_result = run_field_qsar(
            replace(control, alignment=replace(prepared.alignment, molecules=shifted)),
            model,
            split_indices=partition,
        )
        leakage["held_out_coordinates_after_alignment"] = invariance_checks(
            primary, geometry_result, protocol["leakage_controls"]["tolerance"]
        )
        leakage["scope"] = (
            "Post-alignment model path only; common-core alignment QC uses all structures. "
            "CV grid bounds use all outer-training geometry, while masks/scaling/components "
            "are fold-local."
        )
        _json(out / "leakage-controls.json", leakage)
        print("Leakage controls:", leakage, flush=True)

        splits = []
        for seed in protocol["additional_split_seeds"]:
            repeated = run_field_qsar(
                replace(prepared, params=replace(control_settings, seed=seed)),
                replace(model, seed=seed),
            )
            stats = save_result(repeated, out / f"split-{seed}")
            splits.append({"seed": seed, **stats})
            print("Additional split:", seed, stats["test_r2"], flush=True)
        pd.DataFrame(splits).to_csv(out / "additional-splits.csv", index=False)

        historical_train = np.flatnonzero(audit.original_partition.eq("train")).tolist()
        historical_test = np.flatnonzero(audit.original_partition.eq("test")).tolist()
        declared = protocol["historical_partition"]
        if (
            len(historical_train) != declared["training_records"]
            or len(historical_test) != declared["test_records"]
        ):
            raise ValueError("Historical source partition membership differs.")
        historical = run_field_qsar(
            replace(prepared, params=replace(settings, permutations=declared["permutations"])),
            model,
            split_indices=(historical_train, historical_test),
        )
        historical_stats = save_result(historical, out / "historical-63-12")
        print("Historical re-evaluation:", historical_stats, flush=True)

        if protocol["scaffold_diagnostic"]:
            try:
                scaffold = run_field_qsar(control, replace(model, split="scaffold"))
                save_result(scaffold, out / "scaffold")
            except ValueError as exc:
                _json(
                    out / "scaffold-failure.json",
                    {"completed": False, "error": str(exc), "no_random_fallback": True},
                )

        stability = run_sensitivity(
            training_dataset(prepared, train.tolist()),
            settings,
            model,
            max_seconds=protocol["stability"]["max_seconds"],
        )
        stability.summary.to_csv(out / "stability.csv", index=False)
        stability.predictions.to_csv(out / "stability-predictions.csv", index=False)
        stability.contributions.to_csv(out / "stability-contributions.csv", index=False)
        _json(out / "stability-provenance.json", stability.provenance)
        print("Training-only stability:\n", stability.summary.to_string(index=False), flush=True)

    criteria = protocol["criteria"]
    mean_gain = (
        intervals[
            (intervals.comparison == "3d_vs_mean")
            & (intervals.metric == "RMSE gain versus baseline")
        ]
        if not intervals.empty
        else pd.DataFrame()
    )
    assessment = {
        "scope": protocol["scope"],
        "checks": {
            "nested_q2": bool(primary_stats["cv_pooled_q2"] > criteria["nested_q2_gt"]),
            "test_r2": bool(primary_stats["test_r2"] > criteria["test_r2_gt"]),
            "permutation": bool(
                primary_stats["permutation_pvalue"] <= criteria["permutation_p_le"]
            ),
            "bootstrap_gain_lower_positive": bool(
                not mean_gain.empty
                and mean_gain.iloc[0].lower_95 > criteria["bootstrap_rmse_gain_lower_gt"]
            ),
            "all_training_stability_flags": bool(
                stability.summary.completed.all()
                and stability.summary.engineering_stability_flag.all()
            ),
            "post_alignment_leakage_controls": all(
                all(c.values())
                for c in [
                    leakage["held_out_labels"],
                    leakage["held_out_coordinates_after_alignment"],
                ]
            ),
            "independent_validation": False,
        },
        "primary": primary_stats,
        "historical_re_evaluation": historical_stats,
        "scaffold_groups_total": int(table.murcko.nunique()),
        "held_out_scaffold_groups": int(table.iloc[test].murcko.nunique()),
        "median_nearest_training_similarity": float(
            novelty.nearest_training_ecfp4_tanimoto.median()
        ),
        "all_primary_app_checks_pass": bool(primary.checks.passed.all()),
        "predictive_release": (
            "NO-GO: retrospective tutorial evidence is not independent validation"
        ),
        "automatic_preset_changes": False,
    }
    _json(out / "assessment.json", assessment)
    # Detect code/data changes during the run; results with a drifting implementation are invalid.
    if frozen["source_sha256"] != {
        str(p.relative_to(root)): provenance.file_sha256(p) for p in sorted(root.rglob("*.py"))
    }:
        raise ValueError("Source changed during execution; do not interpret this run as frozen.")
    _manifest(out)
    print(json.dumps(assessment, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.protocol, args.out)


if __name__ == "__main__":
    main()

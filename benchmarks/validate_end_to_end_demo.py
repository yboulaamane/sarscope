"""Measured-data worked examples, not a new blinded validation challenge.

Freeze before execution; retain every declared outcome. No external solver,
test-driven choice of split/preset, label shift or executable model pickle.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import zipfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from sklearn.metrics import r2_score, root_mean_squared_error
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.diagnostics import bootstrap_regression
from sarscope.analysis.regression import regression_split
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_campaign import _json, _manifest
from sarscope.field_design import field_model_signature
from sarscope.field_example import field_example_dataset
from sarscope.field_primary_example import archive_from_file, primary_mmff_settings
from sarscope.field_qsar import FieldQsarResult, prepare_field_series, run_field_qsar
from sarscope.field_qsar_report import field_qsar_archive
from sarscope.field_sensitivity import perturb_poses, training_dataset, training_fit
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[1]


def prepare(out: Path, archive: Path) -> None:
    out.mkdir(parents=True, exist_ok=False)
    main = archive_from_file(archive)
    generated = field_example_dataset("ccr5")
    model = ModelParams(split="random", cv_folds=3, test_fraction=0.25, seed=42)
    train, test = regression_split(
        np.zeros((75, 1)), np.full(75, "series"), np.full(75, "series"), model
    )
    # Sort explicit membership as required by the ordinary regression engine.
    train, test = sorted(train.tolist()), sorted(test.tolist())
    ref = min(
        train,
        key=lambda i: (
            generated.molecules[i].GetNumHeavyAtoms(),
            generated.table.iloc[i].molecule_id,
        ),
    )
    cases = [
        {
            "name": "thrombin_source_alignment",
            "source": main.source,
            "table_sha256": hashlib.sha256(main.table.to_csv(index=False).encode()).hexdigest(),
            "train": list(range(59)),
            "test": list(range(59, 88)),
            "fields": asdict(primary_mmff_settings()),
            "model": asdict(model),
        },
        {
            "name": "ccr5_generated_from_csv",
            "source": generated.source,
            "train": train,
            "test": test,
            "fields": asdict(
                replace(
                    primary_mmff_settings(),
                    alignment="constrained",
                    preserve_input_hydrogens=False,
                    assess_prealigned_core=True,
                    reference_index=ref,
                )
            ),
            "model": asdict(model),
        },
    ]
    for data, case in zip((main, generated), cases, strict=True):
        table = data.table[["molecule_id", "smiles", "pactivity"]].copy()
        table["split"] = "train"
        table.loc[case["test"], "split"] = "held_out"
        table.to_csv(out / f"{case['name']}_input.csv", index=False)
    _json(
        out / "protocol.json",
        {
            "scope": "Retrospective measured-data worked examples; not package certification",
            "previous_results_known": True,
            "selection": (
                "Thrombin chosen on prior positive MMFF evidence; "
                "CCR5 tests a new CSV-generated route"
            ),
            "no_test_driven_replacement": True,
            "all_declared_cases_retained": True,
            "source_archive": str(archive.resolve()),
            "source_archive_sha256": provenance.file_sha256(archive),
            "cases": cases,
            "acceptance": {
                "nested_q2_above": 0.5,
                "test_r2_above": 0.6,
                "permutation_p_at_most": 0.05,
                "beats_training_mean_rmse": True,
                "no_label_dependent_predictions": True,
                "numeric_export_replay_atol": 1e-8,
            },
            "bootstrap": {
                "repeats": 2000,
                "seed": 42,
                "method": "paired held-out Murcko scaffold-block resampling",
            },
            "sensitivity": "Training only; baseline, three 0.25 A/5 degree pose perturbations, "
            "half-grid shifts in x/y/z, common translation numerical control",
            "packages": provenance.package_versions(),
            "source_sha256": {
                str(p.relative_to(ROOT)): provenance.file_sha256(p)
                for p in sorted((ROOT / "src/sarscope").rglob("*.py"))
            }
            | {str(Path(__file__).relative_to(ROOT)): provenance.file_sha256(Path(__file__))},
            "input_sha256": {p.name: provenance.file_sha256(p) for p in out.glob("*_input.csv")},
        },
    )


def params(raw: dict) -> FieldQsarParams:
    value = raw.copy()
    for key in ("field_subset", "grid_phase"):
        value[key] = tuple(value[key])
    return FieldQsarParams(**value)


def numeric_replay(payload: bytes) -> np.ndarray:
    """Public export equation; does not deserialize code or call the fitted model."""
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with np.load(io.BytesIO(archive.read("fields.npz")), allow_pickle=False) as fields:
            mask = fields["retained_mask"]
            matrix = fields["fields"][:, mask].copy()
            if np.isinf(matrix).any():
                raise ValueError("Infinite fields cannot be replayed")
            matrix = np.where(np.isnan(matrix), fields["training_column_means"][mask], matrix)
            scaled = matrix / fields["column_scale"][mask]
            return (
                scaled @ fields["coefficients"].reshape(-1)
                + fields["prediction_at_scaled_zero"].item()
            )


def stability(result: FieldQsarResult, model: ModelParams, folder: Path) -> dict:
    """Same MMFF equations/settings throughout; no alternative charges or fitted choices."""
    dataset = training_dataset(result.prepared, result.regression.train_index.tolist())
    settings = replace(
        result.prepared.params,
        alignment="prealigned",
        reference_index=0,
        preserve_input_hydrogens=True,
        assess_prealigned_core=False,
        permutations=0,
    )
    variants = [("baseline", "baseline", dataset, settings)]
    for seed in (1101, 1102, 1103):
        variants.append(
            (
                f"pose_{seed}",
                "relative_pose",
                perturb_poses(dataset, seed=seed, translation=0.25, rotation_degrees=5),
                settings,
            )
        )
    for axis, phase in (("x", (0.5, 0.0, 0.0)), ("y", (0.0, 0.5, 0.0)), ("z", (0.0, 0.0, 0.5))):
        variants.append(
            (f"phase_{axis}", "grid_phase", dataset, replace(settings, grid_phase=phase))
        )
    variants.append(
        (
            "common_translation",
            "numerical_control",
            perturb_poses(
                dataset,
                seed=0,
                translation=0,
                rotation_degrees=0,
                common_translation=True,
                common_step=settings.spacing,
            ),
            settings,
        )
    )
    rows, predictions, baseline = [], [], None
    for name, kind, current, current_params in variants:
        print("Training-only sensitivity:", name, flush=True)
        try:
            fitted = training_fit(current, current_params, model)
            if baseline is None:
                baseline = fitted["prediction"]
            change = np.abs(fitted["prediction"] - baseline)
            p95, maximum = float(np.quantile(change, 0.95)), float(change.max())
            rows.append(
                {
                    "variant": name,
                    "kind": kind,
                    "completed": True,
                    "training_oof_q2": fitted["q2"],
                    "oof_change_p95": p95,
                    "oof_change_max": maximum,
                    "engineering_stability_flag": maximum <= 1e-8
                    if kind == "numerical_control"
                    else p95 <= 0.5 and maximum <= 1.0,
                }
            )
            for identifier, value in zip(
                current.table.molecule_id, fitted["prediction"], strict=True
            ):
                predictions.append(
                    {"variant": name, "molecule_id": identifier, "training_oof_prediction": value}
                )
        except Exception as exc:
            rows.append(
                {
                    "variant": name,
                    "kind": kind,
                    "completed": False,
                    "engineering_stability_flag": False,
                    "error": str(exc),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / "training_sensitivity.csv", index=False)
    pd.DataFrame(predictions).to_csv(folder / "sensitivity_predictions.csv", index=False)
    return {
        "summary": frame,
        "predictions": pd.DataFrame(predictions),
        "contributions": pd.DataFrame(),
        "fitted_model_signature": field_model_signature(result),
        "provenance": {
            "scope": "training_only",
            "settings": asdict(settings),
            "held_out_used": False,
            "held_out_rows_used": 0,
            "held_out_labels_used": False,
            "max_translation_A": 0.25,
            "rotation_degrees": 5,
            "kind": "offline MMFF sensitivity",
        },
    }


def run(out: Path) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    for name, expected in protocol["source_sha256"].items():
        if provenance.file_sha256(ROOT / name) != expected:
            raise ValueError("Source changed after freeze: " + name)
    for name, expected in protocol["input_sha256"].items():
        if provenance.file_sha256(out / name) != expected:
            raise ValueError("Input changed after freeze")
    archive = Path(protocol["source_archive"])
    if provenance.file_sha256(archive) != protocol["source_archive_sha256"]:
        raise ValueError("Source archive changed")
    summaries = []
    for case in protocol["cases"]:
        folder = out / case["name"]
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        summary = {"case": case["name"], "status": "started"}
        try:
            data = (
                archive_from_file(archive)
                if case["name"].startswith("thrombin")
                else (field_example_dataset("ccr5"))
            )
            if case["name"].endswith("from_csv"):
                # Actual CSV route: discard every supplied conformer, including the reference.
                data.molecules = [Chem.MolFromSmiles(s) for s in data.table.smiles]
            field_params, model = params(case["fields"]), ModelParams(**case["model"])
            print(case["name"], "prepare complete series", flush=True)
            prepared = prepare_field_series(data, field_params)
            prepared.alignment.report.to_csv(folder / "alignment.csv", index=False)
            print(case["name"], "nested model + 99 full-selection permutations", flush=True)
            with threadpool_limits(limits=1):
                result = run_field_qsar(
                    prepared, model, split_indices=(case["train"], case["test"])
                )
                result.training_stability = stability(result, model, folder)
                altered = replace(
                    data, table=data.table.copy(), measurements=data.measurements.copy()
                )
                altered.table.loc[case["test"], "pactivity"] += 100
                check = run_field_qsar(
                    replace(
                        prepared, dataset=altered, params=replace(field_params, permutations=0)
                    ),
                    model,
                    split_indices=(case["train"], case["test"]),
                )
            invariant = bool(
                np.allclose(
                    result.predictions.prediction, check.predictions.prediction, atol=1e-10, rtol=0
                )
                and np.allclose(
                    result.regression.cv_predictions,
                    check.regression.cv_predictions,
                    atol=1e-10,
                    rtol=0,
                )
            )
            groups = add_scaffolds(data.table).iloc[case["test"]].murcko.tolist()
            try:
                intervals = bootstrap_regression(result.regression, groups, repeats=2000, seed=42)
                intervals.to_csv(folder / "bootstrap.csv", index=False)
            except ValueError as exc:
                _json(folder / "bootstrap_unavailable.json", {"reason": str(exc)})
            payload = field_qsar_archive(result)
            (folder / "complete_evidence.zip").write_bytes(payload)
            replay_error = float(
                np.max(np.abs(numeric_replay(payload) - result.predictions.prediction))
            )
            with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
                for name in (
                    "statistics.csv",
                    "predictions.csv",
                    "validation_checks.csv",
                    "permutations.csv",
                    "field_contributions.csv",
                    "report.html",
                    "provenance.json",
                    "interpretation.json",
                ):
                    (folder / name).write_bytes(zipped.read(name))
            stat = result.statistics.set_index("algorithm").loc["field_pls"]
            test = result.predictions.iloc[case["test"]]
            mean_rmse = root_mean_squared_error(
                test.pactivity, np.full(len(test), data.table.iloc[case["train"]].pactivity.mean())
            )
            baseline = result.statistics.set_index("algorithm").loc["random_forest"]
            good = (
                stat.cv_pooled_q2 > 0.5
                and stat.test_r2 > 0.6
                and stat.permutation_pvalue <= 0.05
                and stat.test_rmse < mean_rmse
                and invariant
                and replay_error < 1e-8
            )
            summary.update(
                status="completed",
                n=len(data.table),
                train=len(case["train"]),
                test=len(case["test"]),
                nested_q2=float(stat.cv_pooled_q2),
                test_r2=float(stat.test_r2),
                test_rmse=float(stat.test_rmse),
                test_spearman=float(stat.test_spearman),
                permutation_p=float(stat.permutation_pvalue),
                permutations=len(result.permutations),
                mean_rmse=float(mean_rmse),
                rf_test_rmse=float(baseline.test_rmse),
                rf_test_r2=float(baseline.test_r2),
                in_domain=int(test.pca_box_in_domain.sum()),
                components=int(stat.selected_components),
                held_out_label_invariance=invariant,
                numeric_replay_max_error=replay_error,
                passes_worked_example_predictive_criteria=bool(good),
                all_validation_checks_pass=bool(result.checks.passed.all()),
                full_package_certified=False,
            )
            if abs(r2_score(test.pactivity, test.prediction) - stat.test_r2) > 1e-12:
                raise ValueError("Independent score recalculation differs")
        except Exception as exc:
            summary.update(
                status="failed", error=str(exc), passes_worked_example_predictive_criteria=False
            )
        summary["wall_s"] = time.monotonic() - start
        _json(folder / "assessment.json", summary)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(out / "summary.csv", index=False)
        print(summary, flush=True)
    _manifest(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    args = parser.parse_args()
    if args.action == "prepare":
        if args.archive is None:
            parser.error("prepare requires --archive with the separately licensed primary ZIP")
        prepare(args.out.resolve(), args.archive)
    else:
        run(args.out.resolve())

"""SARscope-only reproduction of a documented SEH steroid protocol.

Haga et al. (2025), doi:10.3390/ph18030440, Methods 4.4-4.7.
No Py-CoMSIA executable, module, processed matrix or model is used here.
Only source SDF data are fetched. Historical combined-set scaling is explicitly
benchmark-only, never added to the normal validation workflow.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.fields import GridSpec, compute_fields
from sarscope.analysis.regression import (
    _fit_field_pls,
    _predict,
    regression_cv_predictions,
    regression_metrics,
)
from sarscope.field_benchmark import verify_source
from sarscope.field_qsar import FieldDataset, _pooled_q2, prepare_field_series, run_field_qsar
from sarscope.params import FieldQsarParams, ModelParams

PROTOCOL_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks/field_qsar/independent-protocol.json"
)


def historical_standardise(
    matrix: np.ndarray, n_train: int, n_fields: int
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Own implementation of Methods 4.6; transductive, not validated hold-out scaling.

    This function's name and module keep it out of normal app preprocessing.
    One population SD per complete field, then separate train/test column
    centres. This interprets Methods 4.6; the centres were confirmed against
    frozen numerical outputs, not implementation source. No reference matrix
    is an input to this calculation.
    """
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Historical Gaussian fields must be a finite two-dimensional matrix.")
    if not 1 < n_train < len(matrix) or n_fields < 1 or matrix.shape[1] % n_fields:
        raise ValueError("Historical preprocessing needs train/test rows and equal field blocks.")
    result = matrix.copy()
    scales, train_means, test_means = [], [], []
    width = matrix.shape[1] // n_fields
    for start in range(0, matrix.shape[1], width):
        block = result[:, start : start + width]
        scale = float(block.std(ddof=0))
        scale = scale if scale > 0 else 1.0
        block /= scale
        train_mean, test_mean = block[:n_train].mean(0), block[n_train:].mean(0)
        block[:n_train] -= train_mean
        block[n_train:] -= test_mean
        scales.append(scale)
        train_means.append(train_mean)
        test_means.append(test_mean)
    return result, {
        "scale": np.array(scales),
        "train_mean": np.concatenate(train_means),
        "test_mean": np.concatenate(test_means),
    }


def dataset_from_sources(
    protocol: dict[str, Any], payloads: dict[str, bytes]
) -> tuple[FieldDataset, pd.DataFrame]:
    molecules, rows = [], []
    audit: list[dict[str, Any]] = []
    for split in ("train", "test"):
        info = protocol["data"][split]
        verify_source(payloads[split], info)
        valid, invalid = 0, 0
        for number, mol in enumerate(
            Chem.ForwardSDMolSupplier(io.BytesIO(payloads[split]), removeHs=False), start=1
        ):
            if mol is None:
                invalid += 1
                audit.append(
                    {"split": split, "record": number, "status": "unreadable source record"}
                )
                continue
            value = float(mol.GetProp(protocol["data"]["activity"]))
            if not np.isfinite(value):
                raise ValueError("Source activity is not finite.")
            identifier = mol.GetProp("name") if mol.HasProp("name") else mol.GetProp("_Name")
            valid += 1
            molecules.append(mol)
            rows.append(
                {
                    "molecule_id": identifier,
                    "smiles": Chem.MolToSmiles(Chem.RemoveHs(mol)),
                    "pactivity": value,
                }
            )
            audit.append(
                {
                    "split": split,
                    "record": number,
                    "molecule_id": identifier,
                    "pactivity": value,
                    "status": "retained",
                }
            )
        if (valid, invalid) != (info["valid"], info["invalid"]):
            raise ValueError("Frozen source record accounting differs.")
    table = pd.DataFrame(rows)
    measurements = table.copy()
    return FieldDataset(
        table,
        molecules,
        measurements,
        pd.DataFrame(),
        {"kind": "independent historical reproduction", "protocol": protocol},
    ), pd.DataFrame(audit)


def reproduce(
    protocol: dict[str, Any], payloads: dict[str, bytes], out: Path, include_native: bool = False
) -> dict[str, Any]:
    """Run the entire field-to-model calculation using SARscope-owned orchestration."""
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; preserve previous evidence.")
    if any(
        name.startswith(("src.PLSAnalysis", "src.MolecularFieldCalculator")) for name in sys.modules
    ):
        raise RuntimeError("External reference modules must not be loaded in an independent run.")
    out.mkdir(parents=True, exist_ok=True)
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    dataset, audit = dataset_from_sources(protocol, payloads)
    audit.to_csv(out / "input_audit.csv", index=False)
    settings = dict(protocol["fields"])
    settings["field_subset"] = tuple(settings["field_subset"])
    params = FieldQsarParams(**settings)
    prepared = prepare_field_series(dataset, params)
    # The paper's grid covers the full supplied cohort. Native validation below
    # deliberately uses its existing training-derived grid instead.
    grid = GridSpec.around(prepared.alignment.molecules, params)
    fields = compute_fields(prepared.alignment.molecules, grid, params)
    y = dataset.table.pactivity.to_numpy()
    n_train = protocol["data"]["train"]["valid"]
    matrix, scaling = historical_standardise(fields.matrix, n_train, len(params.field_subset))
    train, test = np.arange(n_train), np.arange(n_train, len(y))
    labels = np.full(n_train, "series", dtype=object)
    model_params = ModelParams(
        split="random", regression_cv="loo", field_qsar=params, regression_algorithms=("field_pls",)
    )
    with threadpool_limits(limits=1):
        filt, model = _fit_field_pls(matrix[train], y[train], model_params, labels, labels, None)
        chosen = int(model.selected_components_)
        conditional = replace(model_params, field_qsar=replace(params, fixed_components=chosen))
        loo, _, _ = regression_cv_predictions(
            "field_pls", matrix[train], y[train], labels, labels, conditional
        )
        prediction = _predict(model, filt.transform(matrix))
    train_r2, train_rmse, _ = regression_metrics(y[train], prediction[train])
    test_r2, test_rmse, test_spearman = regression_metrics(y[test], prediction[test])
    q2 = _pooled_q2(y[train], loo)
    reported = model.selection_table_.loc[
        model.selection_table_.components == chosen, "selection_q2"
    ].iloc[0]
    if abs(q2 - float(reported)) > 1e-10:
        raise RuntimeError("Selected-count LOO predictions disagree with component search.")
    model.selection_table_.to_csv(out / "component_search.csv", index=False)
    predictions = dataset.table.drop(columns="smiles").copy()
    predictions["split"] = ["train"] * n_train + ["test"] * len(test)
    predictions["prediction"] = prediction
    predictions["prediction_minus_observed"] = prediction - y
    predictions["loo_prediction"] = np.r_[loo, np.full(len(test), np.nan)]
    # Evaluation-only diagnostic: measured test labels never enter model fitting
    # or the genuine prediction column. This cannot be used prospectively.
    oracle_shift = float(y[test].mean() - y[train].mean())
    predictions["oracle_test_mean_diagnostic_NOT_prediction"] = np.nan
    predictions.loc[test, "oracle_test_mean_diagnostic_NOT_prediction"] = (
        prediction[test] + oracle_shift
    )
    oracle_r2, oracle_rmse, _ = regression_metrics(y[test], prediction[test] + oracle_shift)
    predictions.to_csv(out / "predictions.csv", index=False)
    pd.DataFrame(fields.charges).to_csv(out / "charges.csv", index=False)
    prepared.alignment.report.to_csv(out / "alignment_checks.csv", index=False)
    np.savez_compressed(
        out / "own_arrays.npz",
        raw_fields=fields.matrix,
        processed_fields=matrix,
        mean=scaling["train_mean"],
        scale=scaling["scale"],
        test_mean=scaling["test_mean"],
        support=filt.support_,
        coefficient=model.coef_,
        intercept=model.intercept_,
    )
    result = {
        "implementation": "SARscope only; no external QSAR software or processed arrays",
        "external_reference_program_used": False,
        "training": n_train,
        "test": len(test),
        "selected_components": chosen,
        "q2": q2,
        "train_r2": train_r2,
        "train_rmse": train_rmse,
        "test_r2": test_r2,
        "test_rmse": test_rmse,
        "test_spearman": test_spearman,
        "paper_q2": protocol["targets"]["q2"],
        "paper_train_r2": protocol["targets"]["train_r2"],
        "meets_declared_paper_tolerances": bool(
            abs(q2 - protocol["targets"]["q2"]) <= protocol["absolute_tolerances"]["q2"]
            and abs(train_r2 - protocol["targets"]["train_r2"])
            <= protocol["absolute_tolerances"]["train_r2"]
            and chosen == protocol["targets"]["components"]
        ),
        "grid": asdict(grid),
        "params": asdict(params),
        "model_params": asdict(model_params),
        "historical_transductive_preprocessing": True,
        "historical_oracle_diagnostic": {
            "test_label_mean_shift": oracle_shift,
            "r2_after_test_label_mean_shift": oracle_r2,
            "rmse_after_test_label_mean_shift": oracle_rmse,
            "is_model_prediction": False,
            "warning": "Uses measured test activities; not prospective/held-out validation. "
            "Not applied to the prediction column or native app.",
        },
        "independent_predictive_validation": False,
        "prealigned_source_frame_not_new_alignment_validation": True,
        "packages": provenance.package_versions(),
        "source_sha256": {
            str(path.relative_to(Path(__file__).parent)): provenance.file_sha256(path)
            for path in (
                Path(__file__),
                Path(__file__).parent / "analysis/fields.py",
                Path(__file__).parent / "analysis/align3d.py",
                Path(__file__).parent / "analysis/field_filter.py",
                Path(__file__).parent / "analysis/regression.py",
                Path(__file__).parent / "params.py",
                Path(__file__).parent / "field_qsar.py",
                Path(__file__).parent / "field_benchmark.py",
                Path(__file__).parent / "provenance.py",
            )
        },
    }
    (out / "assessment.json").write_text(json.dumps(result, indent=2) + "\n")
    if include_native:
        native = replace(
            params,
            **{
                key: value
                for key, value in protocol["native_comparison"].items()
                if key != "warning"
            },
        )
        with threadpool_limits(limits=1):
            native_result = run_field_qsar(
                replace(prepared, params=native),
                model_params,
                split_indices=(train.tolist(), test.tolist()),
            )
        native_result.statistics.to_csv(out / "native_statistics.csv", index=False)
        native_result.predictions.drop(columns="smiles").to_csv(
            out / "native_predictions.csv", index=False
        )
        native_result.checks.to_csv(out / "native_checks.csv", index=False)
    manifest = {
        str(path.relative_to(out)): provenance.file_sha256(path)
        for path in sorted(out.rglob("*"))
        if path.is_file()
    }
    (out / "manifest.sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline-data", type=Path)
    parser.add_argument("--include-native", action="store_true")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    payloads = {}
    for split in ("train", "test"):
        filename = protocol["data"][split]["file"]
        if args.offline_data:
            payloads[split] = (args.offline_data / filename).read_bytes()
        else:
            response = httpx.get(
                protocol["data"]["base_url"] + filename, timeout=30, follow_redirects=True
            )
            response.raise_for_status()
            payloads[split] = response.content
    print(json.dumps(reproduce(protocol, payloads, args.out, args.include_native), indent=2))


if __name__ == "__main__":
    main()

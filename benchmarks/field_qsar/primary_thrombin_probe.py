"""Adjudicate THR against primary supporting data and evaluate a frozen native profile.

Read chemical data only, never an external method's implementation source.
Primary data are CC BY-NC4.0 per the publisher's Figshare metadata, not MIT.
The historical partition has already been inspected: this is characterisation,
not a fresh or prospective validation challenge.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import tarfile
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.diagnostics import bootstrap_regression
from sarscope.analysis.regression import evaluate_regression
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.field_design import field_model_signature
from sarscope.field_qsar import FieldDataset, prepare_field_series, run_field_qsar
from sarscope.field_qsar_report import aligned_sdf, field_qsar_archive
from sarscope.field_sensitivity import run_sensitivity, training_dataset
from sarscope.field_source_quality import require_closed_shell
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[2]
PRIMARY_ARCHIVE_SHA256 = "e69edbb64048a243db5bce8700b198ce9a60b6f8fe5c87f30d863d1f99790ecf"
PRIMARY_SDF_SHA256 = "7e91820c51e6b9bb62e9b542cbc906cb0555cb461033fd1226f35d9ae9c1efff"
FIELDS = ("steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor")


def save(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item()) + "\n")


def source_data(archive: Path, later: Path) -> tuple[FieldDataset, pd.DataFrame]:
    if provenance.file_sha256(archive) != PRIMARY_ARCHIVE_SHA256:
        raise ValueError("Primary supporting archive hash differs; stop and re-audit.")
    with ZipFile(archive) as zipped:
        raw = zipped.read("qsar_sets/3dqsar_sd.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tar:
        member = tar.extractfile("./3dqsar_sd/thr_3dqsar.sd")
        assert member is not None
        sdf = member.read()
    if hashlib.sha256(sdf).hexdigest() != PRIMARY_SDF_SHA256:
        raise ValueError("Original aligned SDF hash differs.")
    primary = list(Chem.ForwardSDMolSupplier(io.BytesIO(sdf), removeHs=False))
    if len(primary) != 88 or any(m is None for m in primary):
        raise ValueError("Account for all88 original records; no silent exclusions.")
    by_id = {m.GetProp("_Name"): m for m in primary}
    if len(by_id) != 88:
        raise ValueError("Original compound identifiers are not unique.")
    # Preserve the already-frozen later-file order so CV fold membership does
    # not change when replacing the graphs with their original source records.
    expected = json.loads(
        (ROOT / "benchmarks/field_qsar/native-validation-protocol.json").read_text()
    )
    info = next(d for d in expected["datasets"] if d["name"] == "THR")
    mols, rows, audits = [], [], []
    for split in ("train", "test"):
        path = later / f"THR_{split}.sdf"
        if provenance.file_sha256(path) != info[split]["sha256"]:
            raise ValueError("Later reference data differ from the frozen files.")
        reference = list(Chem.SDMolSupplier(str(path), removeHs=False))
        assert len(reference) == info[split]["valid"] and all(m is not None for m in reference)
        for old in reference:
            old_id = old.GetProp("_Name")
            identifier = old_id.removesuffix("x")
            original = by_id[identifier]
            value = float(original.GetProp("ACTIVITY"))
            if value != float(old.GetProp("Activity")):
                raise ValueError("Activity changed between primary and later source.")
            if int(original.GetProp("SET")) != (1 if split == "train" else 2):
                raise ValueError("Original training/test membership differs.")
            a, b = Chem.RemoveHs(old), Chem.RemoveHs(original)
            if a.GetNumAtoms() != b.GetNumAtoms():
                raise ValueError("Heavy-atom counts differ; no coordinate transfer inferred.")
            distance = np.linalg.norm(
                a.GetConformer().GetPositions()[:, None] - b.GetConformer().GetPositions()[None, :],
                axis=2,
            )
            left, right = linear_sum_assignment(distance)
            if distance[left, right].max() != 0 or any(
                a.GetAtomWithIdx(int(i)).GetAtomicNum() != b.GetAtomWithIdx(int(j)).GetAtomicNum()
                for i, j in zip(left, right, strict=True)
            ):
                raise ValueError("Primary and later heavy atoms do not match exactly in the frame.")
            mapping = dict(zip(map(int, left), map(int, right), strict=True))
            old_bonds = {
                tuple(sorted((mapping[x.GetBeginAtomIdx()], mapping[x.GetEndAtomIdx()]))): str(
                    x.GetBondType()
                )
                for x in a.GetBonds()
            }
            new_bonds = {
                tuple(sorted((x.GetBeginAtomIdx(), x.GetEndAtomIdx()))): str(x.GetBondType())
                for x in b.GetBonds()
            }
            if old_bonds.keys() != new_bonds.keys():
                raise ValueError("Heavy-atom connectivity differs; needs further adjudication.")
            audits.append(
                {
                    "molecule_id": identifier,
                    "later_identifier": old_id,
                    "split": split,
                    "activity": value,
                    "source_SET": int(original.GetProp("SET")),
                    "heavy_atom_coordinate_max_change_A": float(distance[left, right].max()),
                    "heavy_atom_connectivity_identical": True,
                    "bond_orders_different": sum(old_bonds[k] != new_bonds[k] for k in old_bonds),
                    "later_radical_electrons": sum(
                        x.GetNumRadicalElectrons() for x in old.GetAtoms()
                    ),
                    "primary_radical_electrons": sum(
                        x.GetNumRadicalElectrons() for x in original.GetAtoms()
                    ),
                    "later_formal_charge": Chem.GetFormalCharge(old),
                    "primary_formal_charge": Chem.GetFormalCharge(original),
                }
            )
            mols.append(Chem.Mol(original))
            rows.append(
                {"molecule_id": identifier, "smiles": Chem.MolToSmiles(b), "pactivity": value}
            )
    table = pd.DataFrame(rows)
    require_closed_shell(mols, table.molecule_id.tolist())
    if table.smiles.duplicated().any() or table.molecule_id.duplicated().any():
        raise ValueError("Grouped duplicate validation is required; no silent deduplication.")
    return FieldDataset(
        table,
        mols,
        table.copy(),
        pd.DataFrame(),
        {
            "kind": "primary-data retrospective THR benchmark",
            "doi": "10.1021/jm0497141",
            "supplement_doi": "10.1021/jm0497141.s002",
            "download_url": "https://ndownloader.figshare.com/files/5158327",
            "original_series": "Bohm, Sturzebecher and Klebe, J Med Chem1999,42:458-477",
            "license": "CC-BY-NC-4.0 (publisher Figshare metadata)",
            "data_license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
            "primary_archive_sha256": PRIMARY_ARCHIVE_SHA256,
            "primary_sdf_sha256": PRIMARY_SDF_SHA256,
            "independent_validation": False,
            "design_comparability_review_required": True,
            "source_quality": (
                "Original closed-shell graphs, not guessed corrections to later exports"
            ),
            "endpoint": "original thrombin log activity, source ACTIVITY values unchanged",
            "partition": "source SET1 train / SET2 held-out; all original compounds retained",
            "graph_changes_for_preparation": False,
        },
    ), pd.DataFrame(audits)


def run(archive: Path, later: Path, out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory; retain all earlier runs.")
    out.mkdir(parents=True, exist_ok=True)
    dataset, audit = source_data(archive, later)
    audit.to_csv(out / "source_adjudication.csv", index=False)
    dataset.table.to_csv(out / "primary_thrombin_88.csv", index=False)
    common = dict(
        alignment="prealigned",
        assess_prealigned_core=False,
        preserve_input_hydrogens=True,
        field_subset=FIELDS,
        permutations=99,
        permutation_timeout=600,
        max_grid_points=80000,
        max_memory_mb=512,
        seed=42,
    )
    profiles = {
        "fivefield_1A_block_primary": FieldQsarParams(
            **common, spacing=1.0, field_scaling="block_std", component_selection="max_q2"
        ),
        "app_2A_raw_diagnostic": FieldQsarParams(**common),
    }
    model = ModelParams(
        split="random",
        cv_folds=3,
        test_fraction=0.33,
        oversample=False,
        leakage_audit=False,
        seed=42,
    )
    save(
        out / "protocol.json",
        {
            "frozen_utc": datetime.now(UTC).isoformat(),
            "source": dataset.source,
            "runner_sha256": provenance.file_sha256(Path(__file__)),
            "profiles": {name: asdict(p) for name, p in profiles.items()},
            "model": asdict(model),
            "train_ids": dataset.table.molecule_id.iloc[:59].tolist(),
            "test_ids": dataset.table.molecule_id.iloc[59:].tolist(),
            "selection": "Same profiles fixed before earlier THR scores; corrected inputs are "
            "the unchanged primary source, not an activity-guided graph repair",
            "stability": "Existing ten training-only diagnostics, primary profile only, max600s",
            "bootstrap": {
                "repeats": 1000,
                "seed": 20261018,
                "unit": "held-out Murcko scaffold blocks",
            },
            "source_sha256": {
                name: provenance.file_sha256(ROOT / "src/sarscope" / name)
                for name in (
                    "analysis/fields.py",
                    "analysis/field_filter.py",
                    "analysis/regression.py",
                    "field_qsar.py",
                    "field_source_quality.py",
                    "params.py",
                )
            },
            "packages": provenance.package_versions(),
            "prospective_or_blinded": False,
        },
    )
    split = (list(range(59)), list(range(59, 88)))
    scores = []
    for name, params in profiles.items():
        case = out / name
        case.mkdir()
        print("Running original-source", name, flush=True)
        prepared = prepare_field_series(dataset, params)
        (case / "primary_aligned.sdf").write_text(aligned_sdf(prepared))
        with threadpool_limits(limits=1):
            result = run_field_qsar(prepared, model, split_indices=split)
        for label in ("statistics", "predictions", "checks", "permutations"):
            getattr(result, label).to_csv(case / f"{label}.csv", index=False)
        save(case / "provenance.json", result.provenance)
        actual = result.statistics.query("algorithm == 'field_pls'").iloc[0].to_dict()
        actual.update(profile=name)
        scores.append(actual)
        print(
            name,
            "q2",
            actual["cv_pooled_q2"],
            "R2",
            actual["test_r2"],
            "RMSE",
            actual["test_rmse"],
            flush=True,
        )
        if name == "fivefield_1A_block_primary":
            groups = add_scaffolds(dataset.table).iloc[59:].murcko.tolist()
            bootstrap_regression(result.regression, groups, repeats=1000, seed=20261018).to_csv(
                case / "bootstrap.csv", index=False
            )
            labels = dataset.table.pactivity.to_numpy().copy()
            labels[59:] += 10
            with threadpool_limits(limits=1):
                shifted = evaluate_regression(
                    result.fields.matrix,
                    labels,
                    ["series"] * 88,
                    ["series"] * 88,
                    replace(model, field_qsar=params, regression_algorithms=("field_pls",)),
                    split_indices=split,
                )
            difference = float(
                np.max(np.abs(shifted.test_predictions - result.regression.test_predictions))
            )
            cv_difference = float(
                np.max(np.abs(shifted.cv_predictions - result.regression.cv_predictions))
            )
            if difference > 1e-10 or cv_difference > 1e-10:
                raise ValueError("Test-label invariance failed; do not release.")
            save(
                case / "test_label_invariance.json",
                {
                    "test_label_shift": 10.0,
                    "prediction_max_abs_change": difference,
                    "cv_prediction_max_abs_change": cv_difference,
                    "passed": True,
                },
            )
            print("Running training-only sensitivity diagnostics", flush=True)
            with threadpool_limits(limits=1):
                sensitivity = run_sensitivity(
                    training_dataset(prepared, list(range(59))), params, model, max_seconds=600
                )
            sensitivity.summary.to_csv(case / "stability.csv", index=False)
            sensitivity.predictions.to_csv(case / "stability_predictions.csv", index=False)
            result.training_stability = {
                "fitted_model_signature": field_model_signature(result),
                "summary": sensitivity.summary,
                "predictions": sensitivity.predictions,
                "contributions": sensitivity.contributions,
                "provenance": sensitivity.provenance,
            }
            result.provenance["training_stability"] = sensitivity.provenance
            save(case / "provenance.json", result.provenance)
        (case / "sarscope_evidence.zip").write_bytes(field_qsar_archive(result))
    pd.DataFrame(scores).to_csv(out / "summary.csv", index=False)
    save(
        out / "manifest.sha256.json",
        {
            str(p.relative_to(out)): provenance.file_sha256(p)
            for p in sorted(out.rglob("*"))
            if p.is_file() and p.name != "manifest.sha256.json"
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-archive", type=Path, required=True)
    parser.add_argument("--later-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.original_archive, args.later_data, args.out)

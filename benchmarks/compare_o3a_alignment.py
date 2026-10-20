"""Offline, fixed-reference O3A sensitivity experiment, not model selection.

Only RDKit's public alignment API is used. Standalone Open3DALIGN and
Open3DQSAR are separate black-box comparators, not this implementation.
Primary publisher structures retain their separate CC-BY-NC-4.0 licence.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import tarfile
import time
from dataclasses import asdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdMolAlign
from threadpoolctl import threadpool_limits

from sarscope.analysis.align3d import best_mapping
from sarscope.conformers import embed_ligand_conformer, validate_conformer_stereo
from sarscope.field_example import field_example_files
from sarscope.field_qsar import FieldDataset, prepare_field_series, run_field_qsar
from sarscope.params import FieldQsarParams, ModelParams

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs/3dqsar-validation-20261018"
ARCHIVE = RUNS / "original-sutherland/jm0497141_si_002.zip"
ARCHIVE_SHA = "e69edbb64048a243db5bce8700b198ce9a60b6f8fe5c87f30d863d1f99790ecf"
COHORTS = {
    "THR": ("thr", RUNS / "primary-thrombin-native-v2/fivefield_1A_block_primary"),
    "AChE": ("ache", RUNS / "primary-ache-reproducible-driver/AChE/gaussian_fivefield_primary"),
}
METHODS = ("original", "o3a_mmff", "o3a_crippen")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def graph(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def overlay(probe: Chem.Mol, reference: Chem.Mol, method: str) -> tuple[Chem.Mol, dict]:
    """Proper rigid transform only; never changes conformer or input objects."""
    if method not in METHODS[1:]:
        raise ValueError("Choose o3a_mmff or o3a_crippen")
    for mol in (probe, reference):
        if not mol.GetNumConformers() or not mol.GetConformer().Is3D():
            raise ValueError("O3A requires supplied 3D conformers")
        if not np.isfinite(mol.GetConformer().GetPositions()).all():
            raise ValueError("Non-finite input coordinates")
    start = time.monotonic()
    result, ref = Chem.Mol(probe), Chem.Mol(reference)
    if method == "o3a_mmff":
        if not all(AllChem.MMFFHasAllMoleculeParams(m) for m in (result, ref)):
            raise ValueError("MMFF typing unavailable; no fallback or row exclusion")
        alignment = rdMolAlign.GetO3A(result, ref, reflect=False)
    else:
        alignment = rdMolAlign.GetCrippenO3A(result, ref, reflect=False)
    rmsd, transform = alignment.Trans()
    rotation = np.asarray(transform)[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-7
    ):
        raise ValueError("Alignment did not produce a proper rigid rotation")
    AllChem.TransformMol(result, transform)
    validate_conformer_stereo(result, graph(probe))
    if graph(result) != graph(probe):
        raise ValueError("Alignment changed molecular identity")
    xyz = probe.GetConformer().GetPositions()
    aligned = result.GetConformer().GetPositions()
    before = np.linalg.norm(xyz[:, None] - xyz[None, :], axis=2)
    after = np.linalg.norm(aligned[:, None] - aligned[None, :], axis=2)
    distance_error = float(np.max(np.abs(before - after)))
    if distance_error > 1e-6:
        raise ValueError("Rigid alignment changed internal molecular distances")
    return result, {
        "score": float(alignment.Score()),
        "property_mapping_rmsd_A": float(rmsd),
        "matched_atoms": len(alignment.Matches()),
        "distance_error_A": distance_error,
        "rotation_determinant": float(np.linalg.det(rotation)),
        "identity_and_stereo_preserved": True,
        "elapsed_s": time.monotonic() - start,
    }


def native_inputs(name: str) -> tuple[pd.DataFrame, list, FieldQsarParams, dict]:
    stem, previous = COHORTS[name]
    if sha(ARCHIVE) != ARCHIVE_SHA:
        raise ValueError("Primary archive hash differs")
    with ZipFile(ARCHIVE) as zipped:
        compressed = zipped.read("qsar_sets/3dqsar_sd.tar.gz")
    with tarfile.open(fileobj=io.BytesIO(compressed), mode="r:gz") as tar:
        member = tar.extractfile(f"./3dqsar_sd/{stem}_3dqsar.sd")
        assert member is not None
        raw = member.read()
    molecules = list(Chem.ForwardSDMolSupplier(io.BytesIO(raw), removeHs=False))
    if any(m is None for m in molecules):
        raise ValueError("Source parse failed; no exclusions")
    lookup = {m.GetProp("_Name"): m for m in molecules}
    frozen = pd.read_csv(previous / "predictions.csv", dtype={"molecule_id": str})
    ordered = [lookup[str(identifier)] for identifier in frozen.molecule_id]
    if len(ordered) != len(molecules) or len(set(frozen.molecule_id)) != len(molecules):
        raise ValueError("Source membership differs from fixed cohort")
    for mol, row in zip(ordered, frozen.itertuples(), strict=True):
        if graph(mol) != graph(Chem.MolFromSmiles(row.smiles)):
            raise ValueError("Source graph differs from frozen cohort")
        if abs(float(mol.GetProp("ACTIVITY")) - row.pactivity) > 1e-9:
            raise ValueError("Source activity differs")
        if (mol.GetProp("SET") == "1") != (row.split == "train"):
            raise ValueError("Original SET membership differs")
    settings = json.loads((previous / "provenance.json").read_text())["field_qsar"]
    settings.update(permutations=0)
    settings["field_subset"] = tuple(settings["field_subset"])
    settings["grid_phase"] = tuple(settings["grid_phase"])
    params = FieldQsarParams(**settings)
    source = {
        "doi": "10.1021/jm0497141",
        "licence": "CC-BY-NC-4.0",
        "archive_sha256": ARCHIVE_SHA,
        "sdf_sha256": hashlib.sha256(raw).hexdigest(),
        "reference_id": str(frozen.iloc[0].molecule_id),
        "reference_policy": "First training molecule in previously frozen order; "
        "never activity-selected",
        "previous_predictions_sha256": sha(previous / "predictions.csv"),
        "field_params": asdict(params),
        "membership": frozen[["molecule_id", "split"]].to_dict("records"),
    }
    return frozen, ordered, params, source


def freeze(out: Path) -> None:
    if out.exists():
        raise FileExistsError("Use a new output directory; retain failed attempts")
    out.mkdir(parents=True)
    example = field_example_files("egfr")
    cohorts = {name: native_inputs(name)[3] for name in COHORTS}
    protocol = {
        "scope": "Retrospective fixed-reference geometry and model sensitivity, "
        "not independent validation",
        "python": platform.python_version(),
        "rdkit": rdBase.rdkitVersion,
        "routes": METHODS,
        "seed": 42,
        "permutations": 0,
        "permutation_note": "Omitted for bounded diagnostics; no new validation certification",
        "reference_selection": "Fixed before any fit; no test-driven route selection",
        "grid_policy": "Same enclosing training-only grid rule; "
        "alignment changes bounds/orientation",
        "charge_policy": "Gasteiger QSAR fields unchanged; "
        "MMFF/Crippen used ONLY to score alignment",
        "original_reproduction_tolerance": 1e-8,
        "cohorts": cohorts,
        "egfr": {
            "csv_sha256": hashlib.sha256(example["egfr_3d.csv"]).hexdigest(),
            "source": json.loads(example["source.json"]),
            "generation": "Canonical isomeric SMILES, seed 42, production conformer generator",
        },
        "implementation_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                Path(__file__),
                ROOT / "src/sarscope/analysis/align3d.py",
                ROOT / "src/sarscope/field_qsar.py",
                ROOT / "src/sarscope/params.py",
                ROOT / "src/sarscope/analysis/fields.py",
                ROOT / "src/sarscope/analysis/regression.py",
                ROOT / "tests/test_o3a_comparison.py",
            ]
        },
    }
    write_json(out / "protocol.json", protocol)


def geometry(out: Path) -> None:
    example = field_example_files("egfr")
    frame = pd.read_csv(io.BytesIO(example["egfr_3d.csv"]))
    ref_id = json.loads(example["source.json"])["reference_id"]
    params = FieldQsarParams()
    generated = [embed_ligand_conformer(graph(Chem.MolFromSmiles(s)), 42)[0] for s in frame.smiles]
    ref_index = list(frame.molecule_id).index(ref_id)
    reference = generated[ref_index]
    rows = []
    for method in METHODS[1:]:
        destination = out / f"egfr_{method}.sdf"
        writer = Chem.SDWriter(str(destination))
        for i, (mol, identifier) in enumerate(zip(generated, frame.molecule_id, strict=True)):
            row = {"route": method, "molecule_id": identifier}
            try:
                result, audit = overlay(mol, reference, method)
                # Keep the predetermined reference frame exactly unchanged.
                if i == ref_index:
                    result = Chem.Mol(reference)
                heavy, ref_heavy = Chem.RemoveHs(result), Chem.RemoveHs(reference)
                mapping, rmsd, _ = best_mapping(heavy, ref_heavy, params, fixed_frame=True)
                row.update(
                    audit,
                    core_rmsd_A=rmsd,
                    core_coverage=len(mapping) / heavy.GetNumHeavyAtoms(),
                    status="ok",
                )
                result.SetProp("_Name", str(identifier))
                writer.write(result)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            rows.append(row)
        writer.close()
    pd.DataFrame(rows).to_csv(out / "egfr_geometry.csv", index=False)


def model_fit(out: Path, name: str, method: str) -> None:
    protocol = json.loads((out / "protocol.json").read_text())
    if (
        sha(Path(__file__))
        != protocol["implementation_sha256"][str(Path(__file__).relative_to(ROOT))]
    ):
        raise ValueError("Driver changed after protocol freeze")
    folder = out / f"{name}_{method}"
    folder.mkdir(exist_ok=False)
    start = time.monotonic()
    try:
        frozen, molecules, params, source = native_inputs(name)
        train = list(np.flatnonzero(frozen.split.to_numpy() == "train"))
        test = list(np.flatnonzero(frozen.split.to_numpy() == "held_out"))
        if not test:
            raise ValueError("Fixed held-out partition missing")
        if 0 not in train:
            raise ValueError("Reference must be a training molecule")
        rows = []
        reference = Chem.Mol(molecules[0])
        if method != "original":
            transformed = []
            for i, (mol, identifier) in enumerate(zip(molecules, frozen.molecule_id, strict=True)):
                aligned, audit = overlay(mol, reference, method)
                transformed.append(Chem.Mol(reference) if i == 0 else aligned)
                rows.append({"molecule_id": identifier, **audit})
            molecules = transformed
        pd.DataFrame(rows).to_csv(folder / "geometry_audit.csv", index=False)
        writer = Chem.SDWriter(str(folder / "aligned.sdf"))
        for mol in molecules:
            writer.write(mol)
        writer.close()
        table = frozen[["molecule_id", "smiles", "pactivity"]].copy()
        dataset = FieldDataset(table, molecules, table.copy(), pd.DataFrame(), source)
        model = ModelParams(
            split="random",
            cv_folds=3,
            test_fraction=0.33,
            oversample=False,
            leakage_audit=False,
            seed=42,
        )
        with threadpool_limits(limits=1):
            result = run_field_qsar(
                prepare_field_series(dataset, params), model, split_indices=(train, test)
            )
        for filename, data in (
            ("predictions", result.predictions),
            ("statistics", result.statistics),
            ("checks", result.checks),
            ("charges", pd.DataFrame(result.fields.charges)),
        ):
            data.to_csv(folder / f"{filename}.csv", index=False)
        write_json(folder / "provenance.json", result.provenance)
        row = result.statistics.loc[result.statistics.algorithm == "field_pls"].iloc[0]
        summary = {
            "cohort": name,
            "route": method,
            "status": "ok",
            "n": len(table),
            "train": len(train),
            "test": len(test),
            "reference_id": source["reference_id"],
            "q2": float(row.cv_pooled_q2),
            "test_r2": float(row.test_r2),
            "test_rmse": float(row.test_rmse),
            "selected_components": int(row.selected_components),
            "grid": asdict(result.fields.grid),
            "model_status": result.status,
        }
        if method == "original":
            previous = pd.read_csv(COHORTS[name][1] / "statistics.csv").set_index("algorithm")
            differences = {
                key: abs(float(row[key]) - float(previous.loc["field_pls", key]))
                for key in ("cv_pooled_q2", "test_r2", "test_rmse")
            }
            summary["previous_absolute_differences"] = differences
            summary["original_reproduces"] = max(differences.values()) < 1e-8
    except Exception as exc:
        summary = {
            "cohort": name,
            "route": method,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
    summary["elapsed_s"] = time.monotonic() - start
    write_json(folder / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "geometry", "fit"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cohort", choices=tuple(COHORTS))
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.action == "freeze":
        freeze(args.out)
    elif args.action == "geometry":
        geometry(args.out)
    else:
        if args.cohort is None or args.method is None:
            parser.error("fit requires --cohort and --method")
        model_fit(args.out, args.cohort, args.method)

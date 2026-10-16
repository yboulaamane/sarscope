"""Compare independently executed field APIs; never read external method source.

The reference's runtime signatures/docstrings describe generate_grid and
calc_field. SARscope's equations remain literature-derived. This is field
parity, not full preprocessing/PLS or predictive validation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem

ROOT = Path(__file__).resolve().parent
FIELDS = ("steric", "electrostatic", "hydrophobic", "hbond_donor", "hbond_acceptor")


def worker(side: str, checkout: Path, out: Path) -> None:
    aligned = []
    for name, training in (("STEROIDS_train.sdf", True), ("STEROIDS_test.sdf", False)):
        for mol in Chem.SDMolSupplier(str(checkout / "sdfs" / name), removeHs=False):
            if mol is not None:
                aligned.append((mol, training))
    if len(aligned) != 31:
        raise ValueError("Expected all 31 valid source molecules; no residual exclusions.")
    metadata: dict[str, Any] = {
        "training": 21,
        "external": 10,
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name) for name in ("numpy", "rdkit", "scikit-learn")
        },
    }
    if side == "reference":
        sys.path.insert(0, str(checkout))
        # Execute the external public API in this worker only. No source inspection.
        from src.MolecularFieldCalculator import MolecularFieldCalculator
        from src.MolecularGridCalculator import MolecularGridCalculator

        spacing, shape, origin = MolecularGridCalculator().generate_grid(
            aligned, resolution=1.0, padding=4.0
        )
        fields = MolecularFieldCalculator().calc_field(aligned, spacing, shape, origin)
        arrays = {
            name: np.asarray(
                fields["train_fields"][f"{name}_field"] + fields["pred_fields"][f"{name}_field"]
            )
            for name in FIELDS
        }
        metadata.update(spacing=list(spacing), shape=list(shape), origin=[float(x) for x in origin])
        metadata["charge_output_available"] = all(
            atom.HasProp("_GasteigerCharge") for mol, _ in aligned for atom in mol.GetAtoms()
        )
        if metadata["charge_output_available"]:
            arrays["charges"] = np.array(
                [
                    float(atom.GetProp("_GasteigerCharge"))
                    for mol, _ in aligned
                    for atom in mol.GetAtoms()
                ]
            )
    else:
        sys.path.insert(0, str(ROOT.parents[1] / "src"))
        from sarscope.analysis.fields import CHARGE_MODELS, GridSpec, compute_fields
        from sarscope.params import FieldQsarParams

        metadata.update(json.loads((out / "reference.json").read_text()))
        grid = GridSpec(tuple(metadata["origin"]), tuple(metadata["shape"]), metadata["spacing"][0])
        params = FieldQsarParams(method="comsia", charge_model="gasteiger", alpha=0.3)
        molecules = [mol for mol, _ in aligned]
        result = compute_fields(molecules, grid, params)
        arrays = {
            name: result.matrix[:, i * grid.n_points : (i + 1) * grid.n_points]
            for i, name in enumerate(FIELDS)
        }
        arrays["charges"] = np.concatenate(
            [CHARGE_MODELS["gasteiger"](Chem.Mol(mol)) for mol in molecules]
        )
    np.savez_compressed(out / f"{side}.npz", **arrays)
    (out / f"{side}.json").write_text(json.dumps(metadata, indent=2) + "\n")


def compare_arrays(reference: Any, actual: Any) -> dict[str, Any]:
    rows = {}
    for name in FIELDS:
        expected, observed = reference[name], actual[name]
        if expected.shape != observed.shape:
            raise ValueError(f"Unmatched field columns: {name}")
        finite = bool(np.isfinite(expected).all() and np.isfinite(observed).all())
        rows[name] = {
            "values_compared": expected.size,
            "finite": finite,
            "max_absolute_error": float(np.max(np.abs(observed - expected))),
            "rmse": float(np.sqrt(np.mean((observed - expected) ** 2))),
            "pearson_r": float(np.corrcoef(expected.ravel(), observed.ravel())[0, 1]),
            "matches_at_declared_tolerance": bool(
                finite and np.allclose(observed, expected, rtol=1e-10, atol=1e-10)
            ),
        }
        # Post-hoc diagnostic, not a new pass criterion or production parameter.
        # Distinguish a global sign/unit convention from a different spatial field.
        denominator = float(np.sum(observed**2))
        if finite and denominator > 0:
            factor = float(np.sum(observed * expected) / denominator)
            difference = expected - factor * observed
            rows[name].update(
                posthoc_reference_over_sarscope_factor=factor,
                posthoc_scaled_max_absolute_error=float(np.max(np.abs(difference))),
                posthoc_scaled_rmse=float(np.sqrt(np.mean(difference**2))),
                spatial_shape_agrees_after_posthoc_scaling=bool(
                    np.allclose(factor * observed, expected, rtol=1e-10, atol=1e-10)
                ),
            )
    if "charges" in reference:
        rows["charges"] = {
            "values_compared": reference["charges"].size,
            "max_absolute_error": float(np.max(np.abs(actual["charges"] - reference["charges"]))),
            "matches_at_declared_tolerance": bool(
                np.allclose(actual["charges"], reference["charges"], rtol=0, atol=1e-12)
            ),
        }
    return rows


def run(python: Path, out: Path) -> dict[str, Any]:
    import httpx

    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new empty output directory.")
    out.mkdir(parents=True, exist_ok=True)
    out = out.absolute()
    protocol = json.loads((ROOT / "reference-selection.json").read_text())
    raw = httpx.get(protocol["archive_url"], timeout=40, follow_redirects=True)
    raw.raise_for_status()
    if hashlib.sha256(raw.content).hexdigest() != protocol["archive_sha256"]:
        raise ValueError("External archive hash mismatch.")
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    with tempfile.TemporaryDirectory(prefix="sarscope-field-parity-") as scratch:
        with tarfile.open(fileobj=io.BytesIO(raw.content), mode="r:gz") as archive:
            archive.extractall(scratch, filter="data")
        checkout = Path(scratch) / f"pycomsia-{protocol['commit']}"
        for side in ("reference", "sarscope"):
            with (
                (out / f"{side}.stdout.txt").open("w") as stdout,
                (out / f"{side}.stderr.txt").open("w") as stderr,
            ):
                subprocess.run(
                    [
                        str(python.absolute()),
                        str(Path(__file__).resolve()),
                        "--side",
                        side,
                        "--checkout",
                        str(checkout),
                        "--out",
                        str(out),
                    ],
                    cwd=scratch,
                    env=env,
                    stdout=stdout,
                    stderr=stderr,
                    check=True,
                    timeout=120,
                )
    with np.load(out / "reference.npz") as reference, np.load(out / "sarscope.npz") as actual:
        rows = compare_arrays(reference, actual)
    assessment = {
        "reference_commit": protocol["commit"],
        "archive_sha256": protocol["archive_sha256"],
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_fields_sha256": hashlib.sha256(
            (ROOT.parents[1] / "src/sarscope/analysis/fields.py").read_bytes()
        ).hexdigest(),
        "scope": "Raw fields only, with identical deposited explicit-H coordinates and grid. "
        "Both workers use the same comparator environment; production preparation is bypassed.",
        "tolerances": {"field_rtol": 1e-10, "field_atol": 1e-10, "charge_atol": 1e-12},
        "posthoc_scale_note": "Empirically inferred sign/scale factors are diagnostics, "
        "not literature-specified parameters or a claim of raw-field equivalence. "
        "They are not applied to SARscope production fields.",
        "fields": rows,
        "full_method_reproduction": False,
        "predictive_validation": False,
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
    parser.add_argument("--python", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--side", choices=["reference", "sarscope"])
    parser.add_argument("--checkout", type=Path)
    args = parser.parse_args()
    if args.side:
        worker(args.side, args.checkout, args.out)
    else:
        if args.python is None:
            parser.error("--python is required for the isolated comparator")
        print(json.dumps(run(args.python, args.out), indent=2))


if __name__ == "__main__":
    main()

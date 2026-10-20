"""Count hash-verified frozen field arrays without recomputing fields or models."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / "benchmarks/field_qsar/mmff-mif-independent-20261019/protocol.json"


def sha256(path: Path) -> str:
    """Read the entire file, including its array header, for identity checking."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def build_audit(protocol_path: Path, arrays_root: Path) -> dict[str, Any]:
    """Require the frozen hashes, molecule membership, field count and grid size."""
    protocol = json.loads(protocol_path.read_text())
    results_path = protocol_path.with_name("results.csv")
    with results_path.open(newline="") as handle:
        results = {row["case"]: row for row in csv.DictReader(handle)}
    cases = []
    seen = set()
    for case in protocol["cases"]:
        name = case["name"]
        if name in seen or name not in results or results[name]["status"] != "ok":
            raise ValueError(f"Duplicate, missing or unsuccessful case: {name}")
        seen.add(name)
        matrix_path = arrays_root / name / "native_fields.npy"
        digest = sha256(matrix_path)
        if digest != case["input_sha256"]["native_fields.npy"]:
            raise ValueError(f"Frozen array hash mismatch: {name}")
        n_molecules = len(case["train"]) + len(case["test"])
        membership = case["train"] + case["test"]
        if sorted(membership) != list(range(n_molecules)):
            raise ValueError(f"Incomplete or duplicated molecule partition: {name}")
        grid_shape = case["external_grid"]["shape"]
        if len(grid_shape) != 3 or any(value <= 0 for value in grid_shape):
            raise ValueError(f"Invalid three-dimensional grid: {name}")
        fields = case["params"]["field_subset"]
        if fields != ["steric", "electrostatic"]:
            raise ValueError(f"Expected the two frozen MMFF94 fields: {name}")
        grid_points = prod(grid_shape)
        expected_shape = (n_molecules, len(fields) * grid_points)
        matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
        if matrix.shape != expected_shape or int(results[name]["n"]) != n_molecules:
            raise ValueError(f"Matrix, grid or results membership mismatch: {name}")
        cases.append(
            {
                "case": name,
                "molecules": n_molecules,
                "fields": fields,
                "grid_shape": grid_shape,
                "grid_points_per_field": grid_points,
                "matrix_shape": list(matrix.shape),
                "scalar_values": int(matrix.size),
                "native_fields_relative_path": f"{name}/native_fields.npy",
                "native_fields_sha256": digest,
                "hash_matches_frozen_protocol": True,
            }
        )
    if not cases or seen != set(results):
        raise ValueError("Protocol and results must describe exactly the same nonempty case set")
    return {
        "schema_version": 1,
        "scope": "Post-hoc array-size audit of frozen independent MMFF94 comparisons",
        "protocol_sha256": sha256(protocol_path),
        "results_sha256": sha256(results_path),
        "count_definition": "molecules × fields × grid points; each compared scalar counted once",
        "cases": cases,
        "total_scalar_values": sum(case["scalar_values"] for case in cases),
        "arrays_verified_against_frozen_protocol": True,
        "fields_recomputed": False,
        "models_refitted": False,
        "original_protocol_and_results_modified": False,
        "raw_array_redistribution": "Not included; upstream data licences still apply",
    }


def main() -> None:
    """Write a separate audit; never overwrite frozen comparison evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--arrays-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.protocol, args.arrays_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as handle:
        handle.write(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "total_scalar_values": audit["total_scalar_values"]}))


if __name__ == "__main__":
    main()

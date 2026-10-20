"""Offline generated-alignment validation; no activity fitting or docking.

Run with a new output folder. Original measured EGFR example files and source
licensing remain unchanged. All records are included in source order, with the
previously documented reference (not a reference chosen by fitting activity).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem, rdBase

from sarscope.analysis.align3d import align_series, best_mapping
from sarscope.conformers import validate_conformer_stereo
from sarscope.field_example import field_example_files
from sarscope.params import FieldQsarParams

ROOT = Path(__file__).resolve().parents[1]
TEST_FILES = [
    "tests/test_generated_alignment_validation.py",
    "tests/test_alignment_worker.py",
    "tests/test_conformers.py",
]
WORKFLOW_FILES = [
    "src/sarscope/analysis/align3d.py",
    "src/sarscope/conformers.py",
    "src/sarscope/ligand_states.py",
    "src/sarscope/params.py",
    *TEST_FILES,
    "benchmarks/validate_generated_alignment.py",
]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def audit_molecule(
    mol: Any, reference: Any, expected: str, params: FieldQsarParams
) -> dict[str, Any]:
    graph = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    identity = graph == Chem.MolToSmiles(Chem.MolFromSmiles(expected), isomericSmiles=True)
    try:
        validate_conformer_stereo(mol, expected)
        stereo = True
    except ValueError:
        stereo = False
    xyz, ref_xyz = mol.GetConformer().GetPositions(), reference.GetConformer().GetPositions()
    mapping, _, _ = best_mapping(mol, reference, params, fixed_frame=True)
    # Independently calculate RMSD and mapping compatibility from exported atom
    # coordinates and graphs rather than trusting the AlignmentResult report.
    rmsd = float(np.sqrt(np.mean([np.sum((xyz[i] - ref_xyz[j]) ** 2) for i, j in mapping])))
    types = all(
        (a.GetAtomicNum(), a.GetIsAromatic(), a.GetHybridization())
        == (b.GetAtomicNum(), b.GetIsAromatic(), b.GetHybridization())
        for i, j in mapping
        for a, b in [(mol.GetAtomWithIdx(i), reference.GetAtomWithIdx(j))]
    )
    covalent = Chem.GetPeriodicTable()
    ratios = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        radii = sum(covalent.GetRcovalent(mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
        ratios.append(float(np.linalg.norm(xyz[i] - xyz[j]) / radii))
    return {
        "identity_preserved": bool(identity),
        "coordinate_stereo_preserved": stereo,
        "coordinates_finite": bool(np.isfinite(xyz).all()),
        "mapping_types_compatible": bool(types),
        "bond_ratio_min": min(ratios),
        "bond_ratio_max": max(ratios),
        "bond_geometry_sanity": bool(min(ratios) > 0.6 and max(ratios) < 1.45),
        "independent_core_rmsd_A": rmsd,
        "core_coverage_recalculated": len(mapping) / mol.GetNumHeavyAtoms(),
        "mapping": json.dumps(mapping),
    }


def run(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new evidence folder; do not overwrite a previous attempt")
    out.mkdir(parents=True, exist_ok=True)
    contents = field_example_files("egfr")
    source = json.loads(contents["source.json"])
    frame = pd.read_csv(io.BytesIO(contents["egfr_3d.csv"]))
    ids = list(frame.molecule_id.astype(str))
    reference_index = ids.index(source["reference_id"])
    common = FieldQsarParams(reference_index=reference_index)
    protocol = {
        "scope": "Generated alignment implementation; not an experimental binding-pose study",
        "source": {
            "target": source["target_id"],
            "assay": source["assay"],
            "doi": source["document"]["doi"],
            "license": source["license"],
            "attribution": source["attribution"],
            "csv_sha256": source["files_sha256"]["egfr_3d.csv"],
            "records": len(frame),
            "reference_id": source["reference_id"],
            "selection": "All 50 bundled records in source order; no potency/score selection",
            "activities_used_for_alignment": False,
        },
        "routes": ["rigid_mcs", "constrained"],
        "base_settings": asdict(common),
        "criteria": {
            "molecular_graph_and_coordinate_stereo": "unchanged",
            "finite_coordinates_and_compatible_mapping": True,
            "bond_length_over_covalent_radii_sum": [0.6, 1.45],
            "report_rmsd_and_coverage_absolute_tolerance": 1e-10,
            "constrained_fixed_core_rmsd_A_max": 0.0002,
            "constrained_reference_coordinate_displacement_A_max": 0.0002,
            "rigid_reference_coordinates": "bitwise unchanged",
            "rerun_coordinate_tolerance": "bitwise identical in this pinned environment",
            "replay_records": "first four source records, including the fixed reference",
            "default_core_quality_warnings": "reported, never silently treated as passing",
        },
        "oracle_tests": TEST_FILES,
        "source_sha256": {name: digest(ROOT / name) for name in WORKFLOW_FILES},
        "runtime": {
            "python": platform.python_version(),
            "rdkit": rdBase.rdkitVersion,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    # Write/hash the plan BEFORE executing any new test or measured alignment.
    write_json(out / "protocol.json", protocol)
    plan_hash = digest(out / "protocol.json")
    (out / "source.json").write_bytes(contents["source.json"])
    (out / "input.csv").write_bytes(contents["egfr_3d.csv"])
    junit = out / "alignment-tests.xml"
    command = [sys.executable, "-m", "pytest", *TEST_FILES, "-q", f"--junitxml={junit}"]
    check = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180)
    (out / "alignment-tests.log").write_text(check.stdout + check.stderr)
    print(check.stdout, flush=True)
    rows, summaries = [], []
    for route in protocol["routes"]:
        folder = out / route
        folder.mkdir()
        params = replace(common, alignment=route)
        mols = [Chem.MolFromSmiles(smiles) for smiles in frame.smiles]
        started = time.monotonic()
        try:
            result = align_series(mols, ids, params)
            reference = result.molecules[reference_index]
            writer = Chem.SDWriter(str(folder / "aligned.sdf"))
            for molecule in result.molecules:
                writer.write(molecule)
            writer.close()
            result.report.to_csv(folder / "alignment.csv", index=False)
            route_rows = []
            for identifier, expected, molecule, reported in zip(
                ids, frame.smiles, result.molecules, result.report.itertuples(), strict=True
            ):
                item = audit_molecule(molecule, reference, expected, params)
                item.update(molecule_id=identifier, route=route)
                item["reported_rmsd_matches"] = bool(
                    abs(item["independent_core_rmsd_A"] - reported.core_rmsd_A) < 1e-10
                )
                item["reported_coverage_matches"] = bool(
                    abs(item["core_coverage_recalculated"] - reported.core_coverage) < 1e-10
                )
                item["fixed_core_within_tolerance"] = bool(
                    route != "constrained" or item["independent_core_rmsd_A"] < 0.0002
                )
                item["default_quality_warning"] = reported.warning
                route_rows.append(item)
            audited = pd.DataFrame(route_rows)
            audited.to_csv(folder / "coordinate-audit.csv", index=False)
            # Same reference and seed, changed activities and input order.
            # Replay the first four records; reference membership was fixed above.
            replay_ids = ids[:4]
            replay_mols = [Chem.MolFromSmiles(s) for s in frame.smiles.iloc[:4]]
            for index, mol in enumerate(replay_mols):
                mol.SetDoubleProp("pactivity", 1000 - index * 100)
            order = list(reversed(range(4)))
            reordered = [replay_ids[i] for i in order]
            repeated = align_series(
                [replay_mols[i] for i in order],
                reordered,
                replace(params, reference_index=reordered.index(source["reference_id"])),
            )
            replay_same = all(
                np.array_equal(
                    mol.GetConformer().GetPositions(),
                    result.molecules[ids.index(mol.GetProp("_Name"))].GetConformer().GetPositions(),
                )
                for mol in repeated.molecules
            )
            keys = [
                "identity_preserved",
                "coordinate_stereo_preserved",
                "coordinates_finite",
                "mapping_types_compatible",
                "bond_geometry_sanity",
                "reported_rmsd_matches",
                "reported_coverage_matches",
                "fixed_core_within_tolerance",
            ]
            summary = {
                "route": route,
                "completed": True,
                "molecules": len(audited),
                "records_dropped": len(frame) - len(audited),
                "implementation_checks_pass": bool(audited[keys].all().all() and replay_same),
                "checks_all_pass": {name: bool(audited[name].all()) for name in keys},
                "replay_activity_and_order_invariant": replay_same,
                "default_quality_warnings": int(audited.default_quality_warning.astype(bool).sum()),
                "max_core_rmsd_A": float(audited.independent_core_rmsd_A.max()),
                "min_core_coverage": float(audited.core_coverage_recalculated.min()),
                "seconds": time.monotonic() - started,
            }
            rows.extend(route_rows)
        except Exception as exc:
            summary = {
                "route": route,
                "completed": False,
                "implementation_checks_pass": False,
                "error": str(exc),
                "seconds": time.monotonic() - started,
            }
        summaries.append(summary)
        write_json(folder / "assessment.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    pd.DataFrame(rows).to_csv(out / "coordinate-audit.csv", index=False)
    # Refuse a successful conclusion if any tested module or the protocol drifted.
    assert digest(out / "protocol.json") == plan_hash
    assert all(digest(ROOT / name) == sha for name, sha in protocol["source_sha256"].items())
    result = {
        "protocol_sha256": plan_hash,
        "test_exit_code": check.returncode,
        "routes": summaries,
        "implementation_validation_pass": check.returncode == 0
        and all(r["completed"] and r["implementation_checks_pass"] for r in summaries),
        "interpretation": "Geometry/implementation validation only; actual bound alignment is an "
        "input/method assumption. Default RMSD warnings and failed attempts remain visible. "
        "No activity-model fitting, docking or threshold relaxation occurred.",
    }
    write_json(out / "assessment.json", result)
    write_json(
        out / "manifest.sha256.json",
        {str(f.relative_to(out)): digest(f) for f in sorted(out.rglob("*")) if f.is_file()},
    )
    if not result["implementation_validation_pass"]:
        raise SystemExit("Validation did not pass; retained all artifacts for diagnosis")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())

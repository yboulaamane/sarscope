"""Check deposited SDFs against the tutorial without trusting producer audit flags.

This is a source/derivative comparison, NOT verification of the experimental
assay, protonation state, binding conformation or predictive model.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope.field_example import field_example_files


def verify(source_dir: Path | None) -> dict:
    assets = field_example_files("ccr5")
    metadata = json.loads(assets["source.json"])
    audit = pd.read_csv(io.BytesIO(assets["preparation.csv"]))
    raw = {}
    for name, digest in metadata["source_files_sha256"].items():
        if source_dir:
            payload = (source_dir / name).read_bytes()
        else:
            response = httpx.get(metadata["source_urls"][name], timeout=60, follow_redirects=True)
            response.raise_for_status()
            payload = response.content
        if hashlib.sha256(payload).hexdigest() != digest:
            raise ValueError("Deposited source SDF hash differs from the pinned data.")
        raw[name] = list(Chem.ForwardSDMolSupplier(io.BytesIO(payload), removeHs=False))
    prepared = {
        m.GetProp("_Name"): m
        for m in Chem.ForwardSDMolSupplier(
            io.BytesIO(assets["ccr5_3d_aligned.sdf"]), removeHs=False
        )
    }
    if len(audit) != 75 or len(prepared) != 75 or sum(map(len, raw.values())) != 75:
        raise ValueError("Expected all 75 source and prepared records, without row removal.")
    changes = 0
    for row in audit.itertuples():
        original = raw[row.source_file][row.source_record - 1]
        product = prepared[row.molecule_id]
        if original is None or product is None:
            raise ValueError("A source or prepared record cannot be parsed.")
        if not (
            float(original.GetProp("Activity"))
            == row.source_activity
            == float(product.GetProp("pactivity"))
        ):
            raise ValueError("A deposited activity was changed.")
        original, product = Chem.RemoveHs(original), Chem.RemoveHs(product)
        if original.GetNumAtoms() != product.GetNumAtoms():
            raise ValueError("Heavy atom count changed.")
        a, b = original.GetConformer().GetPositions(), product.GetConformer().GetPositions()
        distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        mapping = distance.argmin(1)
        if len(set(mapping)) != len(mapping) or distance[np.arange(len(mapping)), mapping].max():
            raise ValueError("Exact one-to-one heavy-atom coordinates were not preserved.")
        for i, j in enumerate(mapping):
            if (
                original.GetAtomWithIdx(i).GetAtomicNum()
                != product.GetAtomWithIdx(int(j)).GetAtomicNum()
            ):
                raise ValueError("An atom's element changed.")
        if original.GetNumBonds() != product.GetNumBonds():
            raise ValueError("Heavy-atom bond count changed.")
        for bond in original.GetBonds():
            other = product.GetBondBetweenAtoms(
                int(mapping[bond.GetBeginAtomIdx()]), int(mapping[bond.GetEndAtomIdx()])
            )
            if other is None or other.GetBondType() != bond.GetBondType():
                raise ValueError("Heavy-atom connectivity or bond type changed.")
        changes += Chem.GetFormalCharge(original) != Chem.GetFormalCharge(product)
    return {
        "source_files_sha256": metadata["source_files_sha256"],
        "prepared_sdf_sha256": hashlib.sha256(assets["ccr5_3d_aligned.sdf"]).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "records_checked": len(audit),
        "activities_unchanged": True,
        "heavy_atom_elements_bonds_and_coordinates_identical": True,
        "formal_charge_changed_records": changes,
        "method": "Exact heavy-atom coordinate correspondence, independently of producer flags",
        "experimental_assay_or_pose_validation": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Offline deposited SDF directory")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Retain previous source checks; choose a new output file.")
    result = verify(args.source_dir)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

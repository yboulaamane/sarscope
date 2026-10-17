"""Build the measured CCR5 tutorial through the ordinary uploaded-SDF workflow.

Only publisher-deposited data and its licence are fetched, not model code.
Primary settings are the app defaults; all compounds and all repeat-split
outcomes are retained. This demonstration is not a new independent benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from rdkit import Chem
from sorbent.chem.parse import standardize
from threadpoolctl import threadpool_limits

from sarscope import provenance
from sarscope.analysis.fields import COMSIA_FIELDS
from sarscope.field_qsar import (
    dataset_from_upload,
    prepare_field_series,
    run_field_qsar,
    upload_table,
)

# Kept with the existing frozen source identity; do not change to a moving branch.
COMMIT = "0b30179ecc20aac5a003577b4437399a8fb20bca"
BASE = f"https://raw.githubusercontent.com/clhaga/pycomsia/{COMMIT}/"
SOURCES = {
    "CCR5_train.sdf": "afa97e9a19c65c8d19523233016f3f065ab72cd4a51030b6ce79196290d489f5",
    "CCR5_test.sdf": "ed0d0e29e7d933034f97025780efb2be5e3ac9ef6578f4cd2a18247fe1825ce9",
}
REPEAT_SEEDS = (7, 19, 73, 101)  # diagnostic repeats, never candidate winning seeds


def prepared_source(payloads: dict[str, bytes]) -> tuple[bytes, pd.DataFrame]:
    """Use existing curation without transferring coordinates to a guessed graph.

    Source amines are protonated. The default upload workflow represents
    neutral parents. Make that transformation explicit *before* uploading,
    verify unchanged heavy-atom elements/bonds/coordinates by atom provenance,
    and record the charge/stereo representation change. H coordinates are
    regenerated later by the existing field workflow, not claimed experimental.
    """
    stream = io.StringIO()
    writer = Chem.SDWriter(stream)
    rows = []
    for filename, digest in SOURCES.items():
        data = payloads[filename]
        if hashlib.sha256(data).hexdigest() != digest:
            raise ValueError(f"Source checksum mismatch: {filename}")
        for record, mol in enumerate(
            Chem.ForwardSDMolSupplier(io.BytesIO(data), removeHs=False), 1
        ):
            if mol is None or any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
                raise ValueError(
                    "Unreadable/radical source; no compound was discarded or repaired."
                )
            name = mol.GetProp("name") if mol.HasProp("name") else mol.GetProp("_Name")
            activity = float(mol.GetProp("Activity"))
            original = Chem.RemoveHs(mol)
            for atom in original.GetAtoms():
                atom.SetIntProp("_source_index", atom.GetIdx())
            prepared = standardize(original, canonical_tautomer=False)
            Chem.AssignStereochemistry(prepared, cleanIt=True, force=True)
            if prepared.GetNumAtoms() != original.GetNumAtoms():
                raise ValueError("Standardisation changed the heavy-atom count.")

            def bonds(molecule: Any) -> set[tuple[int, int, str]]:
                return {
                    (
                        *sorted(
                            (
                                b.GetBeginAtom().GetIntProp("_source_index"),
                                b.GetEndAtom().GetIntProp("_source_index"),
                            )
                        ),
                        str(b.GetBondType()),
                    )
                    for b in molecule.GetBonds()
                }

            if bonds(prepared) != bonds(original):
                raise ValueError("Standardisation changed heavy-atom bonds; no guessed remapping.")
            for atom in prepared.GetAtoms():
                source_index = atom.GetIntProp("_source_index")
                if atom.GetAtomicNum() != original.GetAtomWithIdx(source_index).GetAtomicNum():
                    raise ValueError("Heavy-atom element changed.")
                np.testing.assert_array_equal(
                    prepared.GetConformer().GetPositions()[atom.GetIdx()],
                    original.GetConformer().GetPositions()[source_index],
                )
            prepared.SetProp("_Name", name)
            prepared.SetDoubleProp("pactivity", activity)
            writer.write(prepared)
            rows.append(
                {
                    "molecule_id": name,
                    "source_file": filename,
                    "source_record": record,
                    "original_partition": "train" if "train" in filename else "test",
                    "source_activity": activity,
                    "source_smiles": Chem.MolToSmiles(original),
                    "prepared_smiles": Chem.MolToSmiles(prepared),
                    "source_formal_charge": Chem.GetFormalCharge(original),
                    "prepared_formal_charge": Chem.GetFormalCharge(prepared),
                    "heavy_atoms": prepared.GetNumHeavyAtoms(),
                    "heavy_atom_bonds_preserved": True,
                    "heavy_atom_coordinates_preserved": True,
                    "coordinate_max_change_A": 0.0,
                }
            )
    writer.close()
    audit = pd.DataFrame(rows)
    if (
        len(audit) != 75
        or not audit.molecule_id.is_unique
        or not np.isfinite(audit.source_activity).all()
    ):
        raise ValueError("Source cohort differs; use all 75 unique, finite records.")
    return stream.getvalue().encode(), audit


def build(payloads: dict[str, bytes], licence: bytes, out: Path) -> None:
    from sarscope.params import FieldQsarParams, ModelParams

    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a new empty directory; do not overwrite existing evidence.")
    sdf, audit = prepared_source(payloads)
    raw, molecules = upload_table(sdf, ".sdf")
    dataset = dataset_from_upload(
        raw,
        molecules,
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="CCR5 deposited log activity",
        source_sha256=hashlib.sha256(sdf).hexdigest(),
    )
    if len(dataset.table) != 75:
        raise ValueError(
            "Upload curation changed cohort membership; no replacement example written."
        )
    reference_index = min(
        range(len(dataset.molecules)),
        key=lambda i: (
            dataset.molecules[i].GetNumHeavyAtoms(),
            str(dataset.table.iloc[i].molecule_id),
        ),
    )
    source = {
        "kind": "bundled_measured_3d_example",
        "example_id": "ccr5",
        "target_name": "CCR5",
        "compounds": 75,
        "endpoint": "CCR5 deposited log activity",
        "conversion": (
            "Deposited Activity values unchanged; no new IC50/Ki conversion or imputation."
        ),
        "assay_comparability": "Historical benchmark endpoint; not independently assay-adjudicated",
        "design_comparability_review_required": True,
        "reference_id": str(dataset.table.iloc[reference_index].molecule_id),
        "reference_policy": (
            "Fewest heavy atoms, ID tie-break, independent of activities. A smaller fixed "
            "reference bounds the MCS QC search; no supplied coordinates are realigned."
        ),
        "repository": "https://github.com/clhaga/pycomsia",
        "commit": COMMIT,
        "publication": "https://doi.org/10.3390/ph18030440",
        "original_publication": "https://doi.org/10.1007/s00894-007-0173-z",
        "attribution": (
            "Aher et al. (2007); aligned data deposited by Haga et al. (2025), clhaga/pycomsia."
        ),
        "license": "CC-BY-SA-4.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
        "source_files_sha256": SOURCES,
        "source_license_sha256": hashlib.sha256(licence).hexdigest(),
        "source_urls": {name: BASE + "sdfs/" + name for name in SOURCES},
        "selection": (
            "All 63 original training and 12 original test records; "
            "none removed by potency or prediction."
        ),
        "tutorial_split": (
            "New default seed-42 random 56/19 split of the full public cohort; "
            "not the historical 63/12 partition."
        ),
        "independent_validation": False,
        "preparation": (
            "Existing neutral-parent standardisation, no canonical tautomer pass; heavy-atom "
            "elements, bonds and coordinates verified unchanged by atom provenance. Source "
            "protonation/redundant stereo representation changes recorded per compound. "
            "Hydrogens regenerated by normal field preparation. No measured activity changed."
        ),
        "coordinate_assumption": (
            "Publisher-deposited pre-alignment; not crystallographic binding poses "
            "or a validated protonation state."
        ),
        "builder_sha256": provenance.file_sha256(Path(__file__)),
    }
    dataset.source = source
    settings = FieldQsarParams(
        alignment="prealigned", field_subset=COMSIA_FIELDS, reference_index=reference_index
    )
    model = ModelParams(split="random", test_fraction=0.25, cv_folds=3, seed=42)
    with threadpool_limits(limits=1):
        prepared = prepare_field_series(dataset, settings)
        primary = run_field_qsar(prepared, model)
        repeats = []
        for seed in REPEAT_SEEDS:
            repeated = run_field_qsar(
                replace(prepared, params=replace(settings, seed=seed, permutations=0)),
                replace(model, seed=seed),
            )
            stats = repeated.statistics.query("algorithm == 'field_pls'").iloc[0]
            repeats.append(
                {
                    "seed": seed,
                    "cv_pooled_q2": float(stats.cv_pooled_q2),
                    "test_r2": float(stats.test_r2),
                    "test_rmse": float(stats.test_rmse),
                }
            )
            print("Split diagnostic:", repeats[-1], flush=True)
    verification = {
        "settings": asdict(settings),
        "model_settings": asdict(model),
        "statistics": json.loads(primary.statistics.to_json(orient="records")),
        "checks": json.loads(primary.checks.to_json(orient="records")),
        "status": primary.status,
        "packages": provenance.package_versions(),
        "split_indices": primary.provenance["split_indices"],
        "workflow_sha256": primary.provenance["implementation_sha256"],
        "repeat_splits": repeats,
        "primary_seed_changed_after_repeats": False,
        "warning": (
            "Within-series public-data tutorial, not an independent validation. "
            "Borderline/failed gates remain failed. RF baseline is not the 3D model."
        ),
        "development_attempts": [
            "Unmodified charged source refused by upload graph-consistency safeguard "
            "before modelling.",
            "Explicit neutral-parent derivative refused by duplicate symmetry-product count "
            "before modelling.",
            "Distinct-mapping budget correction; default seed-42 result retained without tuning.",
            "An initial decorated reference hit the MCS time budget; "
            "reference now chosen by fewest heavy atoms, "
            "without moving coordinates or tuning scores.",
        ],
    }
    blobs = {
        "ccr5_3d.csv": dataset.table[["molecule_id", "smiles", "pactivity"]]
        .to_csv(index=False)
        .encode(),
        "ccr5_3d_aligned.sdf": sdf,
        "measurements.csv": dataset.measurements.to_csv(index=False).encode(),
        "preparation.csv": audit.to_csv(index=False).encode(),
        "alignment.csv": prepared.alignment.report.to_csv(index=False).encode(),
        "predictions.csv": primary.predictions.to_csv(index=False).encode(),
        "verification.json": (json.dumps(verification, indent=2, allow_nan=False) + "\n").encode(),
        "LICENSE.txt": licence,
    }
    source["files_sha256"] = {
        name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()
    }
    blobs["source.json"] = (json.dumps(source, indent=2, allow_nan=False) + "\n").encode()
    out.mkdir(parents=True, exist_ok=True)
    for name, data in blobs.items():
        (out / name).write_bytes(data)
    print(
        primary.statistics[["algorithm", "cv_pooled_q2", "test_r2", "test_rmse"]].to_string(
            index=False
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--offline-source-dir", type=Path)
    args = parser.parse_args()
    if args.offline_source_dir:
        payloads = {name: (args.offline_source_dir / name).read_bytes() for name in SOURCES}
        licence = (args.offline_source_dir / "LICENSE.txt").read_bytes()
    else:
        payloads = {}
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            for name in SOURCES:
                response = client.get(BASE + "sdfs/" + name)
                response.raise_for_status()
                payloads[name] = response.content
            response = client.get(BASE + "LICENSE")
            response.raise_for_status()
            licence = response.content
    build(payloads, licence, args.out)


if __name__ == "__main__":
    main()

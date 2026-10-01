"""Private isolated worker for pair docking. Invoke through run_pair_docking."""

from __future__ import annotations

import hashlib
import io
import json
import math
import re
import subprocess
import sys
import warnings
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sarscope.docking import DockingError, DockingSettings, validate_receptor_pair
from sarscope.ligand_states import validate_ligand_state


def _phase(directory: Path, message: str) -> None:
    (directory / "progress.json").write_text(json.dumps({"phase": message}))


def _prepare_ligand(smiles: str, seed: int) -> tuple[Any, str]:
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    from rdkit import Chem
    from rdkit.Chem import AllChem

    molecule = Chem.MolFromSmiles(smiles)
    validate_ligand_state(smiles, smiles)
    molecule = Chem.AddHs(molecule)
    embedding = AllChem.ETKDGv3()
    embedding.randomSeed = seed
    embedding.numThreads = 1
    if AllChem.EmbedMolecule(molecule, embedding) != 0:
        raise DockingError("Could not generate a 3D conformer for one ligand.")
    if not AllChem.UFFHasAllMoleculeParams(molecule):
        raise DockingError(
            "UFF parameters are missing for this ligand; preparation is not supported."
        )
    convergence = AllChem.UFFOptimizeMolecule(molecule, maxIters=500)
    if convergence:
        raise DockingError(
            "Ligand conformer minimisation did not converge; review its preparation."
        )
    setups = MoleculePreparation(rigid_macrocycles=True).prepare(molecule)
    if len(setups) != 1:
        raise DockingError(
            "Ligand preparation yielded multiple setups; this workflow expects one state."
        )
    pdbqt, success, error = PDBQTWriterLegacy.write_string(setups[0])
    if not success:
        raise DockingError(f"Meeko could not prepare the ligand: {error}")
    return molecule, pdbqt


def _dock(
    directory: Path, slot: str, molecule: Any, pdbqt: str, settings: DockingSettings, binary: str
) -> dict[str, Any]:
    from meeko import PDBQTMolecule, RDKitMolCreate
    from rdkit import Chem

    input_path = directory / f"input_{slot}.pdbqt"
    output_path = directory / f"output_{slot}.pdbqt"
    input_path.write_text(pdbqt)
    command = [
        binary,
        "--receptor",
        str(directory / "receptor.pdbqt"),
        "--ligand",
        str(input_path),
        "--out",
        str(output_path),
        "--cpu",
        "1",
        "--seed",
        str(settings.seed),
        "--exhaustiveness",
        str(settings.exhaustiveness),
        "--num_modes",
        str(settings.n_poses),
    ]
    for axis, center, size in zip("xyz", settings.center, settings.size, strict=True):
        command.extend([f"--center_{axis}", str(center), f"--size_{axis}", str(size)])
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode or not output_path.exists():
        detail = (completed.stderr or completed.stdout)[-1200:]
        raise DockingError(f"Vina failed for ligand {slot}: {detail}")
    docked_pdbqt = output_path.read_text()
    energies = [float(s) for s in re.findall(r"REMARK VINA RESULT:\s+([-+\d.eE]+)", docked_pdbqt)]
    if not energies or not all(math.isfinite(score) for score in energies):
        raise DockingError("Vina output contains no finite scored poses.")
    structures = RDKitMolCreate.from_pdbqt_mol(PDBQTMolecule(docked_pdbqt, poses_to_read=1))
    if len(structures) != 1 or structures[0] is None:
        raise DockingError(
            "Meeko could not reconstruct the docked pose with its original chemistry."
        )
    docked = structures[0]

    def canonical(mol: Any) -> str:
        return str(Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True))

    if canonical(docked) != canonical(molecule):
        raise DockingError(
            "Docked-pose reconstruction changed ligand chemistry or stereochemistry."
        )
    if not docked.GetNumConformers() or not docked.GetConformer().Is3D():
        raise DockingError("Docked pose does not contain 3D coordinates.")
    if not all(
        math.isfinite(float(v)) for xyz in docked.GetConformer().GetPositions() for v in xyz
    ):
        raise DockingError("Docked pose contains invalid coordinates.")
    # Only the top-ranked pose is analysed. Keep all requested PDBQT poses for inspection.
    docked.SetProp("_Name", f"ligand_{slot}")
    docked.SetProp("vina_score_kcal_mol", str(energies[0]))
    buffer = io.StringIO()
    writer = Chem.SDWriter(buffer)
    writer.write(docked, confId=0)
    writer.close()
    return {
        "score_kcal_mol": energies[0],
        "pose_scores_kcal_mol": energies,
        "prepared_pdbqt": pdbqt,
        "pdbqt": docked_pdbqt,
        "sdf": buffer.getvalue(),
        "_molecule": docked,
    }


def _interactions(directory: Path, poses: list[dict[str, Any]]) -> list[str]:
    import prolif as plf
    from prolif.io import MoleculeStandardizer
    from prolif.plotting.network import LigNetwork
    from rdkit import Chem

    pdb_mol = Chem.MolFromPDBFile(str(directory / "protein.pdb"), removeHs=False)
    if pdb_mol is None:
        raise DockingError("Cannot read the prepared protein PDB for interaction analysis.")
    # Template-derived residue chemistry, not a pKa estimate or H-bond optimisation.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        protein = MoleculeStandardizer()(Chem.RemoveHs(pdb_mol))
        ligands = [plf.Molecule.from_rdkit(Chem.RemoveHs(pose["_molecule"])) for pose in poses]
        fingerprint = plf.Fingerprint(implicit_hydrogens=True, count=True)
        fingerprint.run_from_iterable(ligands, protein, n_jobs=1, progress=False)
        for index, (pose, ligand) in enumerate(zip(poses, ligands, strict=True)):
            rows = []
            for (_, residue), contacts in fingerprint.ifp[index].items():
                for interaction, instances in contacts.items():
                    for metadata in instances:
                        atoms = metadata.get("parent_indices", metadata.get("indices", {})) or {}
                        rows.append(
                            {
                                "protein_residue": str(residue),
                                "interaction": interaction.removeprefix("Implicit"),
                                "distance_angstrom": float(metadata["distance"])
                                if "distance" in metadata
                                else None,
                                "ligand_atom_indices": list(map(int, atoms.get("ligand", []))),
                                "protein_atom_indices": list(map(int, atoms.get("protein", []))),
                            }
                        )
            pose["interactions"] = rows
            if rows:
                # Construct/save directly: the convenience plot method calls IPython.display.
                diagram = LigNetwork.from_fingerprint(
                    fingerprint, ligand, kind="frame", frame=index, display_all=True
                )
                html = io.StringIO()
                diagram.save(html, show_interaction_data=True)
                pose["diagram_html"] = html.getvalue()
            else:
                pose["diagram_html"] = ""
            del pose["_molecule"]
    return sorted({str(w.message) for w in caught})


def execute(directory: Path) -> dict[str, Any]:
    request = json.loads((directory / "request.json").read_text())
    settings = DockingSettings(**request["settings"])
    protein = validate_receptor_pair(request["protein_pdb"], request["receptor_pdbqt"], settings)
    (directory / "protein.pdb").write_text(protein)
    (directory / "receptor.pdbqt").write_text(request["receptor_pdbqt"])
    binary = request["vina_binary"]
    engine = subprocess.run(
        [binary, "--version"], capture_output=True, text=True, check=True
    ).stdout.strip()
    poses = []
    for slot, info in zip(("A", "B"), request["ligands"], strict=True):
        _phase(directory, f"Preparing ligand {slot}: 3D conformer and Meeko atom typing…")
        molecule, pdbqt = _prepare_ligand(info["smiles"], settings.seed)
        _phase(directory, f"Docking ligand {slot} with Vina on one CPU…")
        pose = _dock(directory, slot, molecule, pdbqt, settings, binary)
        pose.update(info)
        poses.append(pose)
    _phase(directory, "Analysing both top-ranked poses with ProLIF…")
    messages = _interactions(directory, poses)
    for slot, ligand in zip(("A", "B"), request["ligands"], strict=True):
        if ligand.get("stereochemistry", {}).get("status") == "exploratory_selection":
            messages.append(
                f"Ligand {slot} is an exploratory stereoisomer. Source experimental activity "
                "is not established for this selected docking state."
            )
    manifest = {
        "schema": "sarscope-pair-docking-v1",
        "receptor_label": request["receptor_label"],
        "receptor_preparation": request.get("receptor_provenance"),
        "settings": request["settings"],
        "cpu": 1,
        "vina_version": engine,
        "versions": {name: version(name) for name in ("rdkit", "meeko", "prolif", "numpy")},
        "input_sha256": {
            key: hashlib.sha256(request[key].encode()).hexdigest()
            for key in ("protein_pdb", "receptor_pdbqt")
        },
        "ligands": request["ligands"],
        "analysed_pose_rank": 1,
        "ligand_preparation": (
            "ETKDGv3 + UFF; selected SMILES charges/tautomer retained; no pKa enumeration; "
            "explicit docking-only stereo choices recorded separately from source SMILES"
        ),
        "interaction_method": (
            "ProLIF default geometric definitions; template-based implicit hydrogens"
        ),
        "contact_atom_indices": (
            "Zero-based indices in ProLIF's standardized heavy-atom molecules, "
            "not original PDB serial numbers or SDF atom labels."
        ),
        "warnings": messages,
        "limitations": [
            "Rigid protein-only receptor; no cofactors, waters, covalent ligands or macrocycles.",
            "Docking scores are not measured affinity or pIC50.",
            "Contacts are pose-based hypotheses, not evidence of an activity-cliff mechanism.",
            "No redocking or pose-recovery validation is automatically performed.",
        ],
    }
    return {"manifest": manifest, "ligands": poses}


def main() -> None:
    directory = Path(sys.argv[1])
    try:
        result = execute(directory)
    except Exception as exc:
        (directory / "result.json").write_text(
            json.dumps({"error": f"{type(exc).__name__}: {exc}"})
        )
        raise SystemExit(1) from exc
    (directory / "result.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    main()

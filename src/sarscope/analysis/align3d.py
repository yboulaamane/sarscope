"""Bounded, activity-blind alignment with explicit assumptions and failures.

Pre-aligned coordinates are preserved, not re-docked or energy-minimised.
Generated conformations reuse SARscope's stereo-checked preparation. Pairwise
MCS mapping uses a fixed user reference; the whole-series MCS is QC only and
does not define features using held-out compounds.
"""

from __future__ import annotations

import importlib.util
import multiprocessing as mp
import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign
from rdkit.Geometry import Point3D

from sarscope.analysis.fields import coordinates
from sarscope.conformers import embed_ligand_conformer, validate_conformer_stereo
from sarscope.params import FieldQsarParams


@dataclass
class AlignmentResult:
    molecules: list[Any]
    report: pd.DataFrame
    common_core_smarts: str
    common_core_coverage: float
    reference_index: int
    assumption: str


def preparation_available() -> bool:
    """Keep generated 3D work behind the existing optional preparation extra."""
    return all(importlib.util.find_spec(name) is not None for name in ("meeko", "gemmi"))


class _CompatibleAtom(rdFMCS.MCSAtomCompare):
    """Constrain the search itself, not merely the first returned maximum.

    Steroid functional-group changes can change whether an sp3 atom is
    stereogenic. Requiring equal chiral-tag *presence* collapses the real
    shared core to an ambiguous isolated ring. Validate each molecule's
    assigned 3D stereochemistry separately; never change it to fit a template.
    """

    def __call__(self, parameters: Any, left: Any, i: int, right: Any, j: int) -> bool:
        a, b = left.GetAtomWithIdx(i), right.GetAtomWithIdx(j)
        return bool(
            a.GetAtomicNum() == b.GetAtomicNum()
            and a.GetIsAromatic() == b.GetIsAromatic()
            and a.GetHybridization() == b.GetHybridization()
            and self.CheckAtomRingMatch(parameters, left, i, right, j)
        )


def _mcs(mols: Sequence[Any], timeout: int) -> Any:
    settings = rdFMCS.MCSParameters()
    settings.Timeout = timeout
    settings.AtomTyper = _CompatibleAtom()
    settings.BondTyper = rdFMCS.BondCompare.CompareOrderExact
    settings.AtomCompareParameters.RingMatchesRingOnly = True
    settings.BondCompareParameters.RingMatchesRingOnly = True
    settings.BondCompareParameters.CompleteRingsOnly = True
    result = rdFMCS.FindMCS(list(mols), settings)
    if result.canceled:
        raise ValueError("MCS search reached its time limit; choose a tighter series.")
    if result.numAtoms < 3:
        raise ValueError("Fewer than three common heavy atoms; no defensible core alignment.")
    return result


def best_mapping(
    mol: Any, reference: Any, params: FieldQsarParams, *, fixed_frame: bool = False
) -> tuple[list[tuple[int, int]], float, Any]:
    """Search all permitted symmetry mappings; refuse a truncated search.

    Aromaticity and hybridisation checks prevent a flat aromatic ring from
    being forced onto a puckered saturated ring with similar connectivity.
    """
    result = _mcs([mol, reference], params.mcs_timeout)
    core = Chem.MolFromSmarts(result.smartsString)
    left = mol.GetSubstructMatches(
        core, uniquify=False, useChirality=True, maxMatches=params.max_mappings + 1
    )
    right = reference.GetSubstructMatches(
        core, uniquify=False, useChirality=True, maxMatches=params.max_mappings + 1
    )
    if len(left) > params.max_mappings or len(right) > params.max_mappings:
        raise ValueError(
            "Symmetry-match enumeration exceeded its limit; use a tighter reference. "
            "For reviewed pre-aligned coordinates, common-core QC may explicitly be "
            "marked unassessed; this does not pass alignment validation."
        )
    best: tuple[list[tuple[int, int]], float, Any] | None = None
    seen = set()
    for match in left:
        for ref_match in right:
            mapping = tuple(sorted(zip(match, ref_match, strict=True)))
            if mapping in seen:
                continue
            seen.add(mapping)
            # Paired query automorphisms often describe the SAME atom mapping.
            # Count distinct maps, not the Cartesian product before deduplication.
            # Still refuse a genuinely truncated search, even after finding a good map.
            if len(seen) > params.max_mappings:
                raise ValueError(
                    "Too many distinct symmetry mappings; use a tighter reference. "
                    "No truncated mapping search is accepted."
                )
            if any(
                mol.GetAtomWithIdx(i).GetIsAromatic() != reference.GetAtomWithIdx(j).GetIsAromatic()
                or mol.GetAtomWithIdx(i).GetHybridization()
                != reference.GetAtomWithIdx(j).GetHybridization()
                for i, j in mapping
            ):
                continue
            if fixed_frame:
                xyz, ref_xyz = coordinates(mol), coordinates(reference)
                rms = float(
                    np.sqrt(np.mean([np.sum((xyz[i] - ref_xyz[j]) ** 2) for i, j in mapping]))
                )
                transform = np.eye(4)
            else:
                rms, transform = rdMolAlign.GetAlignmentTransform(
                    mol, reference, atomMap=list(mapping), reflect=False
                )
            if best is None or rms < best[1]:
                best = list(mapping), float(rms), transform
    if best is None:
        raise ValueError("No MCS mapping preserves aromaticity, hybridisation and stereochemistry.")
    return best


def _generate(
    smiles: str, reference_block: str | None, params: FieldQsarParams
) -> tuple[str, dict[str, Any]]:
    molecule, diagnostics = embed_ligand_conformer(smiles, params.seed)
    if reference_block is None:
        return Chem.MolToMolBlock(molecule), diagnostics
    reference = Chem.RemoveHs(Chem.MolFromMolBlock(reference_block, removeHs=False))
    heavy = Chem.RemoveHs(molecule)
    mapping, _, transform = best_mapping(heavy, reference, params)
    rdMolAlign.AlignMol(heavy, reference, atomMap=mapping)
    if params.alignment == "rigid_mcs":
        # Transform the full molecule, including hydrogens, with the same rigid map.
        AllChem.TransformMol(molecule, transform)
        validate_conformer_stereo(molecule, smiles)
        return Chem.MolToMolBlock(molecule), diagnostics
    coord_map = {i: Point3D(*reference.GetConformer().GetAtomPosition(j)) for i, j in mapping}
    embedding = AllChem.ETKDGv3()
    embedding.SetCoordMap(coord_map)
    embedding.randomSeed, embedding.numThreads = params.seed, 1
    embedding.enforceChirality, embedding.useRandomCoords = True, True
    embedding.maxIterations = 100
    if hasattr(embedding, "timeout"):
        embedding.timeout = min(params.molecule_timeout, 10)
    conformers = AllChem.EmbedMultipleConfs(molecule, params.conformers, embedding)
    best_mol, best_energy = None, float("inf")
    for conf_id in conformers:
        candidate = Chem.Mol(molecule)
        conf = Chem.Conformer(candidate.GetConformer(int(conf_id)))
        candidate.RemoveAllConformers()
        candidate.AddConformer(conf, assignId=True)
        rdMolAlign.AlignMol(candidate, reference, atomMap=mapping)
        for i, point in coord_map.items():
            candidate.GetConformer().SetAtomPosition(i, point)
        properties = AllChem.MMFFGetMoleculeProperties(candidate)
        force = (
            AllChem.MMFFGetMoleculeForceField(candidate, properties)
            if properties is not None
            else AllChem.UFFGetMoleculeForceField(candidate)
        )
        if force is None:
            continue
        for i in coord_map:
            force.AddFixedPoint(i)
        force.Initialize()
        force.Minimize(maxIts=200)
        try:
            validate_conformer_stereo(candidate, smiles)
        except ValueError:
            continue
        energy = force.CalcEnergy()
        if np.isfinite(energy) and energy < best_energy:
            best_mol, best_energy = candidate, energy
    if best_mol is None:
        raise ValueError("No stereo-preserving core-constrained conformer; provide reviewed poses.")
    return Chem.MolToMolBlock(best_mol), {**diagnostics, "restrained_energy": best_energy}


def _worker(connection: Any) -> None:
    """Serial requests amortise spawn/import cost without parallelising RDKit."""
    try:
        connection.send(("ready", None))
        while True:
            try:
                request = connection.recv()
            except EOFError:
                return
            if request is None:
                return
            try:
                connection.send((True, _generate(*request)))
            except Exception as exc:
                connection.send((False, str(exc)))
    finally:
        connection.close()


class _AlignmentGenerator:
    """One killable worker per series, with separate startup/job/series errors.

    A slow or failed request still blocks the series; no compound, stereo
    constraint or alignment check is silently skipped. The outer series
    deadline includes worker startup and every request.
    """

    def __init__(self, params: FieldQsarParams, deadline: float) -> None:
        self.params, self.deadline = params, deadline
        context = mp.get_context("spawn")
        self.connection, self.child = context.Pipe(duplex=True)
        self.process = context.Process(target=_worker, args=(self.child,), daemon=True)

    def _allowance(self) -> tuple[float, bool]:
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise ValueError(
                f"Whole-series alignment budget ({self.params.alignment_timeout}s) exhausted. "
                "This is not evidence that the current compound cannot embed. "
                "Use a smaller, related series or reviewed pre-aligned SDF."
            )
        allowance = min(self.params.molecule_timeout, remaining)
        return allowance, remaining < self.params.molecule_timeout

    def _receive(self, stage: str) -> Any:
        allowance, series_limited = self._allowance()
        try:
            if not self.connection.poll(allowance):
                if series_limited:
                    raise ValueError(
                        f"Whole-series alignment budget ({self.params.alignment_timeout}s) "
                        f"exhausted during {stage}; only {allowance:.1f}s remained. "
                        "This does not identify an invalid compound. "
                        "Use a smaller, related series or reviewed pre-aligned SDF."
                    )
                raise ValueError(
                    f"Alignment {stage} exceeded its {self.params.molecule_timeout}s limit. "
                    "The shared host may be busy, or this job may need reviewed pre-aligned input. "
                    "No molecule or stereochemical constraint was skipped."
                )
            return self.connection.recv()
        except (EOFError, OSError) as exc:
            raise ValueError(
                f"Alignment worker stopped unexpectedly during {stage}. "
                "The host may have stopped the process; no partial alignment is accepted."
            ) from exc

    def __enter__(self) -> _AlignmentGenerator:
        try:
            self._allowance()
            self.process.start()
            self.child.close()
            if self._receive("worker startup") != ("ready", None):
                raise ValueError("Alignment worker did not initialise correctly.")
            return self
        except Exception:
            self.close()
            raise

    def generate(self, smiles: str, reference: Any | None) -> tuple[Any, dict[str, Any]]:
        self._allowance()
        try:
            self.connection.send(
                (
                    smiles,
                    Chem.MolToMolBlock(reference) if reference is not None else None,
                    self.params,
                )
            )
        except (EOFError, OSError) as exc:
            raise ValueError(
                "Alignment worker stopped before receiving the next job; "
                "no partial alignment is accepted."
            ) from exc
        ok, result = self._receive("conformer/MCS job")
        if not ok:
            raise ValueError(result)
        block, diagnostics = result
        molecule = Chem.MolFromMolBlock(block, removeHs=False)
        if molecule is None:
            raise ValueError("Alignment worker returned unreadable coordinates.")
        return molecule, diagnostics

    def close(self) -> None:
        self.connection.close()
        self.child.close()
        if self.process.pid is not None:
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()

    def __exit__(self, *args: Any) -> None:
        self.close()


def _bounded_generate(
    smiles: str, reference: Any | None, params: FieldQsarParams
) -> tuple[Any, dict[str, Any]]:
    with _AlignmentGenerator(params, time.monotonic() + params.alignment_timeout) as worker:
        return worker.generate(smiles, reference)


def align_series(
    mols: Sequence[Any], identifiers: Sequence[str], params: FieldQsarParams
) -> AlignmentResult:
    """No automatic exclusions: a failed molecule blocks preparation for review."""
    deadline = time.monotonic() + params.alignment_timeout
    if len(mols) != len(identifiers) or not mols or len(mols) > params.max_molecules:
        raise ValueError("Molecule count does not match identifiers or exceeds the series limit.")
    if not 0 <= params.reference_index < len(mols):
        raise ValueError("Reference molecule is outside this dataset.")
    if params.alignment != "prealigned" and not preparation_available():
        raise ValueError("Generated alignment requires sarscope[docking]; upload pre-aligned SDF.")
    heavy = [Chem.RemoveHs(Chem.Mol(mol)) for mol in mols]
    for identifier, mol in zip(identifiers, heavy, strict=True):
        if len(Chem.GetMolFrags(mol)) != 1 or not 3 <= mol.GetNumHeavyAtoms() <= 80:
            raise ValueError(f"{identifier}: use a single component with 3–80 heavy atoms.")
        if any(
            info.specified != Chem.StereoSpecified.Specified
            for info in Chem.FindPotentialStereo(mol)
        ):
            raise ValueError(f"{identifier}: resolve unspecified R/S or E/Z stereochemistry first.")
    if params.alignment == "prealigned" and not params.assess_prealigned_core:
        rows, retained = [], []
        inputs = mols if params.preserve_input_hydrogens else heavy
        for identifier, original in zip(identifiers, inputs, strict=True):
            mol = Chem.Mol(original)
            coordinates(mol)
            validate_conformer_stereo(mol, Chem.MolToSmiles(mol))
            mol.SetProp("_Name", str(identifier))
            retained.append(mol)
            rows.append(
                {
                    "molecule_id": identifier,
                    "core_atoms": None,
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                    "core_coverage": np.nan,
                    "core_rmsd_A": np.nan,
                    "warning": "Core alignment QC not assessed; supplied frame retained",
                }
            )
        return AlignmentResult(
            retained,
            pd.DataFrame(rows),
            "",
            float("nan"),
            params.reference_index,
            "Supplied pre-aligned coordinates retained unchanged. Stereo and finite-coordinate "
            "checks passed; common-core RMSD/coverage were not assessed. Alignment validity "
            "must come from the declared external protocol, not this software.",
        )
    session = (
        nullcontext(None)
        if params.alignment == "prealigned"
        else _AlignmentGenerator(params, deadline)
    )
    with session as worker:
        return _align_checked(mols, heavy, identifiers, params, deadline, worker)


def _align_checked(
    mols: Sequence[Any],
    heavy: Sequence[Any],
    identifiers: Sequence[str],
    params: FieldQsarParams,
    deadline: float,
    worker: _AlignmentGenerator | None,
) -> AlignmentResult:
    reference = heavy[params.reference_index]
    if params.alignment != "prealigned" and (
        not reference.GetNumConformers() or not reference.GetConformer().Is3D()
    ):
        assert worker is not None
        try:
            reference, _ = worker.generate(Chem.MolToSmiles(reference), None)
        except ValueError as exc:
            raise ValueError(f"Reference {identifiers[params.reference_index]}: {exc}") from exc
        reference = Chem.RemoveHs(reference)
    coordinates(reference)
    validate_conformer_stereo(reference, Chem.MolToSmiles(reference))
    aligned: list[Any] = []
    rows = []
    for position, (identifier, mol) in enumerate(zip(identifiers, heavy, strict=True)):
        try:
            remaining = int(deadline - time.monotonic())
            if remaining < 1:
                raise ValueError(
                    "Series alignment budget exceeded; use a smaller series or reviewed SDF."
                )
            if params.alignment == "prealigned":
                coordinates(mol)
                validate_conformer_stereo(mol, Chem.MolToSmiles(mol))
                output = Chem.Mol(mol)
                mapping, _, _ = best_mapping(output, reference, params, fixed_frame=True)
            else:
                assert worker is not None
                if position == params.reference_index and params.alignment == "rigid_mcs":
                    output = Chem.Mol(reference)
                else:
                    output, _ = worker.generate(Chem.MolToSmiles(mol), reference)
                output = Chem.RemoveHs(output)
                mapping, _, _ = best_mapping(output, reference, params, fixed_frame=True)
            xyz, ref_xyz = coordinates(output), coordinates(reference)
            rmsd = float(np.sqrt(np.mean([np.sum((xyz[i] - ref_xyz[j]) ** 2) for i, j in mapping])))
            coverage = len(mapping) / mol.GetNumHeavyAtoms()
            warning = []
            if rmsd > params.max_alignment_rmsd:
                warning.append("Core RMSD exceeds the chosen threshold")
            if coverage < params.min_core_coverage:
                warning.append("Common core covers too little of this molecule")
            output.SetProp("_Name", str(identifier))
            if params.alignment == "prealigned" and params.preserve_input_hydrogens:
                # MCS/QC use heavy atoms; retain the complete reviewed input frame.
                output = Chem.Mol(mols[len(aligned)])
                coordinates(output)
                validate_conformer_stereo(output, Chem.MolToSmiles(output))
                output.SetProp("_Name", str(identifier))
            aligned.append(output)
            rows.append(
                {
                    "molecule_id": identifier,
                    "core_atoms": len(mapping),
                    "heavy_atoms": mol.GetNumHeavyAtoms(),
                    "core_coverage": coverage,
                    "core_rmsd_A": rmsd,
                    "warning": "; ".join(warning),
                }
            )
        except Exception as exc:
            raise ValueError(
                f"{identifier} (compound {position + 1}/{len(heavy)}, "
                f"{len(aligned)} completed): {exc}"
            ) from exc
    common = _mcs(heavy, params.mcs_timeout)
    coverage = common.numAtoms / float(np.mean([m.GetNumHeavyAtoms() for m in heavy]))
    assumptions = {
        "prealigned": "Uploaded coordinates retained unchanged; alignment origin is user asserted, "
        "not automatically experimental or a verified binding pose.",
        "rigid_mcs": "Rigid pairwise MCS alignment to a fixed reference; conformations are "
        "generated hypotheses, not observed bound poses.",
        "constrained": "Reference-core coordinates fixed during substituent sampling and "
        "minimisation; no receptor, docking or pose validation is implied.",
    }
    return AlignmentResult(
        aligned,
        pd.DataFrame(rows),
        common.smartsString,
        coverage,
        params.reference_index,
        assumptions[params.alignment],
    )

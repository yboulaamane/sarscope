"""Chunked molecular fields in a reviewed common Cartesian frame.

CoMFA: Cramer et al., JACS 1988, doi:10.1021/ja00226a005.
CoMSIA: Klebe et al., J Med Chem 1994, doi:10.1021/jm00050a010.
These are explicitly *style* implementations: LJ uses RDKit van der Waals
radii and UFF well depths, and Gaussian hydrophobicity uses Crippen atomic
logP. They are not claimed to reproduce proprietary Tripos atom tables or
directional donor/acceptor extension points. No GPL implementation consulted.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Lipinski

from sarscope.analysis.tripos_reference import PARAMETER_SOURCE, vdw_parameters
from sarscope.params import FieldQsarParams

CHARGE_PROPERTY = "_SARscopeFieldCharge"
COMFA_FIELDS = ("steric", "electrostatic")
COMSIA_FIELDS = (*COMFA_FIELDS, "hydrophobic", "hbond_donor", "hbond_acceptor")
COULOMB = 332.0636


def _gasteiger(mol: Any) -> NDArray[np.float64]:
    AllChem.ComputeGasteigerCharges(mol, throwOnParamFailure=True)
    return np.array([float(atom.GetProp("_GasteigerCharge")) for atom in mol.GetAtoms()])


def _mmff(mol: Any) -> NDArray[np.float64]:
    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        raise ValueError("MMFF94 has no parameters for this molecule.")
    return np.array([props.GetMMFFPartialCharge(i) for i in range(mol.GetNumAtoms())])


CHARGE_MODELS: dict[str, Callable[[Any], NDArray[np.float64]]] = {
    "gasteiger": _gasteiger,
    "mmff94": _mmff,
}


def register_charge_model(name: str, fn: Callable[[Any], NDArray[np.float64]]) -> None:
    """Register a local charge provider without making its toolkit a dependency."""
    if not name or name in CHARGE_MODELS or not callable(fn):
        raise ValueError("Use a new, nonempty charge model name and a callable.")
    CHARGE_MODELS[name] = fn


@dataclass(frozen=True)
class GridSpec:
    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    spacing: float = 2.0

    def __post_init__(self) -> None:
        if len(self.shape) != 3 or any(type(n) is not int or n < 2 for n in self.shape):
            raise ValueError("A grid needs three axes of at least two points.")
        if len(self.origin) != 3 or not np.isfinite(self.origin).all():
            raise ValueError("Grid origin must have three finite coordinates.")
        if not np.isfinite(self.spacing) or self.spacing <= 0:
            raise ValueError("Grid spacing must be finite and positive.")

    @property
    def n_points(self) -> int:
        return int(np.prod(self.shape))

    @property
    def points(self) -> NDArray[np.float64]:
        # C order: z varies fastest. OpenDX export preserves this convention.
        index = np.indices(self.shape).reshape(3, -1).T
        return np.asarray(self.origin) + self.spacing * index

    @classmethod
    def around(cls, mols: Sequence[Any], params: FieldQsarParams) -> GridSpec:
        xyz = np.concatenate([coordinates(m) for m in mols])
        if params.grid_convention == "rounded_extent":
            # Haga et al. (2025), Methods 4.4: rounded atomic extents; ceil(range/step)
            # points. Half-open bounds, unlike the inclusive enclosing convention.
            snapped = np.rint(xyz / params.spacing) * params.spacing
            low, high = snapped.min(0) - params.margin, snapped.max(0) + params.margin
            shape = tuple(int(n) for n in np.ceil((high - low) / params.spacing))
        else:
            offset = np.asarray(params.grid_phase) * params.spacing
            low = (
                np.floor((xyz.min(axis=0) - params.margin - offset) / params.spacing)
                * params.spacing
                + offset
            )
            high = (
                np.ceil((xyz.max(axis=0) + params.margin - offset) / params.spacing)
                * params.spacing
                + offset
            )
            shape = tuple(int(n) for n in np.rint((high - low) / params.spacing) + 1)
        grid = cls(tuple(float(n) for n in low), shape, params.spacing)  # type: ignore[arg-type]
        if grid.n_points > params.max_grid_points:
            raise ValueError(
                f"Grid has {grid.n_points:,} points; limit {params.max_grid_points:,}. "
                "Increase spacing or reduce margin. Check that poses share one frame."
            )
        return grid


def coordinates(mol: Any) -> NDArray[np.float64]:
    if mol is None or not mol.GetNumConformers() or not mol.GetConformer().Is3D():
        raise ValueError("Every molecule needs a reviewed 3D conformer.")
    xyz = np.asarray(mol.GetConformer().GetPositions(), dtype=float)
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite molecular coordinates.")
    return xyz


def estimate_field_memory(
    n_molecules: int, n_atoms: int, grid: GridSpec, params: FieldQsarParams
) -> int:
    """Conservative bytes for fields, working copies and a distance chunk.

    Deliberately no p-by-p correlation matrix: field filtering is linear in
    column count. PLS workspaces are included as several matrix copies.
    """
    n_fields = len(params.field_subset) or (5 if params.method == "comsia" else 2)
    matrix = n_molecules * grid.n_points * n_fields * 8
    workspace = min(params.chunk_points, grid.n_points) * n_atoms * 8 * 12
    return matrix * 8 + workspace + grid.n_points * 3 * 8


@dataclass
class FieldResult:
    matrix: NDArray[np.float64]
    labels: tuple[str, ...]
    grid: GridSpec
    charges: list[dict[str, Any]]
    memory_bytes: int
    convention: str


def _charges(mol: Any, params: FieldQsarParams) -> tuple[NDArray[np.float64], str, str]:
    if params.charge_model not in CHARGE_MODELS:
        raise ValueError(f"Unknown charge model: {params.charge_model}")
    actual, reason = params.charge_model, ""
    try:
        values = np.asarray(CHARGE_MODELS[actual](mol), dtype=float)
    except Exception as exc:
        if actual != "mmff94" or not params.allow_charge_fallback:
            raise ValueError(f"Charge calculation failed ({actual}): {exc}") from exc
        actual, reason = "gasteiger", str(exc)
        values = _gasteiger(mol)
    if values.shape != (mol.GetNumAtoms(),) or not np.isfinite(values).all():
        raise ValueError("Charge provider must return one finite charge per explicit atom.")
    if abs(float(values.sum()) - Chem.GetFormalCharge(mol)) > 0.1:
        raise ValueError("Partial charges do not sum to the molecular formal charge (±0.1 e).")
    for atom, value in zip(mol.GetAtoms(), values, strict=True):
        atom.SetDoubleProp(CHARGE_PROPERTY, float(value))
    return values, actual, reason


def _properties(mol: Any, charges: NDArray[np.float64]) -> NDArray[np.float64]:
    radii = np.array([Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()])
    weights = np.zeros((mol.GetNumAtoms(), 5))
    weights[:, 0], weights[:, 1] = radii**3, charges
    weights[:, 2] = [value[0] for value in Crippen._GetAtomContribs(mol)]
    # Lipinski atom SMARTS retain neutral amine acceptors but exclude amide,
    # sulfonamide and protonated N. Pharmacophore BaseFeatures intentionally
    # omit some ionisable amines and are not interchangeable with this rule.
    for column, query in ((3, Lipinski.HDonorSmarts), (4, Lipinski.HAcceptorSmarts)):
        for match in mol.GetSubstructMatches(query):
            weights[list(match), column] = 1.0
    return weights


def compute_fields(mols: Sequence[Any], grid: GridSpec, params: FieldQsarParams) -> FieldResult:
    """Return field-major columns; never mutate the caller's molecules.

    Explicit hydrogens are required: assigning Gasteiger charges only to heavy
    atoms would omit the hydrogen charges from the Coulomb sum. CoMFA NaNs
    remain NaNs; no mean or zero imputation is hidden here.
    """
    if not mols or len(mols) > params.max_molecules:
        raise ValueError(f"Use 1–{params.max_molecules} molecules.")
    if grid.n_points > params.max_grid_points:
        raise ValueError("Grid exceeds the configured point limit.")
    prepared = [Chem.AddHs(Chem.Mol(mol), addCoords=True) for mol in mols]
    memory = estimate_field_memory(len(mols), max(m.GetNumAtoms() for m in prepared), grid, params)
    if memory > params.max_memory_mb * 1024**2:
        raise ValueError("Estimated field workspace exceeds the memory budget; use a coarser grid.")
    names = params.field_subset or (COMSIA_FIELDS if params.method == "comsia" else COMFA_FIELDS)
    points = grid.points
    matrix = np.empty((len(prepared), len(names) * grid.n_points))
    report = []
    for row, mol in enumerate(prepared):
        xyz = coordinates(mol)
        charge, actual, reason = _charges(mol, params)
        report.append(
            {
                "row": row,
                "requested": params.charge_model,
                "used": actual,
                "fallback": bool(reason),
                "reason": reason,
            }
        )
        reference_radii, reference_depths = None, None
        if params.field_definition == "tripos52_se":
            reference_radii, reference_depths, classes = vdw_parameters(mol)
            report[-1].update({"vdw_source": PARAMETER_SOURCE, "parameter_classes": classes})
        weights = None
        if params.method == "comsia" and reference_radii is not None:
            # SE only: no Crippen or atom-centred HB substitution for SYBYL fields.
            weights = np.zeros((mol.GetNumAtoms(), 5))
            weights[:, 0], weights[:, 1] = reference_radii**3, charge
        elif params.method == "comsia":
            weights = _properties(mol, charge)
        if weights is None:
            if reference_radii is not None:
                assert reference_depths is not None
                radii = reference_radii
                epsilon = np.sqrt(reference_depths * params.probe_epsilon)
            else:
                radii = np.array(
                    [Chem.GetPeriodicTable().GetRvdw(a.GetAtomicNum()) for a in mol.GetAtoms()]
                )
                vdw = [AllChem.GetUFFVdWParams(mol, i, i) for i in range(mol.GetNumAtoms())]
                if any(value is None for value in vdw):
                    raise ValueError("UFF well depths unavailable; cannot compute LJ fields.")
                epsilon = np.sqrt(np.array([value[1] for value in vdw]) * params.probe_epsilon)
        result = matrix[row].reshape(len(names), grid.n_points)
        for start in range(0, grid.n_points, params.chunk_points):
            end = min(start + params.chunk_points, grid.n_points)
            delta = points[start:end, None, :] - xyz[None, :, :]
            distance2 = np.einsum("ijk,ijk->ij", delta, delta)
            if weights is not None:
                selected_weights = weights[:, [COMSIA_FIELDS.index(name) for name in names]]
                result[:, start:end] = (np.exp(-params.alpha * distance2) @ selected_weights).T
            else:
                distance = np.sqrt(np.maximum(distance2, params.minimum_distance**2))
                ratio6 = ((radii + params.probe_radius) / distance) ** 6
                steric = (epsilon * (ratio6**2 - 2 * ratio6)).sum(axis=1)
                denominator = distance if params.dielectric_model == "constant" else distance**2
                electro = COULOMB * (charge / denominator).sum(axis=1) / params.dielectric
                if params.electrostatic_cutoff is not None:
                    electro = np.clip(
                        electro, -params.electrostatic_cutoff, params.electrostatic_cutoff
                    )
                electro[steric >= params.steric_cutoff] = np.nan
                values = {
                    "steric": np.minimum(steric, params.steric_cutoff),
                    "electrostatic": electro,
                }
                for column, name in enumerate(names):
                    result[column, start:end] = values[name]
    return FieldResult(
        matrix,
        tuple(f"{name}_{point}" for name in names for point in range(grid.n_points)),
        grid,
        report,
        memory,
        (
            "Public Tripos-5.2 SE parameters; no vendor atom-typing or full SYBYL parity. "
            + (
                "Positive-sum Gaussian; radius^3 and current partial charges only."
                if params.method == "comsia"
                else "LJ 12-6; abrupt plateaus; Coulomb with explicit dielectric/cutoff settings."
            )
        )
        if params.field_definition == "tripos52_se"
        else "positive-sum Gaussian; r^3/RDKit radii; Crippen logP; atom-centred H-bond features"
        if params.method == "comsia"
        else "LJ 12-6 with RDKit radii/UFF well depths; +1e carbon probe; "
        f"{params.dielectric_model} dielectric; "
        f"electrostatic plateau={params.electrostatic_cutoff}",
    )

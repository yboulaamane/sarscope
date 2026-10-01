"""Explicit, reproducible feature schemas for molecular QSAR."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Descriptors

from sarscope.analysis.features import fingerprint_matrix
from sarscope.params import FeatureParams

PHYSICOCHEMICAL = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "NumRotatableBonds",
    "NumHDonors",
    "NumHAcceptors",
    "HeavyAtomCount",
    "FractionCSP3",
    "RingCount",
    "NumAromaticRings",
    "MolMR",
    "qed",
)
MEDCHEM = (
    *PHYSICOCHEMICAL,
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeterocycles",
    "NHOHCount",
    "NOCount",
    "NumAmideBonds",
    "NumSpiroAtoms",
    "NumBridgeheadAtoms",
    "LabuteASA",
    "BertzCT",
    "BalabanJ",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
)
DESCRIPTOR_FUNCTIONS = dict(Descriptors.descList)
# Ipc can be slow and numerically unstable for large molecular graphs.
ALL_2D = tuple(name for name in DESCRIPTOR_FUNCTIONS if name not in {"Ipc", "AvgIpc"})
DESCRIPTOR_PRESETS = {
    "Physicochemical": tuple(name for name in PHYSICOCHEMICAL if name in DESCRIPTOR_FUNCTIONS),
    "Medicinal chemistry": tuple(name for name in MEDCHEM if name in DESCRIPTOR_FUNCTIONS),
    "All RDKit 2D (excluding Ipc/AvgIpc)": ALL_2D,
}


def selected_descriptors(features: FeatureParams) -> tuple[str, ...]:
    return (
        tuple(getattr(features, "descriptor_names", ()))
        or DESCRIPTOR_PRESETS["Medicinal chemistry"]
    )


def resolved_features(features: FeatureParams) -> FeatureParams:
    """Pin fallback descriptor names in persisted bundles, including their order."""
    if getattr(features, "representation", "fingerprint") == "fingerprint":
        return features
    return replace(features, descriptor_names=selected_descriptors(features))


def descriptor_matrix(
    smiles: Sequence[str], names: Sequence[str], *, backend: str = "rdkit"
) -> NDArray[np.float64]:
    """Calculate named 2D descriptors; retain missing values for fold-local imputation.

    Neither backend changes already curated structures. Molfeat's implicit
    standardization, IPC averaging, and augmentation are explicitly disabled.
    No pretrained models are downloaded.
    """
    names = tuple(names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Select at least one descriptor, without duplicate names.")
    unknown = set(names) - DESCRIPTOR_FUNCTIONS.keys()
    if unknown:
        raise ValueError(f"Unknown RDKit descriptor(s): {sorted(unknown)}")
    calculator: Any = None
    if backend == "molfeat":
        try:
            from molfeat.calc import RDKitDescriptors2D
        except ImportError as exc:
            raise ImportError("Molfeat is optional: pip install 'sarscope[molfeat]'") from exc
        calculator = RDKitDescriptors2D(
            descrs=list(names),
            replace_nan=False,
            augment=False,
            avg_ipc=False,
            do_not_standardize=True,
        )
        if tuple(calculator.columns) != names:
            raise ValueError("Molfeat's descriptor schema differs from the requested names.")
    elif backend != "rdkit":
        raise ValueError(f"Unknown descriptor backend: {backend}")
    values = np.full((len(smiles), len(names)), np.nan, dtype=float)
    for row, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            raise ValueError(f"Cannot parse curated SMILES: {smi!r}")
        if calculator is not None:
            values[row] = np.asarray(calculator(mol), dtype=float)
        else:
            for column, name in enumerate(names):
                try:
                    values[row, column] = float(DESCRIPTOR_FUNCTIONS[name](mol))
                except (ValueError, RuntimeError, OverflowError, ZeroDivisionError):
                    pass
    values[~np.isfinite(values)] = np.nan
    return values


def model_matrix(smiles: Sequence[str], features: FeatureParams) -> NDArray[Any]:
    representation = getattr(features, "representation", "fingerprint")
    if representation == "fingerprint":
        return fingerprint_matrix(smiles, features.fingerprint, ecfp_bits=features.ecfp_bits)
    if representation not in {"rdkit2d", "hybrid", "molfeat2d"}:
        raise ValueError(f"Unknown molecular representation: {representation}")
    descriptors = descriptor_matrix(
        smiles,
        selected_descriptors(features),
        backend="molfeat" if representation == "molfeat2d" else "rdkit",
    )
    if representation == "hybrid":
        return np.column_stack(
            (
                fingerprint_matrix(smiles, features.fingerprint, ecfp_bits=features.ecfp_bits),
                descriptors,
            )
        )
    return descriptors


def model_feature_names(features: FeatureParams) -> tuple[str, ...]:
    representation = getattr(features, "representation", "fingerprint")
    bits = {"ecfp4": features.ecfp_bits, "maccs": 166, "pubchem": 881}[features.fingerprint]
    fingerprint_names = tuple(f"{features.fingerprint}_bit_{i}" for i in range(bits))
    if representation == "fingerprint":
        return fingerprint_names
    names = selected_descriptors(features)
    return (*fingerprint_names, *names) if representation == "hybrid" else names

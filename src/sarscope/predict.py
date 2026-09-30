"""Persist a continuous QSAR model and apply it to new structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from sarscope.analysis.features import VarianceCorrelationFilter, fingerprint_matrix
from sarscope.params import FeatureParams

MODEL_FILENAME = "model.joblib"


@dataclass
class PredictionBundle:
    """Everything needed to reproduce fingerprinting and make predictions."""

    algorithm: str
    features: FeatureParams
    feature_filter: VarianceCorrelationFilter
    estimator: Any
    train_fingerprints: NDArray[np.uint8]
    similarity_threshold: float
    target_label: str = "pactivity"
    train_ids: list[str] = field(default_factory=list)
    train_smiles: list[str] = field(default_factory=list)
    train_pactivity: list[float] = field(default_factory=list)
    canonical_tautomer: bool = True
    empirical_half_width: float | None = None
    interval_level: float = 0.9


def _row_max_tanimoto(
    query: NDArray[Any], reference: NDArray[Any], *, exclude_diagonal: bool = False
) -> NDArray[np.float64]:
    """Maximum binary Tanimoto similarity without materialising a cube."""
    maxima, _ = _row_nearest_tanimoto(query, reference, exclude_diagonal=exclude_diagonal)
    return maxima


def _row_nearest_tanimoto(
    query: NDArray[Any], reference: NDArray[Any], *, exclude_diagonal: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Nearest training row per query, in bounded matrix blocks."""
    query = np.asarray(query, dtype=np.uint8)
    reference = np.asarray(reference, dtype=np.uint8)
    if len(reference) == 0:
        return np.full(len(query), np.nan), np.full(len(query), -1, dtype=np.intp)
    maxima = np.zeros(len(query), dtype=float)
    indices = np.zeros(len(query), dtype=np.intp)
    reference_sum = reference.sum(axis=1)
    for start in range(0, len(query), 256):
        block = query[start : start + 256]
        intersections = block.astype(np.uint16) @ reference.T.astype(np.uint16)
        unions = block.sum(axis=1)[:, None] + reference_sum[None, :] - intersections
        similarities = np.divide(
            intersections,
            unions,
            out=np.zeros_like(intersections, dtype=float),
            where=unions != 0,
        )
        if exclude_diagonal and query is reference:
            rows = np.arange(start, min(start + len(block), len(reference)))
            similarities[np.arange(len(rows)), rows] = -1.0
        indices[start : start + len(block)] = similarities.argmax(axis=1)
        maxima[start : start + len(block)] = similarities.max(axis=1)
    return maxima, indices


def similarity_domain_threshold(train_fingerprints: NDArray[Any]) -> float:
    """Fifth percentile of each training row's nearest non-self neighbour."""
    X = np.asarray(train_fingerprints, dtype=np.uint8)
    if len(X) < 2:
        return 1.0
    nearest = _row_max_tanimoto(X, X, exclude_diagonal=True)
    return float(np.quantile(nearest, 0.05))


def save_bundle(bundle: PredictionBundle, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: Path) -> PredictionBundle:
    """Load a SARscope bundle. Only load model files from a trusted source."""
    path = Path(path)
    if path.is_dir():
        path = path / MODEL_FILENAME
    if not path.is_file():
        raise ValueError(f"model file not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, PredictionBundle):
        raise ValueError(f"{path} is not a SARscope prediction model")
    return bundle


def predict_smiles(
    bundle: PredictionBundle, molecule_ids: list[str], smiles: list[str]
) -> pd.DataFrame:
    if len(molecule_ids) != len(smiles):
        raise ValueError("molecule ID and SMILES counts differ")
    features = bundle.features
    X = fingerprint_matrix(smiles, features.fingerprint, ecfp_bits=features.ecfp_bits)
    predictions = np.asarray(
        bundle.estimator.predict(bundle.feature_filter.transform(X)), dtype=float
    )
    similarity, nearest = _row_nearest_tanimoto(X, bundle.train_fingerprints)
    result = pd.DataFrame(
        {
            "molecule_id": molecule_ids,
            "smiles": smiles,
            "predicted_pactivity": predictions,
            "max_training_similarity": similarity,
            "in_applicability_domain": similarity >= bundle.similarity_threshold,
            "domain_threshold": bundle.similarity_threshold,
        }
    )
    half_width = getattr(bundle, "empirical_half_width", None)
    if half_width is not None and np.isfinite(half_width):
        result["empirical_lower_pactivity"] = predictions - half_width
        result["empirical_upper_pactivity"] = predictions + half_width
    train_ids = getattr(bundle, "train_ids", [])
    train_smiles = getattr(bundle, "train_smiles", [])
    train_pactivity = getattr(bundle, "train_pactivity", [])
    if train_ids and len(train_ids) == len(bundle.train_fingerprints):
        result["nearest_training_id"] = [train_ids[i] for i in nearest]
        if len(train_smiles) == len(bundle.train_fingerprints):
            result["nearest_training_smiles"] = [train_smiles[i] for i in nearest]
            result["already_measured"] = [
                smiles[row] == train_smiles[index] for row, index in enumerate(nearest)
            ]
        if len(train_pactivity) == len(bundle.train_fingerprints):
            result["nearest_training_pactivity"] = [train_pactivity[i] for i in nearest]
    return result


def predict_table(
    bundle: PredictionBundle,
    path: Path,
    *,
    id_col: str = "molecule_id",
    smiles_col: str = "smiles",
) -> pd.DataFrame:
    path = Path(path)
    sep = "\t" if path.suffix.lower() in {".tsv", ".tab"} else ","
    raw = pd.read_csv(path, sep=sep, dtype={id_col: str, smiles_col: str})
    missing = [column for column in (id_col, smiles_col) if column not in raw]
    if missing:
        raise ValueError(f"{path.name}: missing column(s) {missing}; found {list(raw.columns)}")
    frame = raw[[id_col, smiles_col]].dropna()
    ids = frame[id_col].astype(str).str.strip().tolist()
    smiles = frame[smiles_col].astype(str).str.strip().tolist()
    return predict_smiles(bundle, ids, smiles)

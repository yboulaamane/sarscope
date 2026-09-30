"""Review uploaded compounds against a trained SARscope regression model."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem

from sarscope.analysis.descriptors import compute_descriptors
from sarscope.analysis.features import fingerprint_matrix
from sarscope.curate import _standardise_one
from sarscope.predict import PredictionBundle, predict_smiles


def predict_candidates(bundle: PredictionBundle, source: pd.DataFrame) -> pd.DataFrame:
    """Standardise and predict valid rows while retaining per-row error reasons."""
    names = {str(name).lower(): name for name in source.columns}
    if "smiles" not in names:
        raise ValueError("Upload needs a 'smiles' column; 'molecule_id' is optional.")
    if not 1 <= len(source) <= 500:
        raise ValueError("Upload 1–500 molecules per run on the hosted app.")
    smiles_col = names["smiles"]
    id_col = names.get("molecule_id")
    rows: list[dict[str, Any]] = []
    valid_positions: list[int] = []
    valid_ids: list[str] = []
    valid_smiles: list[str] = []
    for position, source_row in enumerate(source.itertuples(index=False, name=None)):
        values = dict(zip(source.columns, source_row, strict=True))
        raw = values[smiles_col]
        molecule_id = str(values[id_col]).strip() if id_col and pd.notna(values[id_col]) else ""
        molecule_id = molecule_id or f"row_{position + 1}"
        entry: dict[str, Any] = {
            "input_row": position + 1,
            "molecule_id": molecule_id,
            "input_smiles": raw,
            "canonical_smiles": None,
            "status": "invalid",
        }
        result = _standardise_one(str(raw).strip(), bundle.canonical_tautomer)
        if isinstance(result, str):
            entry["status"] = result
        else:
            canonical, _ = result
            entry["canonical_smiles"] = canonical
            entry["status"] = "predicted"
            molecule = Chem.MolFromSmiles(canonical)
            assert molecule is not None
            entry.update(compute_descriptors(molecule))
            valid_positions.append(position)
            valid_ids.append(molecule_id)
            valid_smiles.append(canonical)
        rows.append(entry)
    output = pd.DataFrame(rows)
    if valid_smiles:
        predictions = predict_smiles(bundle, valid_ids, valid_smiles)
        for column in predictions.columns:
            if column in {"molecule_id", "smiles"}:
                continue
            output[column] = None
            output.loc[valid_positions, column] = predictions[column].tolist()
    return output


def diverse_shortlist(
    predictions: pd.DataFrame,
    bundle: PredictionBundle,
    *,
    n: int = 20,
    max_mw: float = 500.0,
    max_logp: float = 5.0,
    max_tpsa: float = 140.0,
    in_domain_only: bool = True,
    diversity_weight: float = 1.0,
) -> pd.DataFrame:
    """Greedy potency/diversity shortlist from eligible unique structures."""
    if not 1 <= n <= 100 or not math.isfinite(diversity_weight) or diversity_weight < 0:
        raise ValueError("Choose 1–100 compounds and a non-negative diversity weight.")
    eligible = predictions[predictions["status"] == "predicted"].copy()
    if eligible.empty:
        return eligible
    if in_domain_only:
        eligible = eligible[eligible["in_applicability_domain"].eq(True)]
    eligible = eligible[
        (pd.to_numeric(eligible["MW"], errors="coerce") <= max_mw)
        & (pd.to_numeric(eligible["logP"], errors="coerce") <= max_logp)
        & (pd.to_numeric(eligible["TPSA"], errors="coerce") <= max_tpsa)
    ]
    eligible = (
        eligible.sort_values("predicted_pactivity", ascending=False)
        .drop_duplicates("canonical_smiles")
        .head(150)
        .reset_index(drop=True)
    )
    if eligible.empty:
        return eligible
    X = fingerprint_matrix(
        eligible["canonical_smiles"].tolist(),
        bundle.features.fingerprint,
        ecfp_bits=bundle.features.ecfp_bits,
    )
    sizes = X.sum(axis=1)
    remaining = set(range(len(eligible)))
    chosen: list[dict[str, Any]] = []
    for _ in range(min(n, len(eligible))):
        best_index = -1
        best_score = -math.inf
        best_similarity = 0.0
        for index in remaining:
            similarity = 0.0
            for previous in chosen:
                old = int(previous["candidate_index"])
                overlap = int(np.dot(X[index].astype(np.uint16), X[old].astype(np.uint16)))
                union = int(sizes[index] + sizes[old] - overlap)
                similarity = max(similarity, overlap / union if union else 0.0)
            score = float(eligible.iloc[index]["predicted_pactivity"]) - (
                diversity_weight * similarity
            )
            if score > best_score:
                best_index, best_score, best_similarity = index, score, similarity
        remaining.remove(best_index)
        chosen.append(
            {
                "candidate_index": best_index,
                "selection_score": best_score,
                "max_shortlist_similarity": best_similarity,
            }
        )
    picked = eligible.iloc[[row["candidate_index"] for row in chosen]].copy()
    picked.insert(0, "rank", range(1, len(picked) + 1))
    picked["selection_score"] = [row["selection_score"] for row in chosen]
    picked["max_shortlist_similarity"] = [row["max_shortlist_similarity"] for row in chosen]
    return picked.reset_index(drop=True)

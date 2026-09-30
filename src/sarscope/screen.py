"""Qualitative PubChem assay screening, separate from potency QSAR.

Only explicit Active/Inactive calls from one AID are modeled. Activity values
and potency conversions are intentionally absent from this module.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sorbent.chem.parse import parse_smiles, standardize, to_inchikey

from sarscope.analysis.features import fingerprint_matrix
from sarscope.analysis.model import evaluate
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.params import ModelParams
from sarscope.sources.pubchem import PubChemClient, parse_cid


@dataclass
class ScreenResult:
    compounds: pd.DataFrame
    predictions: pd.DataFrame
    scores: dict[str, Any]
    audit: dict[str, int]
    metadata: dict[str, str]
    source_rows: list[dict[str, str]]
    source_smiles: dict[int, str]


def curate_outcomes(
    rows: list[dict[str, str]], smiles_by_cid: dict[int, str]
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Standardize and deduplicate one assay; discard contradictory structures."""
    audit: Counter[str] = Counter()
    grouped: dict[str, list[tuple[int, str, str]]] = defaultdict(list)
    for row in rows:
        audit["raw_calls"] += 1
        outcome = row.get("Activity Outcome", "").strip().lower()
        if outcome not in ("active", "inactive"):
            audit["nonbinary_outcome"] += 1
            continue
        cid = parse_cid(row.get("CID", ""))
        if cid is None or cid not in smiles_by_cid:
            audit["missing_cid_or_structure"] += 1
            continue
        mol = parse_smiles(smiles_by_cid[cid])
        if mol is None:
            audit["invalid_structure"] += 1
            continue
        try:
            mol = standardize(mol, canonical_tautomer=True)
        except Exception:  # noqa: BLE001 - a bad PubChem structure must not abort the assay
            audit["invalid_structure"] += 1
            continue
        if mol is None or mol.GetNumAtoms() == 0:
            audit["invalid_structure"] += 1
            continue
        from rdkit import Chem

        smiles = Chem.MolToSmiles(mol)
        key = to_inchikey(mol) or smiles
        grouped[key].append((cid, smiles, outcome))
    records: list[dict[str, Any]] = []
    for key, calls in sorted(grouped.items()):
        labels = {call[2] for call in calls}
        if len(labels) != 1:
            audit["conflicting_structures"] += 1
            continue
        records.append(
            {
                "cid": min(call[0] for call in calls),
                "cids": ";".join(map(str, sorted({call[0] for call in calls}))),
                "smiles": calls[0][1],
                "inchikey": key,
                "outcome": labels.pop(),
                "n_calls": len(calls),
            }
        )
    audit["curated_compounds"] = len(records)
    return pd.DataFrame.from_records(
        records, columns=["cid", "cids", "smiles", "inchikey", "outcome", "n_calls"]
    ), dict(audit)


def run_screen(
    aid: int, client: PubChemClient, *, cv_folds: int = 3, seed: int = 42
) -> ScreenResult:
    rows, meta = client.concise_assay(aid)
    cids = [
        cid
        for row in rows
        if row.get("Activity Outcome", "").strip().lower() in ("active", "inactive")
        if (cid := parse_cid(row.get("CID", ""))) is not None
    ]
    smiles = client.smiles_for_cids(cids)
    return run_screen_snapshot(aid, rows, smiles, meta, cv_folds=cv_folds, seed=seed)


def run_screen_snapshot(
    aid: int,
    rows: list[dict[str, str]],
    smiles: dict[int, str],
    meta: dict[str, str],
    *,
    cv_folds: int = 3,
    seed: int = 42,
) -> ScreenResult:
    if str(aid) != meta.get("aid") or any(row.get("AID") != str(aid) for row in rows):
        raise ValueError("snapshot AID does not match the requested assay")
    compounds, audit = curate_outcomes(rows, smiles)
    if (
        len(compounds) < 30
        or compounds["outcome"].value_counts().min() < 8
        or compounds["outcome"].nunique() != 2
    ):
        raise ValueError("qualitative screen needs at least 30 compounds and 8 per class")
    compounds = add_scaffolds(compounds)
    groups = compounds["murcko"].tolist()
    if len({group for group in groups if group is not None}) < cv_folds + 2:
        raise ValueError("qualitative screen has too few distinct scaffolds for validation")
    X = fingerprint_matrix(compounds["smiles"].tolist(), "ecfp4")
    params = ModelParams(
        algorithms=("random_forest",),
        split="scaffold",
        cv_folds=cv_folds,
        leakage_audit=False,
        seed=seed,
    )
    result = evaluate(X, compounds["outcome"].tolist(), groups, params)
    test = compounds.iloc[result.test_index].copy()
    truth = (test["outcome"] == "active").to_numpy(dtype=int)
    if len(set(truth)) != 2:
        raise ValueError("held-out scaffold fold has only one outcome; choose another assay")
    classes = result.test_score_classes
    if result.test_class_scores is None or "active" not in classes:
        raise ValueError("model did not provide active-class scores")
    probability = result.test_class_scores[:, classes.index("active")]
    test["predicted_outcome"] = result.test_predictions
    test["active_score"] = probability
    test["split"] = "held_out_scaffold"
    prevalence = float(truth.mean())
    top_n = max(1, int(np.ceil(len(test) * 0.1)))
    top_hits = float(truth[np.argsort(-probability, kind="stable")[:top_n]].mean())
    metrics = {
        "n_train": int(len(result.train_index)),
        "n_test": int(len(result.test_index)),
        "n_test_active": int(truth.sum()),
        "test_prevalence": prevalence,
        "test_roc_auc": float(roc_auc_score(truth, probability)) if len(set(truth)) == 2 else None,
        "test_pr_auc": float(average_precision_score(truth, probability)),
        "pr_auc_random_baseline": prevalence,
        "precision_at_10_percent": top_hits,
        "enrichment_at_10_percent": top_hits / prevalence if prevalence else None,
        "test_mcc": float(result.scores.iloc[0]["test_mcc"]),
        "cv_mcc": float(result.scores.iloc[0]["cv_mcc"]),
        "model": "random_forest",
        "fingerprint": "ecfp4",
        "split": "scaffold",
        "seed": seed,
        "cv_folds": cv_folds,
    }
    meta.setdefault("retrieved_at_utc", datetime.now(UTC).isoformat())
    meta["evaluated_at_utc"] = datetime.now(UTC).isoformat()
    meta["outcome_definition"] = "PubChem Active vs Inactive within one AID"
    meta["raw_rows_sha256"] = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    meta["source_smiles_sha256"] = hashlib.sha256(
        json.dumps(smiles, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ScreenResult(compounds, test, metrics, audit, meta, rows, smiles)


def write_screen(result: ScreenResult, out: Path) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    (out / "source_snapshot.json").write_text(
        json.dumps(
            {
                "assay": result.metadata,
                "rows": result.source_rows,
                "smiles_by_cid": result.source_smiles,
            },
            indent=2,
        )
        + "\n"
    )
    result.compounds.to_csv(out / "qualitative_compounds.csv", index=False)
    result.predictions.to_csv(out / "held_out_predictions.csv", index=False)
    (out / "screen_metrics.json").write_text(
        json.dumps(
            {"assay": result.metadata, "curation": result.audit, "metrics": result.scores},
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return out / "screen_metrics.json"


def read_screen_snapshot(
    path: Path,
) -> tuple[int, list[dict[str, str]], dict[int, str], dict[str, str]]:
    payload = json.loads(path.read_text())
    meta = payload["assay"]
    aid = int(meta["aid"])
    rows = payload["rows"]
    digest = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != meta["raw_rows_sha256"]:
        raise ValueError("PubChem screen snapshot hash mismatch")
    smiles = {int(cid): value for cid, value in payload["smiles_by_cid"].items()}
    smiles_digest = hashlib.sha256(
        json.dumps(smiles, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if smiles_digest != meta["source_smiles_sha256"]:
        raise ValueError("PubChem structure snapshot hash mismatch")
    return aid, rows, smiles, meta

"""Interpretable physicochemical-descriptor models.

Fingerprint models are useful predictors but a numbered fingerprint bit is not
an actionable medicinal-chemistry explanation.  This module instead fits a
small model to named RDKit descriptors on an already-decided train/test split.
It reports held-out performance, model-agnostic permutation importance and,
for Random Forest, impurity importance.  Optional SHAP support is imported only
when explicitly requested so hosted apps do not pay its startup cost.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, Lipinski, rdMolDescriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sarscope.analysis.regression import regression_metrics
from sarscope.analysis.representations import descriptor_matrix

EXPLANATION_DESCRIPTORS: tuple[str, ...] = (
    "MW",
    "logP",
    "TPSA",
    "RB",
    "NumHDonors",
    "NumHAcceptors",
    "HeavyAtoms",
    "FractionCSP3",
    "RingCount",
    "AromaticRings",
    "MolarRefractivity",
    "FormalCharge",
)


@dataclass
class DescriptorExplanation:
    algorithm: str
    metrics: pd.DataFrame
    predictions: pd.DataFrame
    permutation: pd.DataFrame
    intrinsic: pd.DataFrame
    shap_global: pd.DataFrame
    shap_local: pd.DataFrame
    shap_base_value: float | None
    shap_status: str


def add_explanation_descriptors(table: pd.DataFrame) -> pd.DataFrame:
    """Add named, low-dimensional RDKit descriptors to a curated table."""
    rows: list[dict[str, float]] = []
    for molecule_id, smiles in zip(table["molecule_id"], table["smiles"], strict=True):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            raise ValueError(f"cannot parse curated SMILES for {molecule_id}: {smiles!r}")
        rows.append(
            {
                "MW": float(Descriptors.MolWt(molecule)),
                "logP": float(Crippen.MolLogP(molecule)),
                "TPSA": float(Descriptors.TPSA(molecule)),
                "RB": float(Descriptors.NumRotatableBonds(molecule)),
                "NumHDonors": float(Descriptors.NumHDonors(molecule)),
                "NumHAcceptors": float(Descriptors.NumHAcceptors(molecule)),
                "HeavyAtoms": float(molecule.GetNumHeavyAtoms()),
                "FractionCSP3": float(rdMolDescriptors.CalcFractionCSP3(molecule)),
                "RingCount": float(Lipinski.RingCount(molecule)),
                "AromaticRings": float(Lipinski.NumAromaticRings(molecule)),
                "MolarRefractivity": float(Crippen.MolMR(molecule)),
                "FormalCharge": float(Chem.GetFormalCharge(molecule)),
            }
        )
    values = pd.DataFrame(rows, index=table.index, columns=list(EXPLANATION_DESCRIPTORS))
    existing = table.drop(columns=[c for c in EXPLANATION_DESCRIPTORS if c in table])
    return pd.concat([existing, values], axis=1)


def _importance_frame(
    names: tuple[str, ...], values: NDArray[Any], sd: NDArray[Any] | None = None
) -> pd.DataFrame:
    frame = pd.DataFrame({"descriptor": names, "importance": np.asarray(values, dtype=float)})
    if sd is not None:
        frame["importance_sd"] = np.asarray(sd, dtype=float)
    return frame.sort_values("importance", ascending=False).reset_index(drop=True)


def explain_descriptor_model(
    table: pd.DataFrame,
    train_index: NDArray[np.intp],
    test_index: NDArray[np.intp],
    *,
    algorithm: str = "random_forest",
    permutation_repeats: int = 5,
    seed: int = 42,
    compute_shap: bool = False,
    shap_max_samples: int = 100,
    descriptor_names: tuple[str, ...] | None = None,
    backend: str = "rdkit",
) -> DescriptorExplanation:
    """Fit on ``train_index`` and explain performance on ``test_index`` only."""
    if algorithm not in {"random_forest", "mlp"}:
        raise ValueError("descriptor explanation algorithm must be random_forest or mlp")
    names = descriptor_names if descriptor_names is not None else EXPLANATION_DESCRIPTORS
    described = table if descriptor_names is not None else add_explanation_descriptors(table)
    X = (
        descriptor_matrix(table["smiles"].astype(str).tolist(), names, backend=backend)
        if descriptor_names is not None
        else described[list(names)].to_numpy(dtype=float)
    )
    y = described["pactivity"].to_numpy(dtype=float)
    X_train, X_test = X[train_index], X[test_index]
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(X_train)
    X_train, X_test = imputer.transform(X_train), imputer.transform(X_test)
    y_train, y_test = y[train_index], y[test_index]

    if algorithm == "random_forest":
        estimator: Any = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1,
        )
    else:
        estimator = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                alpha=0.01,
                early_stopping=len(X_train) >= 20,
                max_iter=500,
                random_state=seed,
            ),
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X_train, y_train)
        predicted = np.asarray(estimator.predict(X_test), dtype=float)

    r2, rmse, rho = regression_metrics(y_test, predicted)
    metrics = pd.DataFrame(
        [{"test_r2": r2, "test_rmse": rmse, "test_spearman": rho, "n_test": len(y_test)}]
    )
    predictions = pd.DataFrame(
        {
            "molecule_id": described.iloc[test_index]["molecule_id"].astype(str).to_numpy(),
            "pactivity": y_test,
            "predicted_pactivity": predicted,
            "residual": y_test - predicted,
        }
    )
    perm = permutation_importance(
        estimator,
        X_test,
        y_test,
        n_repeats=permutation_repeats,
        random_state=seed,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    permutation = _importance_frame(names, perm.importances_mean, perm.importances_std)

    intrinsic = pd.DataFrame(columns=["descriptor", "importance"])
    if algorithm == "random_forest":
        intrinsic = _importance_frame(names, estimator.feature_importances_)

    shap_global = pd.DataFrame(columns=["descriptor", "importance"])
    shap_local = pd.DataFrame()
    shap_base_value: float | None = None
    shap_status = "not requested"
    if compute_shap and algorithm != "random_forest":
        shap_status = "SHAP is available only for the Random Forest descriptor model."
    elif compute_shap:
        try:
            import shap
        except ImportError:
            shap_status = "Install the optional dependency with: pip install 'sarscope[explain]'"
        else:
            sample_count = min(shap_max_samples, len(X_test))
            sample = X_test[:sample_count]
            explainer = shap.TreeExplainer(estimator)
            explanation = explainer(sample)
            values = np.asarray(explanation.values, dtype=float)
            shap_global = _importance_frame(names, np.mean(np.abs(values), axis=0))
            shap_local = pd.DataFrame(values, columns=list(names))
            shap_local.insert(
                0,
                "molecule_id",
                described.iloc[test_index[:sample_count]]["molecule_id"].astype(str).to_numpy(),
            )
            base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
            shap_base_value = float(base_values[0]) if len(base_values) else None
            shap_status = f"computed for {sample_count} held-out compounds"

    return DescriptorExplanation(
        algorithm=algorithm,
        metrics=metrics,
        predictions=predictions,
        permutation=permutation,
        intrinsic=intrinsic,
        shap_global=shap_global,
        shap_local=shap_local,
        shap_base_value=shap_base_value,
        shap_status=shap_status,
    )

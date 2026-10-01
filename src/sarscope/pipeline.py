"""Source -> curated table -> every analysis -> RunResults. No I/O except the source.

``analyse`` is source-agnostic and is where the order of operations lives:

    1. add_descriptors, add_scaffolds
    2. describe_groups, property_pca
    3. diversity_table, enrichment_table
    4. decompose_top_scaffolds
    5. per fingerprint in params.landscape.fingerprints:
       bit_vectors -> sas_map; then consensus
    6. fingerprint_matrix(params.model.features.fingerprint) -> evaluate
    7. pca_bounding_box on the leak-free train/test split, using the feature
       filter fitted on the training rows

A dataset with a single class, or fewer than ``cv_folds`` molecules in the
smallest class, cannot be modelled: skip steps 6-7, record why in
``RunResults.skipped``, and still return everything else. Losing a whole
report because one step is impossible is worse than a report with a hole in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sarscope import provenance
from sarscope.analysis.descriptors import add_descriptors
from sarscope.analysis.domain import DomainResult, pca_bounding_box
from sarscope.analysis.features import bit_vectors, feature_filter, fingerprint_matrix
from sarscope.analysis.landscape import SasResult, consensus, sas_map
from sarscope.analysis.mmp import matched_molecular_pairs, summarise_transformations
from sarscope.analysis.model import ModelResult, _cv_folds, evaluate
from sarscope.analysis.profile import GroupProfile, PcaResult, describe_groups, property_pca
from sarscope.analysis.regression import (
    RegressionResult,
    evaluate_regression,
    fit_deployment_model,
    regression_metrics,
)
from sarscope.analysis.representations import model_feature_names, model_matrix, resolved_features
from sarscope.analysis.rgroups import ScaffoldSar, decompose_top_scaffolds
from sarscope.analysis.scaffolds import add_scaffolds, diversity_table, enrichment_table
from sarscope.curate import (
    CurationResult,
    curate_chembl,
    curate_table,
    source_safe_table,
    time_safe_table,
)
from sarscope.params import RunParams
from sarscope.predict import PredictionBundle, _row_max_tanimoto, similarity_domain_threshold
from sarscope.sources.chembl import ChemblClient
from sarscope.sources.table import read_activity_table


@dataclass
class RunResults:
    params: RunParams
    curation: CurationResult
    #: Curated table plus descriptor and scaffold columns.
    table: pd.DataFrame
    profile: GroupProfile | None
    pca: PcaResult
    diversity: pd.DataFrame
    enrichment: pd.DataFrame
    landscapes: dict[str, SasResult]
    consensus_cliffs: pd.DataFrame
    consensus_generators: list[str]
    models: ModelResult | None
    regression: RegressionResult | None
    domain: DomainResult | None
    matched_pairs: pd.DataFrame
    transformation_summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Held-out, per-compound predictions with activity-cliff membership.
    model_test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Aggregate held-out performance on cliff compounds versus all others.
    cliff_model_performance: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Refit-on-all-data continuous model written to the report for prediction.
    prediction_bundle: PredictionBundle | None = None
    #: Rows actually passed to modelling (may differ from the full SAR table).
    model_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    regression_test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    #: Per-scaffold R-group decomposition, largest series first.
    rgroups: list[ScaffoldSar] = field(default_factory=list)
    #: step name -> reason, for anything that could not run.
    skipped: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def analyse(curation: CurationResult, params: RunParams) -> RunResults:
    table = add_scaffolds(add_descriptors(curation.table))
    skipped: dict[str, str] = {}

    # A set that is all potent, or all inactive, has no two groups to compare.
    # That is a real dataset shape (a focused in-house series often is), so the
    # comparison is skipped and everything that does not need it still runs.
    groups = sorted(table["group"].unique())
    profile: GroupProfile | None = None
    if len(groups) == 2:
        profile = describe_groups(table)
    else:
        skipped["profile"] = (
            f"every molecule is in Group {groups[0]}, so there are no two groups to "
            "compare. Descriptor values are still in the curated dataset."
        )
    pca = property_pca(table)
    diversity = diversity_table(table)
    enrichment = enrichment_table(table)
    rgroups = decompose_top_scaffolds(table, enrichment)
    matched_pairs = matched_molecular_pairs(
        table,
        max_variable_heavy_atoms=params.matched_pairs.max_variable_heavy_atoms,
        max_pairs=params.matched_pairs.max_pairs,
    )

    landscapes: dict[str, SasResult] = {}
    for name in params.landscape.fingerprints:
        fps = bit_vectors(table["smiles"].tolist(), name, ecfp_bits=params.model.features.ecfp_bits)
        landscapes[name] = sas_map(
            fps,
            table["pactivity"].tolist(),
            table["molecule_id"].tolist(),
            params.landscape,
            fingerprint_name=name,
        )
    consensus_cliffs, consensus_generators = consensus(landscapes, params.landscape.generator_sd)

    models: ModelResult | None = None
    regression: RegressionResult | None = None
    domain: DomainResult | None = None
    prediction_bundle: PredictionBundle | None = None
    model_test_predictions = pd.DataFrame()
    cliff_model_performance = pd.DataFrame()
    regression_test_predictions = pd.DataFrame()
    model_table = table
    split_error = ""
    if params.model.split in ("time", "source"):
        try:
            safe = (
                time_safe_table(curation, params)
                if params.model.split == "time"
                else source_safe_table(curation, params)
            )
            model_table = add_scaffolds(safe)
        except ValueError as exc:
            split_error = str(exc)
    reason = split_error or _modelling_blocked(model_table, params)
    if reason:
        skipped["model"] = reason
        skipped["domain"] = reason
    else:
        X = model_matrix(model_table["smiles"].tolist(), params.model.features)
        domain_X = fingerprint_matrix(
            model_table["smiles"].tolist(),
            params.model.features.fingerprint,
            ecfp_bits=params.model.features.ecfp_bits,
        )
        years = model_table["document_year"].tolist() if "document_year" in model_table else None
        source_test = model_table["source_test"].tolist() if "source_test" in model_table else None
        models = evaluate(
            X,
            model_table["activity_class"].tolist(),
            model_table["murcko"].tolist(),
            params.model,
            years=years,
            source_test=source_test,
        )
        regression = evaluate_regression(
            X,
            model_table["pactivity"].tolist(),
            model_table["murcko"].tolist(),
            model_table["activity_class"].tolist(),
            params.model,
            years=years,
            source_test=source_test,
        )
        regression_test_predictions = regression_error_profile(model_table, domain_X, regression)
        deploy_filter, deploy_model = fit_deployment_model(
            X, model_table["pactivity"].tolist(), regression.best_algorithm, params.model
        )
        prediction_bundle = PredictionBundle(
            algorithm=regression.best_algorithm,
            features=resolved_features(params.model.features),
            feature_filter=deploy_filter,
            estimator=deploy_model,
            train_fingerprints=domain_X,
            similarity_threshold=similarity_domain_threshold(domain_X),
            train_ids=model_table["molecule_id"].astype(str).tolist(),
            train_smiles=model_table["smiles"].astype(str).tolist(),
            train_pactivity=model_table["pactivity"].astype(float).tolist(),
            canonical_tautomer=params.curation.canonical_tautomer,
            empirical_half_width=regression.empirical_half_width,
            feature_names=model_feature_names(params.model.features),
        )
        model_test_predictions, cliff_model_performance = _cliff_model_errors(
            model_table, landscapes, models, regression
        )
        filt = feature_filter(params.model.features).fit(X[models.train_index])
        domain = pca_bounding_box(
            filt.transform(X[models.train_index]), filt.transform(X[models.test_index])
        )

    return RunResults(
        params=params,
        curation=curation,
        table=table,
        profile=profile,
        pca=pca,
        diversity=diversity,
        enrichment=enrichment,
        landscapes=landscapes,
        consensus_cliffs=consensus_cliffs,
        consensus_generators=consensus_generators,
        models=models,
        regression=regression,
        domain=domain,
        matched_pairs=matched_pairs,
        transformation_summary=summarise_transformations(matched_pairs, curation.evidence),
        model_test_predictions=model_test_predictions,
        cliff_model_performance=cliff_model_performance,
        prediction_bundle=prediction_bundle,
        model_table=model_table,
        regression_test_predictions=regression_test_predictions,
        rgroups=rgroups,
        skipped=skipped,
    )


def _cliff_model_errors(
    table: pd.DataFrame,
    landscapes: dict[str, SasResult],
    classification: ModelResult | None,
    regression: RegressionResult,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join held-out predictions to activity-cliff membership and summarise."""
    cliff_ids: set[str] = set()
    for result in landscapes.values():
        cliff_ids.update(result.cliffs["id_a"].astype(str))
        cliff_ids.update(result.cliffs["id_b"].astype(str))

    # Both evaluators deliberately use the same deterministic outer split.
    if classification is not None and not np.array_equal(
        classification.test_index, regression.test_index
    ):
        raise RuntimeError("classification and regression test splits differ")
    test = table.iloc[regression.test_index]
    predictions = pd.DataFrame(
        {
            "molecule_id": test["molecule_id"].astype(str).to_numpy(),
            "is_cliff_compound": test["molecule_id"].astype(str).isin(cliff_ids).to_numpy(),
            "pactivity": regression.test_truth,
            "predicted_pactivity": regression.test_predictions,
            "absolute_error": np.abs(regression.test_truth - regression.test_predictions),
            "activity_class": test["activity_class"].astype(str).to_numpy(),
            "predicted_activity_class": (
                np.asarray(classification.test_predictions, dtype=str)
                if classification is not None
                else np.full(len(test), "not evaluated")
            ),
        }
    )
    predictions["classification_correct"] = (
        predictions["activity_class"] == predictions["predicted_activity_class"]
        if classification is not None
        else np.nan
    )

    rows: list[dict[str, Any]] = []
    for is_cliff in (True, False):
        group = predictions[predictions["is_cliff_compound"] == is_cliff]
        if len(group):
            r2, rmse, rho = regression_metrics(
                group["pactivity"].to_numpy(), group["predicted_pactivity"].to_numpy()
            )
            accuracy = float(group["classification_correct"].mean())
            mae = float(group["absolute_error"].mean())
        else:
            r2 = rmse = rho = accuracy = mae = float("nan")
        rows.append(
            {
                "subset": "cliff compounds" if is_cliff else "other compounds",
                "n": len(group),
                "classification_accuracy": accuracy,
                "regression_r2": r2,
                "regression_rmse": rmse,
                "regression_mae": mae,
                "regression_spearman": rho,
            }
        )
    return predictions, pd.DataFrame(rows)


def regression_error_profile(
    table: pd.DataFrame, fingerprints: np.ndarray, result: RegressionResult
) -> pd.DataFrame:
    """Held-out regression error versus nearest training compound similarity."""
    train = fingerprints[result.train_index]
    nearest = _row_max_tanimoto(fingerprints[result.test_index], train)
    threshold = similarity_domain_threshold(train)
    test = table.iloc[result.test_index]
    frame = pd.DataFrame(
        {
            "molecule_id": test["molecule_id"].astype(str).to_numpy(),
            "document_year": test["document_year"].to_numpy(),
            "pactivity": result.test_truth,
            "predicted_pactivity": result.test_predictions,
            "absolute_error": np.abs(result.test_truth - result.test_predictions),
            "residual": result.test_truth - result.test_predictions,
            "activity_class": test["activity_class"].to_numpy(),
            "murcko": test["murcko"].to_numpy(),
            "within_3_fold": np.abs(result.test_truth - result.test_predictions) <= np.log10(3),
            "within_10_fold": np.abs(result.test_truth - result.test_predictions) <= 1.0,
            "max_training_similarity": nearest,
            "in_training_domain": nearest >= threshold,
        }
    )
    if math.isfinite(result.empirical_half_width):
        frame["empirical_lower_pactivity"] = result.test_predictions - result.empirical_half_width
        frame["empirical_upper_pactivity"] = result.test_predictions + result.empirical_half_width
        frame["inside_empirical_band"] = frame["absolute_error"] <= result.empirical_half_width
    return frame


def _modelling_blocked(table: pd.DataFrame, params: RunParams) -> str:
    """Why the model step cannot run, or "" when it can.

    Checked before any fitting so a small dataset produces a report with a
    stated gap, rather than a traceback three minutes in. Both limits bite on
    real data: a focused in-house set often has few scaffolds, and a scaffold
    split cannot make more folds than there are scaffolds.
    """
    model = params.model
    counts = table["activity_class"].value_counts()
    if len(counts) < 2:
        return (
            f"every molecule falls in one activity class ({counts.index[0]}), "
            "so there is nothing to classify. Widen the class bounds, or check "
            "that the potency column is on the scale you think it is."
        )
    if model.split not in ("time", "source") and counts.min() < model.cv_folds:
        return (
            f"the smallest class has {counts.min()} molecules, fewer than the "
            f"{model.cv_folds} cross-validation folds. Lower cv_folds, or widen "
            "the class bounds so the rarest class is better populated."
        )
    if model.split == "source" and "source_test" not in table:
        return "source split needs ChEMBL measurement origins"
    if model.split in ("scaffold", "source"):
        # Acyclic molecules are each their own group, so they only ever help.
        groups = int(table["murcko"].nunique() + table["murcko"].isna().sum())
        outer = max(2, round(1 / model.test_fraction))
        # The outer split takes its share of scaffolds first, so cross-validation
        # runs on roughly (1 - test_fraction) of them and needs cv_folds of those.
        needed = (
            model.cv_folds
            if model.split == "source"
            else max(outer, math.ceil(model.cv_folds / (1 - model.test_fraction)))
        )
        if model.split == "source":
            train = table[~table["source_test"]]
            groups = int(train["murcko"].nunique() + train["murcko"].isna().sum())
        if groups < needed:
            if model.split == "source":
                return (
                    f"source holdout training set needs at least {needed} distinct "
                    f"scaffolds for {model.cv_folds}-fold CV and has {groups}; "
                    "use fewer folds or choose another held-out origin"
                )
            return (
                f"a {model.split} split with {model.cv_folds} scaffold CV folds needs about "
                f"{needed} distinct "
                f"scaffolds and this dataset has {groups}. Lower cv_folds, or pass "
                "--split random (which will flatter the model, because one series can "
                "then sit on both sides of the split)."
            )
    if model.split == "source":
        train = table[~table["source_test"]]
        test = table[table["source_test"]]
        if train.empty or test.empty:
            return "source split needs training and distinct held-out-origin compounds"
        train_counts = train["activity_class"].value_counts()
        if len(train_counts) < 2 or train_counts.min() < model.cv_folds:
            return (
                f"source split training set cannot support {model.cv_folds}-fold "
                "stratified scaffold model selection; use fewer folds"
            )
    if model.split == "time":
        if "document_year" not in table:
            return (
                "a time split needs document years. ChEMBL runs carry them automatically; "
                "for a table, pass --input-year-col COLUMN."
            )
        years = pd.to_numeric(table["document_year"], errors="coerce")
        if years.isna().any():
            return (
                f"a time split needs a known document year for every molecule; "
                f"{int(years.isna().sum())} are missing"
            )
        train = table[years <= model.time_cutoff]
        test = table[years > model.time_cutoff]
        if train.empty or test.empty:
            return (
                f"the time split at {model.time_cutoff} leaves {len(train)} training and "
                f"{len(test)} later test molecules"
            )
        try:
            list(
                _cv_folds(
                    train["activity_class"].to_numpy(),
                    None,
                    model,
                    train["document_year"].tolist(),
                )
            )
        except ValueError as exc:
            return str(exc)
    return ""


def run_chembl(target_id: str, params: RunParams, client: ChemblClient) -> RunResults:
    """Fetch (via the client's cache), curate_chembl, analyse, attach provenance."""
    target = client.target(target_id)
    records = client.activities(target_id, params.curation.standard_types)
    if params.curation.min_confidence_score is not None:
        scores = client.assay_confidences(target_id)
        records = [
            {**record, "confidence_score": scores.get(str(record.get("assay_chembl_id")))}
            for record in records
        ]
    results = analyse(curate_chembl(records, params), params)
    results.provenance = provenance.collect(
        params, provenance.chembl_source(target, client.release, len(records))
    )
    return results


def run_table(path: Path, params: RunParams, **read_kwargs: Any) -> RunResults:
    """read_activity_table, curate_table, analyse, attach provenance."""
    measurements = read_activity_table(path, **read_kwargs)
    results = analyse(curate_table(measurements, params), params)
    results.provenance = provenance.collect(params, provenance.table_source(path))
    return results

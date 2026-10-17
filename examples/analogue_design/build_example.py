"""Build the measured demo from downloaded ChEMBL JSON; never synthesise labels.

The source directory must contain CHEMBL944276.json, metadata-944276.json,
document-944276.json and status.json. URLs are recorded in the output source.
This script runs one fixed RF evaluation, not a hyperparameter/seed search.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sarscope import provenance
from sarscope.analogue_design import fit_design_model, generate_analogues, score_analogues
from sarscope.analysis.rgroups import decompose_scaffold
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.curate import curate_chembl
from sarscope.field_qsar import assay_contexts, dataset_from_assay, dataset_from_upload
from sarscope.params import AnalogueDesignParams, CurationParams, ModelParams, RunParams

FILES = ("egfr_ic50.csv", "measurements.csv", "source.json", "verification.json")
EVIDENCE = (
    "molecule_id",
    "source_molecule_id",
    "record_id",
    "pactivity",
    "smiles",
    "assay_chembl_id",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "assay_variant_mutation",
    "document_chembl_id",
    "document_year",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if any((args.output_dir / name).exists() for name in FILES):
        parser.error("Use a fresh output directory; frozen demo files are not overwritten.")
    paths = [
        args.source_dir / name
        for name in (
            "CHEMBL944276.json",
            "metadata-944276.json",
            "document-944276.json",
            "status.json",
        )
    ]
    raw, assay, document, status = [json.loads(path.read_bytes()) for path in paths]
    if (
        raw["page_meta"]["next"] is not None
        or len(raw["activities"]) != raw["page_meta"]["total_count"]
    ):
        raise ValueError("The complete assay response is required, not a truncated page.")
    if (
        assay["assay_chembl_id"] != "CHEMBL944276"
        or document["document_chembl_id"] != assay["document_chembl_id"]
    ):
        raise ValueError("The metadata do not identify the selected assay/document.")
    curation = CurationParams(canonical_tautomer=False)
    curated = curate_chembl(raw["activities"], RunParams(curation=curation))
    contexts = assay_contexts(curated.evidence)
    if len(contexts) != 1:
        raise ValueError("The demo must have exactly one retained assay/endpoint/variant context.")
    context = contexts.iloc[0]
    scoped = dataset_from_assay(
        curated.table,
        curated.evidence,
        context.assay_chembl_id,
        context.standard_type,
        context.variant_scope,
    )
    csv_data = scoped.table[["molecule_id", "smiles", "pactivity"]].to_csv(index=False).encode()
    evidence_data = scoped.measurements[list(EVIDENCE)].to_csv(index=False).encode()
    # Verify the actual serialized upload path, not a different in-memory dataset.
    loaded = dataset_from_upload(
        pd.read_csv(io.BytesIO(csv_data)),
        [],
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="IC50",
        source_sha256=hashlib.sha256(csv_data).hexdigest(),
    )
    evidence = pd.read_csv(io.BytesIO(evidence_data), dtype={"record_id": str})
    table = add_scaffolds(loaded.table)
    settings = ModelParams(
        regression_algorithms=("random_forest",),
        split="random",
        cv_folds=3,
        test_fraction=0.25,
        oversample=False,
        leakage_audit=False,
    )
    model = fit_design_model(table, evidence, AnalogueDesignParams(), settings)
    recommendation = {
        "scaffold": "c1ccc(Cc2ncnc3ccccc23)cc1",
        "parent_id": "CHEMBL522134",
        "reason": (
            "Three variable positions and several unmeasured combinations; "
            "not selected by predicted potency."
        ),
    }
    series = decompose_scaffold(table, recommendation["scaffold"], min_substituent_count=1)
    if series is None:
        raise ValueError("Recommended scaffold could not be decomposed.")
    designed = generate_analogues(
        series,
        table,
        evidence,
        recommendation["parent_id"],
        series.positions,
        AnalogueDesignParams(),
    )
    scored = score_analogues(designed, model)
    unmeasured = scored.candidates.known_ids.eq("")
    source = {
        "kind": "bundled_chembl_assay",
        "release": status["chembl_db_version"],
        "retrieved_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "target_id": assay["target_chembl_id"],
        "target_name": "EGFR",
        "assay": assay["assay_chembl_id"],
        "description": assay["description"],
        "confidence_score": assay["confidence_score"],
        "endpoint": "IC50",
        "variant": context.variant_scope,
        "document": {
            key: document.get(key)
            for key in ("document_chembl_id", "title", "authors", "journal", "year", "doi")
        },
        "urls": {
            "activities": "https://www.ebi.ac.uk/chembl/api/data/activity.json?assay_chembl_id=CHEMBL944276&limit=1000",
            "assay": "https://www.ebi.ac.uk/chembl/api/data/assay/CHEMBL944276.json",
            "document": "https://www.ebi.ac.uk/chembl/api/data/document/CHEMBL1140899.json",
            "status": "https://www.ebi.ac.uk/chembl/api/data/status.json",
        },
        "license": "CC-BY-SA-3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "attribution": (
            "Data from ChEMBL, EMBL-EBI; original source La Motta et al., "
            "J Med Chem 2009, doi:10.1021/jm800829v."
        ),
        "conversion": (
            "pIC50 = 9 - log10(IC50 in nM); retained same-assay exact records, "
            "median per curated isomeric structure; no fabricated or imputed activities."
        ),
        "comparability_note": (
            "One ChEMBL assay annotation from a retrospective computational study. "
            "Underlying experimental protocol comparability was not independently verified. "
            "Unreported variants are not confirmed wild type."
        ),
        "curation": asdict(curation),
        "curation_steps": [asdict(step) for step in curated.steps],
        "input_records": len(raw["activities"]),
        "retained_records": len(evidence),
        "compounds": len(table),
        "recommended": recommendation,
        "files_sha256": {
            "egfr_ic50.csv": hashlib.sha256(csv_data).hexdigest(),
            "measurements.csv": hashlib.sha256(evidence_data).hexdigest(),
        },
        "download_sha256": {path.name: provenance.file_sha256(path) for path in paths},
        "builder_sha256": provenance.file_sha256(Path(__file__)),
        "selection_note": (
            "Selected after examining four EGFR assay examples for usable SAR and "
            "fixed-default RF performance. Demonstration selection is not independent "
            "benchmark validation. No seed, hyperparameter, cutoff or potency-based "
            "row selection was used."
        ),
    }
    verification = {
        "python": provenance.collect(RunParams(), {})["python"],
        "packages": provenance.package_versions(),
        "model_settings": asdict(settings),
        "scoring_checks": json.loads(model.checks.to_json(orient="records")),
        "scores": json.loads(model.regression.scores.to_json(orient="records")),
        "walkthrough": {
            **recommendation,
            "positions": series.positions,
            "series_compounds": series.n_molecules,
            "proposals": len(scored.candidates),
            "unmeasured": int(unmeasured.sum()),
            "scored_unmeasured": int(
                (unmeasured & scored.candidates.predicted_pactivity.notna()).sum()
            ),
        },
        "limitation": (
            "Selected retrospective random-split example, not independent/prospective "
            "validation or evidence of improved leads. Broad assay model performance does "
            "not establish reliable within-series potency differences. Supporting "
            "substitutions may have only one measured pair."
        ),
    }
    if not model.checks.passed.all() or verification["walkthrough"]["scored_unmeasured"] < 1:
        raise ValueError("This snapshot does not reproduce a scoreable analogue demonstration.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in zip(
        FILES,
        (
            csv_data,
            evidence_data,
            json.dumps(source, indent=2, allow_nan=False).encode() + b"\n",
            json.dumps(verification, indent=2, allow_nan=False).encode() + b"\n",
        ),
        strict=True,
    ):
        (args.output_dir / name).write_bytes(data)
    print(
        json.dumps({"compounds": len(table), "walkthrough": verification["walkthrough"]}, indent=2)
    )


if __name__ == "__main__":
    main()

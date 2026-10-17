"""Build a focused measured 3D tutorial; no fabricated activities or model tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

from sarscope import provenance
from sarscope.analysis.scaffolds import add_scaffolds
from sarscope.curate import curate_chembl
from sarscope.field_qsar import (
    assay_contexts,
    dataset_from_assay,
    dataset_from_upload,
    prepare_field_series,
    run_field_qsar,
    subset_dataset,
    upload_table,
)
from sarscope.field_qsar_report import aligned_sdf
from sarscope.params import CurationParams, FieldQsarParams, ModelParams, RunParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if (args.output_dir / "egfr_3d.csv").exists():
        parser.error("Use a fresh directory; the example will not be overwritten.")
    paths = [
        args.source_dir / name
        for name in (
            "assay-680017.json",
            "metadata-680017.json",
            "document-680017.json",
            "status.json",
        )
    ]
    raw, assay, document, status = [json.loads(p.read_bytes()) for p in paths]
    if raw["page_meta"]["next"] is not None:
        raise ValueError("Supply the complete assay response.")
    curated = curate_chembl(
        raw["activities"], RunParams(curation=CurationParams(canonical_tautomer=False))
    )
    context = assay_contexts(curated.evidence).iloc[0]
    dataset = dataset_from_assay(
        curated.table,
        curated.evidence,
        context.assay_chembl_id,
        context.standard_type,
        context.variant_scope,
    )
    table = add_scaffolds(dataset.table)
    # Structural selection only; do not inspect test scores to select rows.
    scaffold = "c1ccc(Nc2ncnc3ccccc23)cc1"
    ids = table[table.murcko.eq(scaffold)].molecule_id.tolist()
    dataset = subset_dataset(dataset, ids)
    reference = ids.index("CHEMBL29197")
    params = FieldQsarParams(alignment="rigid_mcs", reference_index=reference, permutations=19)
    source = {
        "kind": "bundled_measured_3d_example",
        "target_id": "CHEMBL203",
        "target_name": "EGFR",
        "assay": assay["assay_chembl_id"],
        "endpoint": "IC50",
        "variant": "Unreported",
        "description": assay["description"],
        "release": status["chembl_db_version"],
        "document": {
            k: document.get(k) for k in ("document_chembl_id", "title", "doi", "year", "authors")
        },
        "license": "CC-BY-SA-3.0",
        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
        "attribution": "ChEMBL/EMBL-EBI; Bridges et al., J Med Chem 1996, doi:10.1021/jm9503613.",
        "conversion": (
            "pIC50 = 9 - log10(IC50 in nM); exact retained records only; "
            "same-assay median aggregation."
        ),
        "selection": (
            "All retained compounds with the specified dominant 4-anilinoquinazoline "
            "Murcko scaffold. No potency/test-score selection."
        ),
        "coordinate_assumption": (
            "Seed-42 generated conformations, rigid MCS aligned to CHEMBL29197; "
            "not crystal structures or experimentally established bound poses."
        ),
        "input_records": len(raw["activities"]),
        "curated_assay_compounds": len(table),
        "compounds": len(dataset.table),
        "reference_id": "CHEMBL29197",
        "scaffold": scaffold,
        "alignment_settings": asdict(params),
        "download_sha256": {p.name: provenance.file_sha256(p) for p in paths},
        "builder_sha256": provenance.file_sha256(Path(__file__)),
    }
    dataset.source = source
    prepared = prepare_field_series(dataset, params)
    csv = dataset.table[["molecule_id", "smiles", "pactivity"]].to_csv(index=False).encode()
    sdf = aligned_sdf(prepared).encode()
    evidence = dataset.measurements.to_csv(index=False).encode()
    source["files_sha256"] = {
        name: hashlib.sha256(data).hexdigest()
        for name, data in (
            ("egfr_3d.csv", csv),
            ("egfr_3d_aligned.sdf", sdf),
            ("measurements.csv", evidence),
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, data in (
        ("egfr_3d.csv", csv),
        ("egfr_3d_aligned.sdf", sdf),
        ("measurements.csv", evidence),
    ):
        (args.output_dir / name).write_bytes(data)
    (args.output_dir / "alignment.csv").write_text(prepared.alignment.report.to_csv(index=False))
    (args.output_dir / "source.json").write_text(
        json.dumps(source, indent=2, allow_nan=False) + "\n"
    )
    print("Prepared measured example:", len(dataset.table), "compounds", flush=True)
    # Evaluate exactly what the bundled SDF loader will use, including its coordinate precision.
    raw_sdf, mols = upload_table(sdf, ".sdf")
    imported = dataset_from_upload(
        raw_sdf,
        mols,
        id_col="molecule_id",
        smiles_col="smiles",
        value_col="pactivity",
        unit="p",
        endpoint="IC50",
        source_sha256=source["files_sha256"]["egfr_3d_aligned.sdf"],
    )
    imported.measurements, imported.source = dataset.measurements, source
    imported_params = replace(params, alignment="prealigned")
    imported_prepared = prepare_field_series(imported, imported_params)
    model = ModelParams(split="random", cv_folds=3, test_fraction=0.25)
    result = run_field_qsar(imported_prepared, model)
    verification = {
        "packages": provenance.package_versions(),
        "statistics": json.loads(result.statistics.to_json(orient="records")),
        "checks": json.loads(result.checks.to_json(orient="records")),
        "status": result.status,
        "settings": asdict(imported_params),
        "model_settings": asdict(model),
        "warning": (
            "3D scores, not the previous demo's 2D RF scores. This is a measured workflow "
            "example, not a claim of a validated/predictive model."
        ),
    }
    (args.output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, allow_nan=False) + "\n"
    )
    print(
        result.statistics[["algorithm", "test_r2", "cv_pooled_q2"]].to_string(index=False),
        flush=True,
    )


if __name__ == "__main__":
    main()

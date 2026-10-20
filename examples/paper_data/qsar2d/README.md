# Frozen 2D regression benchmark

| Target | Curated | Train / held-out | Original test R² | RMSE |
|---|---:|---:|---:|---:|
| BRAF / CHEMBL5145 | 2,917 | 2,333 / 584 | 0.762 | 0.646 |
| DRD2 / CHEMBL217 | 492 | 393 / 99 | 0.310 | 0.902 |
| ER-alpha / CHEMBL206 | 3,258 | 2,607 / 651 | 0.696 | 0.828 |

- `CHEMBL*_compounds_and_split.csv`: SMILES, pIC50 and explicit `split`.
- `CHEMBL*_measurement_context.csv.gz`: retained assay/document context.
- `held_out_predictions.csv`, `target_metrics.csv`, `summary.json`: **original**
  saved predictions, scores and protocol; not a newly fitted benchmark.
- `benchmark-chembl37.tar.gz`: the exact original raw snapshots, source hashes
  and saved outputs. Members are under `runs/benchmark-chembl37/`.

The full-cohort CSVs reconstruct curation from those frozen inputs. Test IDs
come from the original saved predictions, training is the curated complement.
Counts, all saved held-out SMILES/activities and nonempty scaffold separation
were checked. The original full training tables were not separately archived
for byte-level comparison. [Export provenance](export-provenance.json) records
the environments, including ER-alpha's RDKit 2026.3.6 dependency for matching
the saved SMILES representation. Raw snapshots are the authoritative source.

These larger target collections are not the three single-assay SAR cohorts.
Endpoint: exact binding IC50 in nM, aggregated to median pIC50 per standardized
structure. The historical protocol's `wild_type_or_unannotated` label does not
establish wild-type identity. This is one retrospective scaffold split, not
prospective validation or a general ranking of learners.

Uploading the CSV and selecting a new split in the app will **not** automatically
reproduce the historical partition. Preserve the provided membership for an
exact protocol comparison; see the [benchmark notes](../../../docs/benchmark-chembl37.md).

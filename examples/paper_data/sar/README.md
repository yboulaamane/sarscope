# Assay-matched SAR cases

`raw/` contains the exact assay snapshots and original manifest. Each case folder
contains curated `compounds.csv`, retained `measurements.csv`, the final
`selected_series.csv`, `stereo_audit.csv` and `positional_pairs.csv`.

| Case | Assay | Curated | Final series | Positional pairs |
|---|---|---:|---:|---:|
| BRAF | CHEMBL5046398 | 44 | 16 | 29 |
| DRD2 | CHEMBL671074 | 30 | 23 | 36 |
| ER-alpha | CHEMBL829540 | 98 | 53 | 198 |

`compounds.csv` includes `selected_series`. DRD2's final series excludes the
opposite/unassigned core states; it is **23**, not the original scaffold group's
26 compounds. This selection uses chemical/stereochemical membership, not potency.
`analysis/` preserves the reported omission, threshold and permutation results.

These are measured SAR examples, not necessarily large enough for predictive
3D-QSAR. Descriptive repeated-pair support is not statistical significance;
no transformation passed the reported permutation/BH-adjusted threshold.

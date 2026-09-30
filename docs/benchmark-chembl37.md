# Frozen ChEMBL 37 three-family benchmark — first local run

This is a descriptive, **within-target scaffold-held-out** pIC50 benchmark,
not an external prospective or cross-target validation. It was run on
2026-09-30 against frozen [ChEMBL](https://www.ebi.ac.uk/chembl/) release 37
activity records. The fixed panel covers BRAF (kinase), DRD2 (class A GPCR),
and ESR1 (nuclear receptor). Full frozen data and molecule-level predictions
are local ignored artifacts under `runs/benchmark-chembl37/`, not bundled or
redistributed in this MIT repository.

| Target | Family | Raw IC50 | Curated | Train / test | Test R² | Test RMSE | Test Spearman | Median baseline RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| CHEMBL5145 (BRAF) | Kinase | 11,017 | 2,917 | 2,333 / 584 | 0.762 | 0.646 | 0.859 | 1.342 |
| CHEMBL217 (DRD2) | GPCR | 2,480 | 492 | 393 / 99 | 0.310 | 0.902 | 0.574 | 1.087 |
| CHEMBL206 (ESR1) | Nuclear receptor | 7,699 | 3,258 | 2,607 / 651 | 0.696 | 0.828 | 0.836 | 1.503 |

RMSE is in log10 molar potency units. The baseline is the *training-set*
median pIC50 predicted for every held-out molecule. These are single-split
point estimates, not uncertainty intervals. No model or target was chosen by
held-out test performance.

Protocol: exact nM wild-type/unannotated binding IC50, default duplicate and
validity exclusions, median per standardized structure, ECFP4 fingerprint,
one 200-tree random-forest regressor (maximum depth 15), fold-local feature
filtering, 3-fold scaffold-group training CV, 20% outer scaffold holdout,
seed 42. Acyclic compounds are treated as individual scaffold groups. The
snapshot manifest SHA-256 was
`f48c8aec796c8986de3938febbc95b99149355660e1c31315f98852e3395a493`.
The manifest also carries a separate SHA-256 for each target's raw records,
target identity, release, protocol, and package versions. At evaluation the
versions were RDKit 2026.3.6, scikit-learn 1.9.1, NumPy 2.5.3,
pandas 3.0.6, SciPy 1.18.1, and Sorbent 0.1.0.

The weakest target is DRD2; do not cite the panel mean alone. These records
remain retrospective and can share assays, papers, and chemistry series even
when scaffolds do not overlap. The panel has only three handpicked targets,
was run once, and has no external-origin/prospective test or repeated-seed
confidence intervals. A stronger claim requires preregistered independent
source holdouts and additional target families, with assay-context auditing,
repeatability checks, and public versioned benchmark assets subject to the
source-data license.

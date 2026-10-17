# Measured EGFR example (not a validation benchmark)

**Historical 2D example:** retained for provenance, no longer the app's demo.
For **3D QSAR · own compounds**, use the focused 50-compound CSV and aligned
SDF in the neighbouring `field_example/` directory. None of the 2D RF scores
below establish 3D-field performance. The original walkthrough below records
the earlier 2D-only implementation, not current 3D controls.

`egfr_ic50.csv` contains 106 curated isomeric structures and literature-reported
IC50 potencies from ChEMBL 37 assay **CHEMBL944276**, target **CHEMBL203 (EGFR)**.
It uses the upload format `molecule_id,smiles,pactivity`.

Activities are **not synthetic**: pActivity is pIC50, calculated as
`9 - log10(IC50 in nM)`. Only exact, unflagged, non-duplicate binding records
surviving SARscope's recorded curation are included. Structures are standardised
without canonical tautomerisation; replicate aggregation, if needed, is the
same-assay median. No activity-based row selection or imputation was used.
See `source.json` for input/retained counts, filtering, conversion and hashes;
`measurements.csv` maps each value to its original ChEMBL activity record.

## Try it

In Streamlit choose **Design analogues · own compounds**, expand **Try a
measured EGFR example**, and check **Use measured EGFR example**. Alternatively,
download/upload the CSV, select endpoint **IC50**, units **pActivity**, and review
comparability before confirming the upload declaration.

1. Choose shared scaffold `c1ccc(Cc2ncnc3ccccc23)cc1` (preselected for the built-in example).
2. Click **Prepare design series**. This scaffold has nine measured members.
3. Choose parent **CHEMBL522134** (preselected for the built-in example), keep
   all editable positions, and click **Generate analogue proposals**.
4. Expect seven proposals: **six unmeasured combinations and one known control**.
   Supporting substitutions have only one measured pair each: they are weak
   transfer evidence, not confirmed gains. Examine property trade-offs and
   source pairs before interpreting a suggestion.
5. Optionally fit **random_forest**, **random** split, held-out fraction **0.25**,
   **3** CV folds, seed **42** (fixed by the UI), then explicitly score proposals.

The checked Python 3.13 environment produced approximately held-out R² **0.92**,
RMSE **0.41 pIC50 units**, Spearman **0.89**, and pooled training-CV q² **0.59**.
All six unmeasured combinations passed the existing scoring/domain gates.
The broad training-CV residual band is approximately **±1.59 pIC50 units**:
a high test R² does not mean precise potency gains. Re-run settings and exact
versions/checks are in `verification.json`; results may vary with versions.

## What this does NOT establish

This example was selected after examining **four** EGFR assays. The default
RF/seed/split/quality gates were not tuned, but selecting a dataset after
examining its scores still makes this a **selected demonstration**, not an
independent performance estimate. `selection_audit.json` in the repository's
`examples/analogue_design/` directory records the earlier trials.

The source paper is a retrospective computational study using reported
inhibitors, not proof that all underlying measurements used identical
experimental protocols. One ChEMBL assay ID and confidence score 9 establish
the annotation/target assignment, not independent experimental comparability.
Mutation annotation is unreported, **not confirmed wild type**. Random splitting
can put close analogues on both sides; performance across new scaffolds, future
chemistry or an independent assay is not established. Broad assay-level R² does
not establish reliable within-series differences. A predicted increase is a
hypothesis; no proposed compound has been synthesised or tested by SARscope.

This CSV is a **2D SAR/analogue-design example**, not a pre-aligned or validated
3D-QSAR dataset. It does not reproduce the source paper's 3D models.

## Attribution and separate data licence

Data from **ChEMBL, European Molecular Biology Laboratory—European Bioinformatics
Institute (EMBL-EBI)**, available under
[Creative Commons Attribution-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-sa/3.0/).
The CSV, retained measurements and accompanying data metadata in this directory
are redistributed under **CC BY-SA 3.0**, separately from SARscope's MIT-licensed
software. Retain attribution and this licence when redistributing/adapting data.
Changes: filtering, structure standardisation, pIC50 conversion, formatting and
addition of a software walkthrough/evaluation record. No endorsement is implied.

Original publication: La Motta C, Sartini S, Tuccinardi T, Nerini E, Da Settimo F,
Martinelli A. *Computational studies of epidermal growth factor receptor:
docking reliability, three-dimensional quantitative structure-activity
relationship analysis, and virtual screening studies.* J Med Chem.
2009;52:964–975. [doi:10.1021/jm800829v](https://doi.org/10.1021/jm800829v).

Source annotation:
[ChEMBL assay CHEMBL944276](https://www.ebi.ac.uk/chembl/explore/assay/CHEMBL944276).
Builder: `examples/analogue_design/build_example.py` in the SARscope repository.

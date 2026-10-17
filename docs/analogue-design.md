# Design next analogues

SARscope can propose small, traceable experiments around a selected parent.
This is **experimental lead-optimisation decision support**, not a validated
autonomous lead optimiser. Generating a plausible graph or obtaining a higher
predicted potency does not establish an improved compound.

## Start in Streamlit

For ChEMBL data, curate the target, prepare series in **3 · SAR map**, then open
**11 · Design analogues**. Choose one assay/endpoint/structured-variant context,
a prepared scaffold series, a parent and editable R-positions. Activities are
reaggregated within that context; target-wide medians are not substituted.
Unreported mutation annotation is not evidence of wild type.

For your own compounds, choose **Design analogues · own compounds** in the
sidebar. Upload CSV/TSV (`molecule_id,smiles,pactivity`) or SDF with measured
activities, select endpoint/units and confirm comparability. Prepare a shared
scaffold, then choose the parent and positions. Uploaded assay comparability
is explicitly **user-declared**, not verified against ChEMBL. Qualitative
screening calls are not converted into potency values. No receptor, docking,
3D alignment or field-QSAR model is needed.

Click **Generate analogue proposals**. Nothing is generated or trained on
page load. Changing the parent, positions, assay, structures, activities or
limits hides incompatible previous results until you regenerate.

### Optional measured example

The upload workflow includes **Try a measured EGFR example** with a downloadable
[106-compound CSV](../src/sarscope/data/analogue_design/egfr_ic50.csv), original
ChEMBL record links, conversion, separate CC BY-SA 3.0 data licence, verification
results and a [walkthrough](../src/sarscope/data/analogue_design/README.md).
Nothing is auto-loaded on the homepage. Check **Use measured EGFR example**,
then explicitly prepare, generate and optionally fit/score. The preselected
nine-member scaffold and parent CHEMBL522134 yield six unmeasured combinations
plus one measured control. The fixed-default RF passed the current quality and
domain gates in the checked environment (test R² ≈0.92, training-CV q² ≈0.59).

This dataset was selected after examining four assays, so those favourable
scores are **not an independent benchmark**. It is a literature-compiled,
random-split example; underlying experimental protocol comparability was not
independently verified. Broad assay-level model accuracy does not validate
within-series predicted gains. It is not a validated 3D-QSAR example.

## What is proposed

The engine reuses SARscope's positional controlled pairs and exact mapped
R-group reconstruction:

1. Keep the parent's mapped, isomeric core state and every locked fragment.
2. Find substitutions at an editable position supported by measured pairs in
   the **same assay, endpoint and variant context**, with all other R-positions
   held constant within each supporting pair.
3. Apply one substitution per candidate. Only terminal, single-attachment
   fragments within the configured heavy-atom budget are supported.
4. Require the original molecule to round-trip exactly through reconstruction;
   sanitise and reject invalid, disconnected, radical or duplicate candidates.
   No stereochemical enumeration or guessed graph repairs are performed.

By default, a run allows 50 proposals, 500 attempted substitutions, fragments
of at most 10 heavy atoms, 20,000 evidence pairs and an assay corpus of at most
500 compounds. The UI allows up to 100 proposals and 20 fragment heavy atoms.
Skipped attempts and reached limits are recorded. Hydrogen is a substituent.
R labels are decomposition attachment labels, not IUPAC ring numbering.

Both upward and downward observed effects can be proposed: a potency decrease
may accompany a useful property trade-off. Exact structures already in the
selected assay remain labelled as known controls, with their measured values;
they are not claimed as novel. “Not known” means absent from this supplied
corpus, **not** globally novel or patentable.

## Read evidence separately from predictions

Each proposal shows the measured supporting molecular pairs, record identifiers,
assay/endpoint/variant, number of distinct and disjoint pairs, median and
quartiles of their signed potency differences, and conflicting directions.
Repeated assay records do not multiply molecular-pair support.

An observed substitution effect is **not** the proposed compound's activity.
Effects may not transfer across substitution contexts; single-pair support is
particularly weak. Quartiles are descriptive, not confidence intervals. New
compounds have no measured activity assigned to them.

Property comparisons include RDKit molecular weight, calculated logP, TPSA,
rotatable bonds, H-bond donors and acceptors. The nearest measured analogue
and similarity to measured activity-cliff compounds are shown. The default
cliff criteria use the existing ECFP4/Tanimoto landscape, similarity >0.85 and
absolute ΔpActivity >1; identical-fingerprint pairs with large activity
differences are also retained. A neighbour flag uses similarity ≥ the chosen
similarity threshold. These are heuristics, not calibrated uncertainty or a
guarantee of biological similarity.

PAINS/Brenk substructure alerts are review prompts, not toxicity assessments.
No alert does not establish safety. Valid valence does not establish
synthesizability. Selectivity, solubility, ADME and experimental efficacy remain
unestablished. There is no universal composite “optimisation score.”

## Optional potency scoring

Click **Fit scoped 2D model** explicitly to use the existing regression engine
with ECFP4 and random forest, extra trees or ridge. It uses the entire selected
assay context (30–500 unique isomeric structures), not pooled target-wide
activities or the 3D field model. Duplicate structures must be merged first.
The user chooses random or scaffold validation, test fraction and CV folds.
A single-scaffold series cannot support a meaningful scaffold holdout; choose
an appropriate broader assay or use a clearly labelled random split instead.

Current default scoring gates require all of:

- At least 10 held-out test compounds.
- Held-out R² ≥0.5.
- Pooled training cross-validation q² ≥0.3.
- Test RMSE lower than the training-mean baseline's test RMSE.
- A finite, positive training-CV absolute-residual band.
- A non-baseline selected estimator.

These are conservative **engineering policy**, not a universal validation
standard, a prospective result, or a scientific qualification certificate.
Scores, settings and gates are exported. All-data deployment refitting follows
the unchanged evaluation. Repeated choices based on the same test performance
invalidate that test's independence; no automated search against the test set
is performed. External and prospective validation remain necessary.

Only a passing model enables **Score generated analogues**. The parent and
each candidate must also retain their exact proposed molecular graph through
the existing prediction standardisation, have resolved stereochemistry and
pass the fingerprint-similarity applicability rule. Otherwise potency is
withheld. Weak models do not disable measured-evidence proposals.

Predicted ΔpActivity is the candidate prediction minus the **same model's
parent prediction**, not minus the parent's measured activity. Approximate
90% residual bands concern activity levels; they are neither gain intervals
nor calibrated probabilities that an analogue is better. Cliff-neighbour
flags do not repair a model's inability to predict a cliff.

## Downloads and reproducibility

Download the CSV directly or the ZIP evidence package containing candidates,
2D structure SDF, parent, supporting pairs, retained measurements, scoped
compounds, attempt accounting, settings, provenance and a SHA256 manifest.
Scored packages include model-check and evaluation tables. The SDF does not
contain validated 3D poses; no deployment-model pickle is included.

Scope signatures bind compound IDs, isomeric structures, activity values,
assay/endpoint/variant and retained record identifiers. Mismatched models are
refused, and failed scoring clears any previous numeric predictions.

Automated tests check reconstruction, one-position locking, assigned stereo,
scope separation, limits, weak-model/domain/state withholding, actual 2D model
fitting, archive contents and Streamlit opt-in/stale-result behaviour. These
tests establish software behaviour, **not** prospective medicinal-chemistry
success. No lead-optimisation benchmark or experimental synthesis has been
performed for this feature.

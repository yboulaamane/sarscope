# Design next analogues

For the **3D-field-model design workflow**, see
[3D design and prediction](field-qsar.md#design-and-predict-with-the-fitted-3d-model).
The separate 2D workflow described below is explicitly labelled **2D** in the
app; it is not the scoring engine used after a 3D-QSAR fit.
For 3D design, unassessed/incomplete/sensitive pose evidence follows every
candidate as an exploratory interpretation warning. Score checks alone cannot
remove the explicit consent requirement. New model-stability evidence hides
stale candidate predictions and requires rescoring before export. Training-only
pose-stress ranges are not candidate confidence intervals. The simpler 2D
baseline remains visible above the 3D design step.

SARscope can propose small, traceable experiments around a selected parent.
This is **experimental lead-optimisation decision support**, not a validated
autonomous lead optimiser. Generating a plausible graph or obtaining a higher
predicted potency does not establish an improved compound.

## Start in Streamlit

For ChEMBL data, curate the target, prepare series in **3 · SAR map**, then open
**11 · 2D analogue design**. Choose one assay/endpoint/structured-variant context,
a prepared scaffold series, a parent and editable R-positions. Activities are
reaggregated within that context; target-wide medians are not substituted.
Unreported mutation annotation is not evidence of wild type.

For your own compounds, choose **2D analogue design · own compounds** in the
sidebar. Upload CSV/TSV (`molecule_id,smiles,pactivity`) or SDF with measured
activities, select endpoint/units and confirm comparability. Prepare a shared
scaffold, then choose the parent and positions. Uploaded assay comparability
is explicitly **user-declared**, not verified against ChEMBL. Qualitative
screening calls are not converted into potency values. No receptor, docking,
3D alignment or field-QSAR model is needed.

Click **Generate analogue proposals**. Nothing is generated or trained on
page load. Changing the parent, positions, assay, structures, activities or
limits hides incompatible previous results until you regenerate.

### Measured example location

The app's main measured example is in **3D QSAR · own compounds**, with 75
CCR5 compounds, CSV and aligned SDF. See its
[source, preparation audit and walkthrough](../src/sarscope/data/field_ccr5_example/README.md).
Its native default 3D test R² is 0.642, with q² 0.499 (still below the check).
The earlier 50-compound EGFR data remain selectable as a weak-model example.
For historical CCR5 data, confirm original experimental comparability before
transferring matched-pair effects as design evidence; this is not verified
single-assay ChEMBL evidence.
The previous 106-compound 2D data files remain as an attributed historical
example, but are no longer offered as the UI's demo. Their RF scores must not
be used to claim performance of the 3D workflow.

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

# Supplementary Information: SARscope positional SAR study

Related manuscript: SARscope: an interactive workbench for positional structure–activity relationships and evidence stress testing.

## S1. Source datasets and curation

The frozen ChEMBL 37 parent manifest has SHA-256
f48c8aec796c8986de3938febbc95b99149355660e1c31315f98852e3395a493.
Assay snapshots and their individual hashes are in the case manifest. Complete
assay cohorts were extracted before record filtering; successful transformations
were not selected for inclusion in the snapshots.

| Series | Target ID | Assay ID | Document ID | Raw records | Retained records | Curated compounds |
|---|---|---|---|---:|---:|---:|
| BRAF | CHEMBL5145 | CHEMBL5046398 | CHEMBL5046217 | 46 | 44 | 44 |
| DRD2 | CHEMBL217 | CHEMBL671074 | CHEMBL1133592 | 34 | 31 | 30 |
| ER-alpha | CHEMBL206 | CHEMBL829540 | CHEMBL1138773 | 107 | 100 | 98 |

Settings: literature source ID 1; endpoint IC50; exact relation “=”;
concentration unit nM; ChEMBL assay type B; no structured variant annotation;
exclude potential duplicates and records carrying validity comments. Missing
variant annotation is not interpreted as verified wild type. The ER-alpha assay
is a counter-assay in a study of ER-beta ligands and was not relabeled ER-beta.

Structure standardization used Sorbent with canonical tautomerization enabled.
Records were combined by standardized structure using median pIC50. This is
a concentration-unit conversion and structure-level aggregation, not a biological
conversion between IC50, Ki, Kd and EC50. Individual retained records remain
available with source activity IDs and original concentration values.

## S2. Stereochemical core audit

The largest Murcko series was selected by compound count, with lexical scaffold
tie-breaking. The Murcko graph did not retain all information needed to separate
core stereochemical states in the DRD2 series. Per-compound R-group decomposition
cores were therefore recovered with RDKit and checked against the original
R-position assignments. Every recovered substituent assignment matched the
stored mapping before stereochemical restriction was applied.

The complete mapped, isomeric core string was used to define a core state.
This avoids comparing R/S labels in isolation: a change in substituent priorities
can alter a CIP label without altering the corresponding geometry. The largest
core state was selected by count and lexical tie-breaking, without using potency.

The DRD2 group contained 23 compounds with one assigned core state, two with the
opposite state and one with unassigned core stereochemistry. Excluded compounds
were CHEMBL125455, CHEMBL126493 and CHEMBL2109988. All 23 selected complete
structures carry the (S,S) cyclopropyl assignment according to their encoded
structures. This refers to reported structural annotations, not independent
experimental verification of stereochemical purity.

| Series | Original scaffold group | Core states | Selected compounds | Selected measurements | Positional pairs |
|---|---:|---:|---:|---:|---:|
| BRAF | 16 | 1 | 16 | 16 | 29 |
| DRD2 | 26 | 3 | 23 | 24 | 36 |
| ER-alpha | 53 | 1 | 53 | 55 | 198 |

The audit files list the core state and retention decision for every compound.
This additional restriction was applied by the original case-study analysis
script. The current workbench now retains the mapped core for each compound,
separates core states in the browser and defaults to the largest state in CLI maps.
The current audit reproduces the saved restricted case pairs and records exclusions. Earlier manuscript tables containing 26 DRD2 series compounds and
46 positional pairs are superseded for the scientific analysis by 23 and 36.
The earlier exports are preserved as the input record, not silently overwritten.

## S3. Pair eligibility and effect summaries

The comparison requires the same selected isomeric core and the same fragment
at every other mapped R-position. Potencies are recalculated from the selected
assay records. Their medians must match the member table, and every signed pair
delta is checked against the underlying compound values.

Transformation direction follows the lexical fragment ordering used by the
software and is not chosen to maximize an observed gain. Summaries retain assay,
endpoint and structured variant context. In these selected cases there is one
assay and endpoint per series and no structured variant annotation. One molecular
pair contributes one delta, even if the compound has multiple retained measurements.

The median and interquartile range describe observed pair effects. A conservative
deterministic matching counts pairs that share no molecule IDs. Disjoint pairs
are not experimental replicates, and overlapping chemical contexts can remain.

Descriptive support criteria reproduce the existing map implementation:

- Mixed: absolute median below the effect threshold, fewer than 80% of pair
  effects agreeing in sign, or effects beyond the threshold in opposite directions.
- Limited: a directional effect not meeting the disjoint-pair criteria below.
- Moderate: a directional, assay-matched effect with at least two disjoint pairs.
- Strong: a directional, assay-matched effect with at least three disjoint pairs
  and two remaining-substituent contexts.

The default threshold was 0.5 pIC50 units. It was fixed for the reference analysis,
not estimated from assay noise or optimized against the outputs. The terms
moderate and strong are interface labels, not significance levels or probabilities.
No case reached the map's 20,000-row cap.

## S4. Threshold and individual-compound sensitivity

The transformation family was kept fixed while effect thresholds were changed.
The support criteria yielded the following numbers of moderate/strong labels:

| Series | 0.25 units | 0.50 units | 0.75 units | 1.00 unit |
|---|---:|---:|---:|---:|
| BRAF | 7 | 5 | 4 | 4 |
| DRD2 | 6 | 5 | 4 | 2 |
| ER-alpha | 12 | 7 | 3 | 1 |
| Total | 25 | 17 | 11 | 7 |

For each transformation, every constituent compound was omitted in turn. All
pairs involving that compound were removed together. The median was recalculated
only when at least one pair remained. The proportion of valid omissions retaining
the original median's sign and the minimum/maximum resulting medians were recorded.
For isolated one-pair transformations no valid omission remains; these fields
are missing, not set to perfect stability.

All 17 reference moderate/strong transformations retained their median sign for
every valid omission. This assessment measures influence within the observed
network. It does not add independent evidence, estimate assay precision or
demonstrate performance on future compounds.

## S5. Label-permutation diagnostic

Five thousand potency-label permutations were generated for each selected
series with NumPy random-generator seeds 20261013, 20261014 and 20261015 for
BRAF, DRD2 and ER-alpha, respectively. Complete compound values were permuted,
not individual pair deltas. Consequently the same shuffled value is used whenever
a compound appears in more than one pair. Structures, pair membership and
support counts remain fixed.

For each transformation, the two-sided permutation P value was calculated as
(1 + number of shuffled absolute medians at least as large as the observed
absolute median)/(5,001). A numerical tolerance of 1e-12 was used for comparison
of floating-point values. Benjamini–Hochberg adjustment was applied to all 191
values together, not only the transformations selected for figures or support
labels. Shared compounds and chemical backgrounds create dependent comparisons;
the resulting adjusted values are reported as exploratory diagnostics without
claiming guaranteed false-discovery control.

| Series | Observed qualifying count | Shuffled mean count | Shuffled central 95% range | Probability of count at least observed |
|---|---:|---:|---|---:|
| BRAF | 5 | 4.362 | 1–8 | 0.453 |
| DRD2 | 5 | 2.887 | 0–6 | 0.136 |
| ER-alpha | 7 | 5.898 | 2–11 | 0.389 |

No transformation had an adjusted P value below 0.05. This result
limits claims based on support labels; it does not establish that the recorded
pair differences are erroneous or that all transformations have no biological
effect. The null removes the relationship between complete chemical structure
and potency and assumes exchangeability of the measured values within a series.
It is not a model of experimental noise or a substitute for new measurements.

## S6. Four-compound cycles and mapped-pair cliff sensitivity

Complete cycles were constructed on two mapped positions while holding all
other positions and the selected isomeric core constant. Every cell had to
contain exactly one standardized compound. The enumeration yielded seven BRAF,
six DRD2 and 25 ER-alpha cycles; these overlap and are not independent observations.
This is not an exhaustive graph-based MMP nonadditivity search.

The reported DRD2 chlorine cycle contains CHEMBL341497 (H/H), CHEMBL419610
(para-Cl/ortho-H), CHEMBL126633 (para-H/ortho-Cl) and CHEMBL302075 (Cl/Cl).
Its contrast is pIC50(Cl/Cl) − pIC50(Cl/H) − pIC50(H/Cl) + pIC50(H/H), equal to
−0.3689947652 units. CHEMBL302075 has retained IC50 values of 65 and 93 nM;
median aggregation on the logarithmic scale gives a back-converted equivalent
concentration of 77.7496 nM. The published records do not identify independent
replicate experiments for these entries. The contrast was not assigned a
mechanistic or thermodynamic interpretation.

Morgan fingerprints used radius 2, 2,048 bits and chirality disabled. Similarity
was calculated with Tanimoto, and the potency gap was the absolute pair ΔpIC50.
At similarity above 0.90 and gap above 2.0, one of the 263 mapped pairs was
called a cliff: CHEMBL5092007/CHEMBL5076831 in BRAF. The reported values are
97 and 9,710 nM; the gap is 2.00045 and similarity 0.918033. No call remains
at similarity above 0.95 or gap above 2.5 units. The full cutoff grid and pair
values are supplied. These counts concern mapped positional pairs only.

## S7. mmpdb coverage comparison

Both methods received the same curated compounds from the complete assay cohorts,
before the selected-core restriction used for the positional study. Counts
therefore use 44 BRAF, 30 DRD2 and 98 ER-alpha compounds. The comparison used
mmpdb 3.1.4, one cut, the RDKit default cut pattern, no extra salt removal,
variable fragments of at most ten heavy atoms and at most half of the complete
molecule. Commands, logs, fragment databases and pair membership are archived.

| Series | SARscope unique pairs | mmpdb unique pairs | Shared | mmpdb-only |
|---|---:|---:|---:|---:|
| BRAF | 114 | 116 | 114 | 2 |
| DRD2 | 205 | 205 | 205 | 0 |
| ER-alpha | 441 | 495 | 441 | 54 |

All 56 mmpdb-only pairs contain an explicit hydrogen variable fragment in the
mmpdb representation. This identifies a limitation of the current single-cut
enumerator, not of the R-group map's representation of H alternatives. Timing
values are retained in the original log but are not used as a speed comparison.
No counterbalanced DataWarrior/RDKit/SARscope human-workflow study has been run.

## S8. Supporting retrospective regression benchmark

The previously run frozen panel retained exact nM binding IC50 values lacking
a structured variant annotation, excluded duplicate/validity flags and aggregated
by standardized structure. It was not restricted to the three case-study assays.
The label in the historical protocol should not be taken as confirmation of
wild-type protein identity when annotations were absent.

Models used ECFP4 with fold-local feature filtering, a 200-tree random forest with
maximum depth 15, three-fold scaffold-group training cross-validation, an outer
20% scaffold holdout and seed 42. Acyclic molecules were separate groups. The
baseline predicted the training-set median. A thousand resamples of the saved
held-out scaffold groups, seed 42, provided conditional RMSE intervals without
retraining the models.

| Target | Test compounds | R² | RMSE | Conditional 95% RMSE interval | Spearman | Baseline RMSE |
|---|---:|---:|---:|---|---:|---:|
| BRAF | 584 | 0.762 | 0.646 | 0.578–0.720 | 0.859 | 1.342 |
| DRD2 | 99 | 0.310 | 0.902 | 0.756–1.041 | 0.574 | 1.087 |
| ER-alpha | 651 | 0.696 | 0.828 | 0.744–0.925 | 0.836 | 1.503 |

The intervals condition on one fitted model and split per target. Assay and
document overlap may remain, and the selected panel is not external validation,
cross-target transfer or a prospective trial. The case-study cohorts overlap
this source pool and must not be described as independent QSAR test datasets.

Figure S1: `benchmark-audit/benchmark.svg`. Previously saved scaffold-held-out
regression errors and a training-median baseline. Intervals are conditional
test-scaffold resampling intervals, not repeated-training uncertainty.

## S9. Software, execution and availability

The analysis layer is implemented in Python; the browser interface uses Streamlit.
Structure standardization uses Sorbent/RDKit, tabular operations pandas, numerical
operations NumPy/SciPy and optional modeling scikit-learn. The analysis does not
require the browser, a remote rendering service or docking. An SVG component
supports position selection, while CSV exports retain measurements and source IDs.

The scientific robustness analysis is reproducible with the supplied analysis
script, retained source tables and a new output directory. It performs no API
queries, docking or model training. Seeds, settings, individual input SHA-256
hashes and package versions are recorded in its provenance file. The original
case evaluation was also reproduced in a fresh isolated Linux/Python 3.13
environment; all 26 CSV outputs matched after timing fields were excluded.
The supplied wheel-hashed runtime lock is platform-specific.

Observed analysis versions: RDKit 2026.3.6, NumPy 2.5.3, pandas 3.0.6,
SciPy 1.18.1, scikit-learn 1.9.1 and Sorbent 0.1.0. The source revision base
was 7a8e4c805d671d92bccf38673bd00bc2207ad7cd with manuscript/evaluation changes
not yet committed at preparation. File hashes, rather than this base revision
alone, define the accompanying working package.

Independent source-paper SAR annotation, expert review, human-workflow
comparisons and external release-archive reproduction remain unperformed.
A publicly accessible versioned code/data deposit must be provided before
submission. Source code is MIT; ChEMBL-derived evidence is CC BY-SA 3.0.

## S10. Current workbench audit

The production workbench was re-evaluated against the frozen, hash-checked raw
assay snapshots. Records were curated and decomposed anew, without downloads,
docking or model training. The largest identical mapped isomeric core state was
used for each series. The scoped pair identities and effects agreed with both
the saved scientific-study pairs and a separate exhaustive enumeration within
1e-10 pIC50 units. Both calculations share RDKit decomposition, so this checks
implementation consistency rather than independent chemical truth.

| Series | Selected compounds | Excluded other-core compounds | Positional pairs | Default repeated-comparison transformations | Full directional criterion retained |
|---|---:|---:|---:|---:|---:|
| BRAF | 16 | 0 | 29 | 5 | 4 |
| DRD2 | 23 | 3 | 36 | 5 | 5 |
| ER-alpha | 53 | 0 | 198 | 7 | 6 |
| Total | 92 | 3 | 263 | 17 | 15 |

All 17 retained median sign for their valid compound omissions, but two did
not retain the full directional criterion at the default 0.5-unit threshold.
BRAF R1 F→H changed from −0.684312 to −0.255919 after omission of
CHEMBL5077573 or CHEMBL5083036. ER-alpha R3 bromovinyl→cyano changed from
+0.603087 to +0.152377 after omission of CHEMBL186513 or CHEMBL188900.
The equivalent chemical-context omissions exposed the same dependencies.
These results distinguish sign preservation from satisfying a practical
magnitude/consistency rule; neither is an estimate of statistical confidence.

Each case contains one assay and one publication. Every assay or publication
omission therefore leaves no eligible evidence for all 17 transformations.
Such omissions are counted as unevaluable, not successful persistence. The
three cases cannot demonstrate cross-assay or publication-independent
transferability. No omission or cycle-candidate bound was reached. Suggested
fourth analogues or retests were not experimentally evaluated and are not ranked
by predicted potency or synthetic feasibility.

The audit covers the 17 default repeated-comparison claims, not every possible
design proposal. Per-claim checks, contexts, thresholds, cycles and suggestions
are in `workbench-audit/`. The combined CSVs include case and claim IDs; cycles
and suggestions can recur across claims and their row counts are not counts
of independent experiments or unique compounds. The provenance file hashes
all inputs, relevant source modules and generated CSVs.

Reproduce with a new output directory:

`python paper/audit_current_workbench.py --out runs/workbench-reproduction`

The preceding full local software test run completed 414 tests, with three skips
and two network tests deselected. Unit and Streamlit tests cover mismatched
contexts, source-record copies, preservation of distinct rounded measurements,
ambiguous cycle cells, stereochemical core selection, incomplete searches,
stale-result invalidation, a frozen BRAF cycle display and export hashes. This
is not an independent reviewer assessment or a comparison of chemist task
performance. The new audit utility is additional evaluation code, not a new
public release.

## S11. Scope of validation and submission status

The supplied evidence supports local numerical reproduction, implementation
checks and descriptive analysis of selected historical series. It does not
establish recovered agreement with independently annotated original-paper SAR
claims, prospective prediction, successful suggested experiments, improved
interpretation by chemists or external release reproduction. The original
41-task review pack remains unreviewed; its historical DRD2 tasks must be
regenerated from the restricted core before expert use.

Before submission, the tested code revision and required data must be available
anonymously in a versioned public archive, with source-data licenses preserved.
The supplied affiliation is included on the title page; funding,
competing-interest and authorship statements still require author confirmation.
Journal-format preparation does not remove these
gates or guarantee editorial acceptance.

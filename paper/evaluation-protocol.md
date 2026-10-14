# Evaluation protocol — working version 1

This protocol was written **after** the existing QSAR results. It is not a
preregistration of those results. The small SAR examples are retrospective and
overlap the benchmark's source pool. They cannot serve as independent QSAR tests.

## Questions and units of analysis

1. Does every annotated position change trace to the exact molecules, assay,
   endpoint and measured values? Unit: a position-pair record.
2. Does the tool correctly calculate differences while holding other mapped
   positions fixed? Unit: an unordered molecular pair at one position.
3. Are directional statements and descriptive evidence grades defensible to
   medicinal chemists? Unit: a transformation within a fixed assay/core/endpoint.
4. Does the staged browser workflow reduce time and errors in interpreting a
   series compared with existing tools? Unit: a participant completing a task.

The implementation verifies questions 1–2. Question 3 needs human review;
question 4 needs an actual user/comparator study. Neither is answered by a test pass.

## Case selection and frozen inputs

Fixed target families: BRAF kinase, DRD2 GPCR, ESR1 nuclear receptor. The source
panel is ChEMBL 37. Three assays from literature source `src_id=1` were selected
for populated, chemically related series. Protocol IDs, papers, scope notes and
hashes are in `case-protocol.json` and `cases/manifest.json`.

Keep exact binding IC50 in nM, exclude potential duplicates and validity flags,
and retain records lacking a structured variant annotation. Missing annotation
is **not confirmation of wild type**. Retain raw descriptions for expert audit;
do not infer a mutation from them automatically. In particular the paper on
ER-beta ligands supplies an ER-alpha counter-assay here, not an ER-beta dataset.

Within each assay choose the largest populated Murcko scaffold after curation;
break ties lexically, without inspecting potency effects. Report all scaffold
counts and the fraction of curated molecules in that selected series. Fixed
effect threshold: 0.5 log10 molar units. No changing it to improve agreement.

## Literature-trend recovery (still pending)

Before reviewing SARscope outputs, two chemists read the full paper and relevant
supplementary tables independently. Record expected **specific transformations**,
assay identity, author compound labels mapped to ChEMBL IDs, endpoint, sign or
context dependence, exact table/figure/page, and caveats in `literature-expectations.csv`.
The empty template contains no guessed trends. No full-text access was available
for independent table-level annotation in this package.

Resolve disagreements with an adjudicator while still blinded to tool output.
Freeze/hash the finalized annotations before calculating recovery. Include
unsupported/ambiguous cases, missing comparators and transformations spanning
multiple sites. Distinguish correct direction, wrong direction, abstention,
not expressible by the current core and missing measurements. Report counts and
denominators for every case, not a best-example percentage. Evidence-grade
threshold tuning must use separate development data, not this evaluation set.

## Descriptive grade review (41 tasks prepared, zero expert decisions)

The deterministic sample takes up to five transformations per grade per case,
including mixed and limited. It is grade-stratified, **not representative
prevalence sampling**. Shared compounds make tasks dependent.

Distribute only task definitions, pair evidence, source measurements and
structures. Withhold `review-answer-key.csv`, `transformations.csv` and annotated
SVGs. Reviewers see the directed change and raw measurements, not the tool's
grade or narrative. Two reviewers independently classify each task as
`increase`, `decrease`, `mixed` or `insufficient`; optionally flag a source issue
and cite the original table/figure. `insufficient` means no defensible general
direction from these records; `mixed` includes contradictory or subthreshold
changes. Decide from all pair contexts, not the median alone.

Analysis: exact agreement on resolved tasks; unsupported directional claims
among moderate/strong tasks; results by case and grade. Record reviewer
disagreement and every unreviewed task separately. Unanimity of at least two
distinct reviewer IDs is required for the automated scorer's consensus. Export
adjudication as a separate signed/dated record, not edits to original reviews.
Keep effect size, support counts and expert confidence distinct. These grades
are **not calibrated probabilities**, even if review agreement is high.

## Comparator protocol

### Automated mmpdb and exhaustive positional reference

The implemented mmpdb comparison uses the same curated molecules and one assay
per input, one cut, the RDKit default single-cut SMARTS, no additional salt
removal, variable fragments up to 10 heavy atoms and at most half the molecule.
Full commands/versions and pair membership are retained. Count unordered unique
compound pairs and context rows separately. mmpdb adds implicit-H substitutions;
SARscope's current single-cut MMP enumeration does not. This difference is a
limitation, not a false positive or a speed/novelty advantage.

The direct positional reference uses the same RDKit decomposition and separately
enumerates every two-compound combination differing in exactly one mapped
position. It checks pair identity and assay-median deltas. Shared RDKit chemistry
means this is an implementation sanity check, not an independent algorithm test.

### DataWarrior vs SARscope vs RDKit notebook (not yet run)

Use identical input molecules, source measurements, assay subsets and thresholds.
Document DataWarrior version, descriptor choice and grouping-by-assay settings.
DataWarrior already supports R-group decomposition, activity cliffs and grouping
by experimental conditions; do not call these features new. Descriptor-dependent
cliff sets must be compared with their definitions recorded, not treated as errors.

Run six fixed tasks: curate a named assay; identify a scaffold series; inspect
a single-position substitution; detect conflicting/sparse evidence; find multiple
activity-cliff pairs; export a traceable conclusion and supporting measurements.
Use supplied task instructions, identical hardware and cached input. Separate
download/startup time from analysis and task-completion time. Prespecify a
20-minute cap per task and record failures/timeouts, not just successful runs.

For a pilot, recruit at least six users with mixed computational/medicinal
chemistry experience; treat this as a descriptive usability pilot, not a powered
superiority trial. Counterbalance tools/case order, record prior familiarity and
training time, and blind accuracy scoring to tool. The scripted RDKit notebook
baseline must be archived with versions, code and manual steps.

Save timings, errors, traceability (can the conclusion be traced to exact assay
and activity IDs?), manual operations and interpretation accuracy in
`comparator-results-template.csv`. Report per-tool distributions and paired
within-participant differences. The current mmpdb subprocess vs in-process
function timings are **not** substitutes for this study.

## QSAR supporting result

The existing benchmark is one within-target scaffold split across three families,
not cross-target transfer, independent external validation or prospective proof.
The bootstrap resamples held-out scaffold clusters, using separate IDs for acyclic
compounds, with 1,000 replicates/seed 42. Report per-target point estimates and
95% percentile intervals for RMSE and paired baseline improvement. The intervals
condition on the existing fitted model and split; they do not include training,
target-selection or source/assay uncertainty. Publish full frozen inputs and
predictions before making a reproducibility claim for the QSAR result.

## Decision gate

No-Go for submission until independently annotated literature trends, the human
comparison/review, anonymous public versioned code/data, complete author
declarations and independent release-archive reproduction are available. Local
clean-environment case reproduction is complete; external reproduction is not. New QSAR
performance claims additionally require independent source/time holdouts.
Do not add docking mechanistic claims without robust evidence; docking is optional
and deliberately excluded from the main paper's contribution.

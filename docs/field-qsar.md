# Experimental field-based 3D QSAR

## Uncertainty-aware 3D SAR, not automatic lead optimisation

Use a closely related series with comparable measured endpoints and a reviewed
alignment. A pre-aligned SDF is preferred; CSV/SMILES and generated alignment
remain available with an explicit pose-assumption warning. A shared scaffold
does not establish a common binding mode. Nothing fits or stress-tests itself
when the page opens.

After the user-triggered fit, the same-split **3D PLS, 2D ECFP4 RF and training-mean
RMSE** remain visible. They are point estimates, not a significance test or
permission to choose a preset from held-out performance.

The existing training-only stability check now appears **before contours**,
with an explicit button. It retains the same ten declared charge/grid/pose/control
variants; it is not a new modelling engine. Three deterministic relative-pose
stress tests use 0.25 Å translations and 5° rotations, refitting training-only
nested CV. The per-compound table and range plot describe fitting-set OOF
variation, **not candidate confidence intervals**. Held-out rows are excluded.

Pose evidence is labelled separately from predictive score checks:

- `not_assessed`: no pose test was run; missing evidence is not a pass.
- `incomplete`: missing/failed computation, missing controls, legacy/unbound
  results, or a latest failed attempt; rerun for this fitted model.
- `sensitive`: completed, model-bound pose checks exceed a declared prediction-change
  limit. Contours and predicted gains remain unstable hypotheses.
- `within_tested_limits`: the three declared pose checks and numerical controls
  completed within limits for this model. This does **not** validate binding
  poses, spatial contour agreement, generated conformers or prospective predictions.

The fixed engineering limits remain p95 prediction change ≤ 0.5 and maximum
≤ 1.0 log units, with common-translation error ≤ 1e−8. They are not universal
QSAR-validity standards. No alternative is selected automatically. A separate
95% single-field contribution warning prompts review of scaling; coefficient ×
training-SD contributions are not causal fractions of biological activity.

Contours always carry exploratory interpretation. Candidate scoring requires
explicit exploratory consent whenever model checks fail **or pose evidence is
unqualified**, even if the displayed score checks are green. Stereo, grid,
alignment, missing-field and domain failures cannot be bypassed by that consent.
Each candidate CSV carries `model_pose_status` and `interpretation_warning`;
the display includes the pose-evidence status. Training-only stress ranges are
never substituted for candidate uncertainty bands.

The field evidence package now includes `interpretation.json`; candidate packages
record the same interpretation and available model-sensitivity summary/provenance.
Diagnostics are bound to the fitted model identity. New stability evidence
invalidates old candidate-result displays/downloads without changing coefficients
or refitting PLS. A failed recheck retains earlier raw evidence but removes its
successful status. These UI/report changes do not repair the earlier predictive
validation failures or the separately documented intermittent MCS timeout.

## Design and predict with the fitted 3D model

In **3D QSAR · own compounds**, the main measured example is now **75 CCR5
antagonists**, with a [CSV](../src/sarscope/data/field_ccr5_example/ccr5_3d.csv),
[aligned SDF](../src/sarscope/data/field_ccr5_example/ccr5_3d_aligned.sdf),
preparation audit, unchanged source activities, licence and
[walkthrough](../src/sarscope/data/field_ccr5_example/README.md). The native
default 3D run gives held-out R² **0.642**, RMSE **0.521**, pooled nested-CV
q² **0.499**, with a 56/19 random split. The q² > 0.5 check remains failed;
there is no rounding-based pass or package-certification claim. Four fixed
repeat splits have test R² 0.395–0.649, and every outcome is retained. The
primary seed remains 42, not a selected best seed. The same-split RF baseline
performs better and is not relabelled as a 3D result.

**Stricter validation attempt:** the [full retrospective report](ccr5-demo-validation.md)
finds historical test R² **−0.683**, scaffold test R² **−3.184**, wide uncertainty
and two failed relative-pose stability checks. Five further fixed random
splits give R² 0.162–0.689. The positive default tutorial therefore does not
qualify the workflow for predictive lead optimisation. No preset is selected
from these diagnostic results.

This is a tutorial on an already inspected public cohort, not a new external
benchmark or a reproduction of its original 63/12 partition. Source charged
amines are explicitly prepared using the app's existing neutral-parent
standardisation, with heavy-atom elements/bonds/coordinates verified unchanged.
The source protonation and supplied alignment are not established binding
states. Use the SDF to reproduce the recorded result; generating fresh poses
from the CSV changes the input geometry. The old **50-compound EGFR** example
remains selectable as a weak-model limitation case, with its original negative
scores intact. Neither case uses the previous 2D demo's RF scores.

**Restore example model defaults** is an explicit reset, not an automatic fit.
It restores the native field/split settings and dataset reference, clears old
results, and requires preparation and alignment review again. Historical CCR5
matched-pair design requires a separate experimental-comparability declaration;
the dataset does not supply independently adjudicated ChEMBL assay records.

After preparing/reviewing alignment and fitting the field model, **4 · Design
experiments · exploratory 3D predictions** appears below the results. Prepare
R-positions, choose a parent/editable positions and generate up to ten
one-position substitutions supported by measured positional pairs. The
fixed mapped core is reconstructed using existing SARscope chemistry; no
unobserved measured potency is assigned. This post-evaluation exploration may
inspect the full measured series, including test rows; it is not another
independent validation or an optimisation-success benchmark.

Predictions use the **same displayed training-fitted PLS estimator**, not a
new all-data refit or the RF baseline. The original training grid, field
definition, charge settings, column ordering, retained mask, mean-imputation
contract, scaling and reference coordinates are reused. Known structures reuse
their exact stored pose/fields. Novel candidates use bounded stereo-checked
generation to that fixed reference; a pre-aligned model requires an explicit
choice of candidate alignment route. Model/geometry signatures hide or refuse
stale proposals and predictions after a change.

No numerical potency is shown by default for an unvalidated model. Explicit
exploratory consent permits inspection without passing/changing any validation
checks. Candidate and parent stereo/alignment/grid/missing-field/domain failures
still withhold values. Applicability is the original fitting-field two-component
PCA box, a weak heuristic rather than proof of reliable prediction. Batches
are capped at ten, with a total preparation budget of at most 120 seconds;
failed rows and reasons are retained. New conformations require the existing
preparation extra; no docking is run and no 2D fallback exists.

Predicted delta is candidate prediction minus the **same model's parent
prediction**. Each field's contribution is the sum of
`(candidate_filtered_field - parent_filtered_field) * fitted_coefficient` for
that field's retained columns. Their sum is checked against the model's delta.
These are model associations, not physical interaction energies or biological
causes. Approximate training-CV residual bands describe activity levels, not
gain intervals or probabilities that a compound is better.

The evidence ZIP includes candidate CSV, aligned scored poses, fitted reference,
per-field delta contributions, charges, unchanged model checks/settings, source
evidence and hashes. This is experimental decision support, not validated lead
optimisation. Native 3D validation remains incomplete.

For the current published-method reference direction, see
[SYBYL reference: implemented components and remaining parity gaps](sybyl-reference.md).
The default SARscope variant is retained; the separate public Tripos-5.2 SE
path has explicit cutoff/dielectric controls and opt-in fitting-mean overlap
handling. It is not a complete SYBYL-compatible implementation.

The [latest public-SE validation attempt](public-se-validation.md) passes
numerical checks but fails its additional held-out predictive criteria.
It also identifies entire-field removal by screening and alignment sensitivity.
The implementation remains experimental; a small permutation p-value or high
PCA-box coverage does not override poor predictions.

This is an **optional small-series analysis**, not an automatic target-wide
model. Its intended question is: within a coherent assay and chemical series,
do spatial molecular-property fields add predictive information beyond a 2D
fingerprint baseline?

**Status:** experimental, with independent three-field training/LOO benchmark
agreement. SARscope now computes the SEH steroid reproduction itself, including
preparation checks, grid, fields, preprocessing, component search, fitting and
prediction—no external QSAR API. q² = 0.607851 versus published 0.609 and
training R² = 0.917438 versus 0.917, with three components selected from 1–12.
All 21 training and 10 test compounds are retained. See the
[independent evidence](../benchmarks/field_qsar/INDEPENDENT.md).

The [18 October validation update](3d-qsar-validation-20261018.md) adds matched
thrombin LOO/reference execution, primary-source graph adjudication and native
THR/AChE prediction evidence. It includes a paper draft, generated figures and
177 passing targeted tests. This does not override failed qualification checks
or establish exact five-field/CoMFA equivalence.

An [opt-in primary-source thrombin worked example](primary-thrombin-worked-example.md)
is now available in the measured-example selector. It fetches the separately
licensed original data only on request, preserves reviewed input states and
supplied 59/29 membership, and reproduces the recorded metrics through the actual
Streamlit modelling path. Applying its profile is explicit; review and Run
remain separate. Fine-grid work opts into a bounded 20,000-point/512 MB workspace;
ordinary limits stay unchanged. Failed qualification checks remain failed.

This validates neither the native end-to-end workflow nor its predictive
utility: historical reproduction uses combined-set field scales and separate
subset centres. The exported reference test R² additionally matches a known
test-label mean offset; our genuine prediction column does not use it. The
oracle-shifted statistic is explicitly labelled a non-predictive diagnostic.
Production training-only preprocessing remains unchanged.
Five-field/H-bond equivalence, CoMFA reproduction, general external validation
and upstream SDF redistribution permission remain unestablished. Native
settings can predict poorly; some scaling variants are pathological. Do not
describe the feature as a validated general tool, a Tripos replica, or a first
implementation. Field-based methods and open implementations already exist.

## Browser workflow

There are two entry points:

1. Curate a ChEMBL target, then open **10 · 3D QSAR**. Choose one assay,
   endpoint and structured variant context. Potency is recomputed from the
   exact retained measurements in that context, not target-wide medians.
   Unreported variant annotation does not mean experimentally confirmed WT.
2. Choose **3D QSAR · own compounds** in the sidebar. Upload SDF, CSV or TSV,
   identify the activity/identifier columns, specify units and endpoint, and
   confirm that the measurements are comparable. Active/inactive screening
   calls are never converted to potency.

Select a scaffold and explicitly choose the compounds. There is no silent
random sampling or exclusion. The browser supports **20–100 unique compounds**;
20 is a computational minimum, not a scientifically sufficient sample size.
Uploads are limited to 5 MB / 500 measurement records. Existing SARscope
curation aggregates upload replicates and logs the accounting. Standardisation
that would change a coordinate-bearing molecular graph blocks the upload
rather than transferring coordinates to a different molecule.

### Review geometry before fitting

- **Pre-aligned SDF:** preserve coordinates. Uploaded alignment is a user
  assertion, not automatically an experimental pose. No energy minimisation,
  rigid superposition or docking is applied to this route.
- **Rigid MCS:** stereo-checked conformers are superposed on a fixed reference
  with symmetry-aware, aromaticity/hybridisation-compatible heavy-atom maps.
- **Core-constrained:** reference-core coordinates are fixed while sampling
  and minimising substituents. This is a conformational hypothesis, not a
  receptor-derived binding mode.

Generated alignment requires the existing `sarscope[docking]` preparation
extra. It **does not invoke Vina**, require a receptor or imply redocking
validation. Calculation from an aligned SDF adds no required Python package.
Every molecule must have resolved absolute stereochemistry and a valid 3D
geometry. A failure blocks the series and identifies the molecule; the user
can review/exclude it explicitly. The MCS search is bounded, and an excessive
symmetry count refuses alignment instead of silently taking one arbitrary map.
The budget counts **distinct atom maps**, not duplicate query-automorphism
pairs: for example, 12 × 12 benzene embeddings describe only 12 distinct maps.
Match enumeration and distinct-map limits still refuse truncated searches.
Generated jobs use **one serial, killable worker per preparation run**, rather
than reloading Python/RDKit in a new process for every compound. There is no
parallel burst of workers. Each conformer/MCS job has a 30-second default
limit; worker startup also has a bounded wait and counts towards the existing
180-second whole-series budget. A rigid alignment reuses the fixed reference
coordinates rather than embedding that reference again in the compound loop.
Timeout errors distinguish startup, a compound job and whole-series exhaustion,
and compound failures identify the position and completed count. A timeout
does not establish that the current structure is invalid. No molecule or
stereo constraint is skipped, and partial preparations are not accepted.
Whole-series MCS coverage is QC only;
alignment uses pairwise maps to the fixed reference, not a test-derived core.

Review the overlay, core RMSD and coverage, then confirm the alignment. The
browser viewer loads 3Dmol.js from its public host. Coordinates and grids are
embedded in the viewer, not fetched by a compound identifier. SDF and OpenDX
downloads remain usable when the browser cannot load that viewer.

## Precisely which fields?

The UI deliberately labels these **CoMFA-style** and **CoMSIA-style**:

- LJ 12–6 uses RDKit van der Waals radii and UFF atomic well depths, with a
  carbon probe radius 1.70 Å and well depth 0.1070 kcal/mol. Mixing uses the
  radii sum and geometric mean of the well depths. Coulomb uses 332.0636,
  a +1 e probe and constant dielectric. Distances are bounded at 0.3 Å;
  steric energies are capped at +30 kcal/mol. Electrostatics at that cap
  remain undefined (`NaN`), never imputed to zero or the mean.
- Gaussian fields use the positive sum `Σ weight × exp(-α r²)`, α = 0.3 Å⁻²
  by default. Steric weights are RDKit radius³; electrostatics use partial
  charges; hydrophobicity uses atomic Crippen logP; donor/acceptor weights use
  atom-centred RDKit feature SMARTS. These are **not** proprietary Tripos
  hydrophobic tables or directional H-bond extension points. The positive
  convention is explicit because historical implementations differ in sign.

Explicit hydrogens participate in charge/field calculation. Gasteiger and
MMFF94 are available. A failed charge calculation blocks the browser run.
Pre-aligned input can explicitly retain supplied hydrogen coordinates. The
browser also exposes SEH/other field subsets, training-only raw/block/column
scaling and full capped maximum-q² selection versus the early-stop rule.
None of those choices enables historical combined-set or test-label centring.
The API can explicitly allow per-molecule MMFF94→Gasteiger fallback; every
fallback and its reason are reported, never silently pooled. Local providers
can use `register_charge_model(name, callable)`; they must return one finite
charge per explicit atom, summing to the molecule's formal charge within
0.1 e. Provider versions should be added to provenance by the caller.

Grid defaults are 2 Å spacing and 4 Å margin. Bounds are determined from
outer-training coordinates only, not held-out molecules. Grid count and
workspace estimates are checked before field allocation. Distance computation
is chunked, and field screening does not allocate a p×p correlation matrix.

## Evaluation and its limits

The workflow reuses SARscope's regression engine, outer split and CV engine.
It adds training-only field screening and nested PLS component selection:

- Undefined columns are removed using fitting rows only. If a held-out CoMFA
  molecule makes a retained electrostatic column undefined, scoring is
  refused. Use Gaussian fields or review the alignment; no global
  held-out-derived mask or implicit imputation is used.
- CoMSIA uses a top-variable-fraction filter. CoMFA also offers a raw-unit
  minimum-SD filter. Screening is fitted anew in every model-selection fold.
  PLS mean-centres without autoscaling, so raw field magnitude is preserved.
- Component count is limited by the requested cap, fitting-row n/5 and
  feature rank. The n/5 cap is a conservative heuristic, not proof of validity.
  Components are selected with pooled PRESS/SST q², stopping when relative
  improvement falls below the recorded threshold (default 5%).
- The grid is fixed from the outer training geometry. Training CV therefore
  evaluates models **conditional on that grid and the chosen alignment**;
  it does not independently regenerate the entire geometric representation
  in each fold. The final held-out set never determines grid bounds, feature
  masks or component count.
- Random splitting measures within-series interpolation. Scaffold splitting
  may be impossible for a single-scaffold series. Time/source splits are not
  offered in this experimental workflow; the ordinary 2D workflow retains
  its existing validation options.
- An ECFP4 Random Forest and mean baseline use the same outer split as field
  PLS. The RF and field model use identical outer CV folds; compare retained
  column counts as well as q², held-out R², RMSE, MAE and Spearman correlation.
- Training-label permutations rerun the nested component-selection pipeline.
  The empirical p-value is `(1 + C)/(1 + B)`, where C counts permuted q² ≥
  observed q². No permutation result is discarded. Nineteen permutations
  give only a coarse screen; zero means the test was not run. The default
  permutation time budget is checked between repetitions; exceeding it
  refuses a partial result rather than reporting an incomplete p-value.
- Origin-regression slopes/R0² and conventional calibration heuristics are
  diagnostic, not sufficient conditions for a useful QSAR model. Predictive
  held-out R² is distinguished from squared Pearson correlation. Small,
  constant or wrongly predicted test sets must not receive an automatic pass.
- PCA-box membership uses the existing training-fitted applicability-domain
  implementation. It is coarse and does not guarantee prediction accuracy.

The report lists failed/not-run checks and always retains the experimental
benchmark warning. Repeatedly choosing alignment, charge, grid or method by
test performance contaminates the hold-out, even if each individual fit is
correctly ordered. Fresh external data are needed for a final evaluation.

See the [benchmark source audit](field-qsar-benchmark-status.md) for the actual
dataset investigation, hash and unresolved permissions/protocol requirements.

## Reading validation results and preparing a series

The scaffold selector defaults to **All compounds**, without silently reducing
the upload to its largest scaffold. The selection count and model-size gate
are explicit; **Select all available compounds** restores a manually edited
selection within the current filter (up to 100). Alignment/model controls need
20–100 selected compounds. A scaffold filter can leave too few compounds;
restore All compounds or add comparable measurements rather than interpreting
the absent controls as a running calculation.

Split feasibility is checked before field computation, including the minimum
15 training/3 held-out rows and the scaffold groups needed by nested CV. The
held-out scaffold partition uses approximately `1 / test_fraction` groups of
folds; its count is distinct from the requested CV fold count. A single
scaffold cannot support scaffold validation. Random splitting is offered as
within-series interpolation, not a substitute for scaffold extrapolation.

Strict CoMFA checks every fitting-derived retained mask against its validation
partition before any PLS/RF fitting. An overlap error names **Missing
electrostatic descriptors**, now visible above the advanced settings. Explicit
fitting-mean replacement remains opt-in for native SARscope fields; it is not
silently enabled to make a failed model run. Error messages replace the
previous run prompt and persist until the settings change or the model is retried.

`cv_pooled_q2` pools out-of-fold squared errors and divides by the full
training-set SST. `cv_fold_mean_r2` and `cv_fold_sd_r2` describe the unweighted
mean and sample SD of the fold-specific predictive R² scores, each with its
own fold mean/SST. Narrow-potency folds can have tiny denominators and very
negative R², even when pooled q² is only moderately negative. The legacy
`cv_r2`/`cv_r2_sd` columns remain aliases for compatibility. Definitions ship
in `metric_definitions.csv`. Pearson r² in the calibration check is separately
labelled; it is not predictive R². The grid check reports the number of
held-out compounds with **heavy atoms outside** the training grid, so zero is
a pass. Alignment checks display their coverage/RMSD criteria.

The charge-audit rows are **compounds**, with identifiers and zero-based
aligned-series row numbers, not PLS components. A blank fallback reason is
displayed as “Not needed (no fallback)”. Component selection has a separate
table: its requested maximum is a ceiling, not a promise to test every count.
The smallest fitting-fold n/5 cap, feature rank/count and early stopping can
shorten that search; the fitted model retains the actual cap/stop explanation.

After 3D analogue scoring, **Opposes measured pair(s)** flags predictions
whose direction contradicts any supporting measured pair for that directed
swap. Duplicate pair records count once. The observed median and opposing
pair count are exported with each scored candidate. This is a warning about
context-dependent evidence, not a claim that pair effects transfer perfectly:
neither scores nor model fits are changed to force agreement. Unscored rows
have no agreement verdict. The proposed-experiments result table opens only
after alignment/prediction has been requested for the current design.

Viewers use a larger canvas and a molecule-oriented camera, with Reset view.
Only the camera changes: aligned coordinates and contour grids retain the
same reference frame. Tiny/absent contour regions are not artificially
inflated or presented as stronger evidence. Startup displays a loading notice
before the scientific imports, and report plotting imports are deferred until
export; shared-host cold-start latency is still outside the model workflow.

## Contours and export

Contours are **raw-field coefficient × raw training standard deviation**,
mapped back through the fitted feature mask. Unretained points are zero,
meaning unmodelled—not evidence of no biological effect. Levels are the
chosen magnitude percentile separately within positive and negative values.
The default is the 80th percentile for both signs, not an 80/20 volume ratio.

Positive contours associate increasing that descriptor with higher pActivity;
negative contours associate it with lower pActivity. In particular, positive
LJ energy is not an unconditional recommendation to add bulk. Donor/acceptor
labels describe ligand features, not observed complementary protein atoms.

Percentages are shares of absolute coefficient×SD contributions, not fractions
of binding energy or independent explained variance. Correlated fields make
that distinction important.

The evidence ZIP includes a self-contained HTML report, measured/predicted
values and residuals, training OOF predictions, split assignments, checks,
permutations, charge audit, input accounting, component selection, aligned
SDF, reference SDF, full fields/retained masks/coefficients in NPZ, signed
OpenDX grids, a PyMOL script, package/parameter/source provenance and SHA-256
file hashes. It contains no executable pickle. The main report can include
the completed optional stage; the standalone upload workflow has its own ZIP.

## Remaining release gates

The agreed release scope is SARscope's **own documented methods**, not exact
proprietary CoMFA/CoMSIA compatibility. See the
[qualification record](3d-qsar-validation.md) for native three-family results,
numerical verification and remaining operational/scientific release gates.
Mixed predictive performance does not itself prove a numerical implementation
wrong, and a working implementation does not certify every fitted model.

- Source hashes, input accounting and a limited SEH benchmark reproduction
  are now retained. Upstream **data** redistribution permission and independent
  experimental structure/label adjudication remain separate requirements;
  no source SDF is bundled. Do not reconstruct values from memory.
- Native Gaussian evaluation on AChE, CCR5 and thrombin now runs through the
  existing app backend with fixed external splits and 99 permutations. The
  LJ/Coulomb cases abstain on held-out electrostatic overlap. Resolve that
  prediction contract without test-derived masking. Exact legacy CoMFA and
  directional hydrogen-bond equivalence are not the agreed release scope;
  retain explicit own-method conventions instead of claiming compatibility.
- Test sensitivity to alignment, charge and split on multiple real series.
- Separately implement/review receptor redocking, covalent-connectivity
  detection and assembly handling. None is implied by the current alignment
  workflow, and existing docking/receptor behaviour is unchanged.

No GPL 3D-QSAR source is needed or consulted for these implementations.

## Method references and prior art

- Haga CL, Le CN, Yang XD, Phinney DG (2025). Py-CoMSIA: An Open-Source
  Implementation of Comparative Molecular Similarity Indices Analysis in Python.
  Pharmaceuticals 18:440.
  [doi:10.3390/ph18030440](https://doi.org/10.3390/ph18030440).
- Cramer RD, Patterson DE, Bunce JD (1988). Comparative molecular field
  analysis (CoMFA). [doi:10.1021/ja00226a005](https://doi.org/10.1021/ja00226a005).
- Klebe G, Abraham U, Mietzner T (1994). Molecular similarity indices in a
  comparative analysis (CoMSIA).
  [doi:10.1021/jm00050a010](https://doi.org/10.1021/jm00050a010).
- Golbraikh A, Tropsha A (2002). Beware of q²!
  Journal of Molecular Graphics and Modelling 20:269–276.
  [doi:10.1016/S1093-3263(01)00123-1](https://doi.org/10.1016/S1093-3263(01)00123-1).
- [RDKit force-field parameter API](https://www.rdkit.org/docs/source/rdkit.Chem.rdForceFieldHelpers.html).
- [Scikit-learn PLS](https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html),
  [nested validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html),
  and [permutation testing](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html).
- Prior implementations include Open3DQSAR/Open3DALIGN, Cloud 3D-QSAR,
  3d-qsar.com and
  [Py-CoMSIA (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945924/).
  Their existence rules out claiming that field QSAR in Python is new.

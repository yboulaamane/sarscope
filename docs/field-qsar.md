# Experimental field-based 3D QSAR

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

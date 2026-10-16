# Public SE validation attempt — 16 October 2026

**Outcome: numerical checks pass; the additional predictive qualification fails.**
The documented steric/electrostatic implementation has stronger verification
and cross-environment evidence. Neither reserved-cohort model meets the
predeclared predictive criteria. Do not describe this as a fully validated
SYBYL-compatible package or deploy these particular models as predictive.

## Scope and protocol

The [protocol](../benchmarks/field_qsar/public-se-qualification-protocol.json)
fixes a primary positive-Gaussian SE workflow using public Tripos-5.2 R/K
parameters, explicit-H reviewed input, RDKit Gasteiger–Marsili charges,
1 Å sampling, training-only block scaling, nested capped PLS selection,
99 complete training-label permutations and 1000 conditional bootstrap
resamples. A 2 Å LJ/Coulomb model with explicit fitting-mean overlap handling
is secondary characterisation, **not a replacement primary**.

The two profiles have different filters, grids and retained-column counts.
They are not a matched-feature superiority experiment. Neither profile
implements the complete SYBYL atom typer, hydrophobic tables, directional
H-bond sites or Gasteiger–Hückel charges. Generated alignment/binding modes
are outside this validation scope.

The primary finer Gaussian grid was chosen before this campaign on numerical
sampling grounds and the preceding native-field sensitivity analysis, not
after searching ATA test outcomes. Previously inspected AChE/CCR5/thrombin
test scores are not counted as new independent evidence.

All sources are fixed to commit `0b30179ecc20aac5a003577b4437399a8fb20bca`
of the upstream dataset repository, with byte lengths, SHA-256 checks and
original membership retained. The source is attributed to
[Haga et al. (2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11945924/).
Only source SDF data, not another QSAR program, are used for prediction.
The upstream data licence is recorded as CC BY-SA 4.0; original SDFs are
fetched rather than bundled. The independent author-written scalar checks
verify the documented equations, not the vendor's unavailable numerical outputs.

## Numerical, leakage and audit controls

- Literal public R/K values and scalar LJ/Coulomb/Gaussian calculations.
- Charge zero/sign reversal; unchanged steric fields under charge changes.
- Both built-in charge providers, proper rigid-frame covariance and atom-order
  independence; chunking equivalence and input immutability.
- Training-only missing means and feature masks. Extreme changes to held-out
  activities **and coordinates** leave the grid, fitted mask/scales/coefficients
  and training OOF predictions unchanged.
- Freeze refusal on failed/missing/stale sampling evidence, code/configuration
  drift or modified evidence. An atomic evaluation-start receipt is created
  before fetching the test SDF, so this freeze cannot silently be evaluated
  again in a different output folder.
- Negative outcomes remain negative in assessment tests. The report never
  turns completed runs or small permutation p-values into certification.

The frozen [three-family component campaign](../benchmarks/field_qsar/tripos-components-20261016-v3/)
was repeated in a [fresh Python 3.12 stack](../benchmarks/field_qsar/tripos-components-20261016-py312/).
Across six cases, field-model predictions differ by at most
2.67 × 10⁻¹⁵; selected component and retained-column counts agree. RF baseline
RMSE differs by up to 0.00219 across scikit-learn versions. This is a
same-machine environment replication, not independent-laboratory validation.
The targeted numerical/qualification controls also ran on Python 3.12;
[JUnit records](../benchmarks/field_qsar/public-se-verification-20261016/) are retained.
Final local checks: **471 passed, 22 skipped, 2 network tests deselected** on
Python 3.13; **43 targeted backend checks passed** on Python 3.12. The latter
does not include Streamlit/browser checks. Ruff, formatting, mypy and
`git diff --check` pass. CI is configured but has not been run remotely here.

## Sampling and pose sensitivity

The [new training-only sensitivity record](../benchmarks/field_qsar/public-se-sensitivity-20261016/)
contains ten fixed diagnostics for each of three training cohorts. No held-out
SDF is fetched by that command. All **12/12 half-step lattice-phase flags**
pass; the largest OOF prediction change is 0.00235 activity units. Common-frame
translation changes predictions by at most 6.22 × 10⁻¹⁵.

By contrast, **5/9 relative-pose flags fail** under independent 0.25 Å/5°
rigid perturbations while fixing the first training reference. These perturbations
preserve intramolecular geometry and stereo, but are not alternative validated
binding modes. The thresholds are predeclared engineering flags, not universal
QSAR validity criteria. Passing numerical sampling does not resolve uncertainty
in molecular alignment.

The [freeze record](../benchmarks/field_qsar/public-se-qualification-20261016.freeze.json)
retains tracked workflow-source/configuration hashes and all five pose warnings. It is not
a certification. No candidate was changed after the additional test was scored.

## Additional held-out evaluation: actual outcome

The originally reserved **ATA cohort has now been consumed**: 72 training and
22 held-out compounds. It is a historical phenotypic anti-tuberculosis endpoint,
not another protein target family. Deposited log-activity labels are used
unchanged; no pIC50 conversion or single-assay homogeneity is asserted.
Published outcomes and source metadata were already known. This is an
additional held-out evaluation, **not a blinded prospective challenge**.

| Model | Nested q² | Held-out R² | RMSE | Retained S/E columns |
| --- | ---: | ---: | ---: | ---: |
| Gaussian SE, primary | 0.071 | −1.025 | 1.371 | 2000 / 0 |
| LJ/Coulomb SE, secondary | 0.126 | −1.066 | 1.385 | 217 / 170 |
| Same-split ECFP4 RF baseline | — | — | 0.831 | Not field descriptors |
| Training-mean baseline | — | — | 1.045 | None |

Both field models fail the predeclared q² > 0.5, test R² > 0.6 and
better-than-training-mean RMSE criteria. Both permutation p-values are 0.01;
this does not rescue weak internal or poor held-out predictions.

The 1000 scaffold-block bootstrap resamples use 13 held-out blocks. The
conditional R² interval is approximately −3.859 to −0.199 for the primary
and −5.111 to 0.156 for the secondary. These intervals condition on the fitted
model and public test population; they are not uncertainty over retraining,
alignment, assay noise, or future target/chemical-space shift.

PCA-box membership is 21/22 for the primary and 22/22 for the secondary,
despite poor prediction. Fourteen secondary-model test compounds exceed the
configured mean-replacement warning limit. An inside-domain flag is not
evidence of accuracy, particularly when descriptors are mean-filled.

The [evaluation evidence](../benchmarks/field_qsar/public-se-evaluation-20261016/)
contains individual predictions, permutations, checks, bootstrap summaries,
alignment/source accounting and manifests. The
[independently recalculated assessment](../benchmarks/field_qsar/public-se-assessment-20261016-v2/assessment.json)
verifies frozen files, recalculates R²/RMSE/permutation p-values and records
the explicit **NO-GO** outcome. Neither failed models nor source rows were
removed, relabelled, repaired or rescored to improve performance.

## Concrete problems found

1. **Entire fields can disappear during screening — corrected.** The global
   raw-variance screen removed all electrostatic columns before block scaling,
   in both the AChE training diagnostic and the final ATA primary fit. Those
   fitted models are therefore steric-only despite receiving SE inputs, and
   the absence of charge sensitivity in that situation is not proof of charge
   robustness.

   The cause is unit incommensurability, not a coding error: ranking all
   columns on one raw standard deviation compares Gaussian steric weights in
   cubed angstroms against electrostatic weights in elementary charge, so the
   larger-magnitude block takes the whole budget. The matched LJ/Coulomb fit,
   whose blocks are both in kcal/mol, retained 217 and 170 columns; the
   Gaussian fit retained 2000 and 0. That asymmetry is the signature.

   `FieldQsarParams.field_screen` now defaults to `per_field`, which ranks each
   block only against itself and gives it its own share of the budget. The
   previous behaviour remains available as `global_raw` so this frozen evidence
   stays reproducible. Two property tests pin the correction: the retained set
   must be invariant to rescaling any one field, and no requested field may end
   up empty. Neither uses ATA.

   **The frozen results above were produced with `global_raw` and are not
   re-scored here.** Changing `analysis/field_filter.py` alters the tracked
   workflow hashes, so the existing freezes will now report drift. That is the
   drift detection working, not a new failure. Whether the corrected screen
   predicts better is untested and cannot be settled on ATA, which is consumed.
   The Streamlit view and HTML exports warn when a field retains no columns and
   do not offer its empty contour.
2. **The source graphs need deeper adjudication.** A retrospective audit of
   exported graphs flags radical electrons in 59/88 thrombin structures;
   none are flagged in the other three cohorts. These are concrete encoded
   graph flags, not proof of experimental radicals or an automatic diagnosis
   of an incorrect structure. No bond orders or charges were "repaired" from
   a guess. The positive thrombin scores must not override this source concern.
3. **Alignment remains a substantial uncertainty.** The five pose-sensitivity
   failures are retained. There is no experimental proof that these supplied
   alignments describe their bound states.

## Release decision and next falsification test

**Verified:** specified numerical properties, fitting/test separation,
same-machine cross-environment reproducibility, and faithful retention of
failed predictive checks.

**Not established:** useful prediction on the additional cohort, full SYBYL
compatibility, generated-alignment validity, or broad validated release status.
This distinction follows the need for explicit endpoint, algorithm, domain
and predictive evidence in the
[OECD QSAR validation guidance](https://doi.org/10.1787/9789264085442-en);
it is not an OECD endorsement.

The field-balanced screen described above is now implemented and property
tested, which closes the numerical half of this step. What remains is the
empirical half: compare the two screens on the three existing training cohorts
using training-only statistics, and report the comparison without using it to
claim predictive benefit. Do not reuse ATA to choose the winner.
Resolve flagged source bond orders against primary records before treating
those molecules as a clean benchmark. A subsequent prediction claim needs
another frozen, adequately annotated endpoint/cohort not used for this redesign,
plus external scientific review. Negative results here must remain in the paper.

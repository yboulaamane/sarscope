# Positive, measured-data 3D-QSAR worked example

The primary-source thrombin series now has an executed end-to-end **MMFF94
worked example**: source/state verification, preserved alignment, field
calculation, nested PLS selection, prediction on all original held-out
compounds, same-split baselines, permutation testing, uncertainty/domain
diagnostics and portable prediction replay.

This is a **retrospective, fixed-alignment demonstration**, not a new blinded
validation study, guaranteed automatic alignment, exact legacy CoMFA
reproduction or package-wide certification. The separate generated-alignment
route and unsuccessful checks are reported, not hidden.

## Run it in Streamlit

1. Open **3D QSAR · own compounds**, then **Try a measured 3D-QSAR example**.
2. Choose **Thrombin · 88 primary-source compounds**.
3. Select **MMFF94 · positive fixed-partition worked example** in the
   worked-example profile selector.
4. Review the supporting data's CC BY-NC 4.0 terms, then fetch the example.
   Nothing downloads or fits merely by opening the page.
5. Select **Use original thrombin example** and click **Apply recorded thrombin
   profile**. This changes explicit settings only; it does not review or run.
6. Keep the complete 88-compound cohort and supplied 59/29 partition. Prepare
   the alignment, inspect the overlay and acknowledge its source assumptions.
7. Run the model separately. The full profile uses 99 permutations and the
   opt-in larger bounded workspace; local execution is recommended.
8. Inspect held-out predictions, the identical-split 2D baseline, domain and
   field contours, then download the evidence archive.

The recorded-score caption is labelled as previous evidence, not a result
calculated in the current session. The earlier Gaussian comparison remains
available and is not silently replaced. No score or validation check is
hard-coded to pass.

## Executed results

The original [Sutherland et al. benchmark](https://doi.org/10.1021/jm0497141)
and [Böhm et al. series](https://doi.org/10.1021/jm981062r) supply 88 original
closed-shell structures, activities and the 59-training/29-test partition.
The source archive and SDF are hash-verified. Source ACTIVITY values, ionic
states and explicit-H coordinates are preserved; no guessed IC50 conversion,
potency-selected subset or residual-based exclusion is used.

| Quantity | Actual result |
|---|---:|
| Nested training-CV q² | 0.612159 |
| Held-out predictive R² | 0.629830 |
| Held-out RMSE | 0.668703 |
| Held-out MAE | 0.527687 |
| Held-out Spearman | 0.729020 |
| Full-selection training-label permutation p | 0.01 (99 permutations) |
| Selected PLS components | 4, with unchanged n/5 cap |
| Same-split 2D ECFP4 RF RMSE | 0.950122 |
| Training-mean baseline RMSE | 1.118904 |

The fields are SARscope-native MMFF94 buffered 14-7 and buffered Coulomb,
with a 1 Å training-derived grid, independent ±30 kcal/mol plateaus,
training-fold screening/block scaling and nested component selection.
These settings were already documented before this demonstration; no
alignment, grid phase, field subset, split or seed was chosen from these
newly calculated test results. The main cohort's earlier positive MMFF
result was known and explicitly recorded when choosing it as a demo.

The 2,000-resample paired scaffold-block intervals cover 24 held-out Murcko
blocks:

- Test R²: **0.351–0.801**.
- Test RMSE: **0.484–0.843**.
- RF RMSE minus 3D RMSE: **0.0315–0.5469**, estimate 0.2814.

The latter supports a conditional error advantage over this RF baseline on
this historical partition, not general method superiority. Intervals do not
cover alignment uncertainty, experimental error, retraining uncertainty or
future-distribution shifts. All 29 test compounds fall within the training
PCA box; that coarse range check is not a calibrated confidence probability.

## Prediction reuse and leakage checks

- Changing only held-out activities by +100 leaves fitted predictions and
  training out-of-fold predictions unchanged.
- The evidence archive now includes `prediction_at_scaled_zero` alongside
  coefficients, column mask/scales and fitting-column means in `fields.npz`.
  This accounts for PLS's internal centring without an executable pickle.
- Replaying numerical exports reproduces all predictions within 7.2 × 10⁻¹⁵.
- A separate audit removes molecule properties, recalculates only the 29
  held-out molecules' fields on the frozen grid, and applies exported
  coefficients without refitting. Fields agree exactly and predictions agree
  within 7.2 × 10⁻¹⁵. This requires a supplied alignment in the same frame;
  it is not a general unaligned-SMILES predictor.

## Limits that remain visible

Half-grid shifts in x/y/z and the common-translation numerical control pass
the unchanged engineering checks. However, **all three small relative-pose
stress tests fail**: 95th-percentile training OOF prediction changes are
0.770–1.085 log units, with maxima up to 1.738. They are training-only
refitted diagnostics, not held-out tuning or new-compound error bars.
Contours and potency-gain hypotheses therefore remain alignment-sensitive.
Do not describe this example as robust automatic lead optimisation.

The supplied alignment's common-core qualification remains unassessed and
the package-wide experimental-status check remains failed. Those facts do
not erase the measured fixed-frame predictive result, but they limit its
interpretation. Model-level results and package certification are distinct.

A second frozen case discarded all supplied CCR5 conformers and attempted
core-constrained alignment directly from its 75 measured SMILES, with a
reference chosen from training structures by size/ID—not potency. It failed
at compound 9/75; no partial cohort was fitted or substitute route selected.
The failed-embedding path now reports a clear absence of a conformer rather
than `Bad Conformer Id`, with regression tests for empty, negative and stale
conformer IDs. No stereo constraint or timeout was relaxed.

## Evidence and repeatability

- [Readable illustrated report](../benchmarks/field_qsar/end-to-end-demo-20261019-v2/report.html).
- [Compact frozen evidence](../benchmarks/field_qsar/end-to-end-demo-20261019-v2/).
- Local full evidence archive:
  `runs/end-to-end-demo-20261019/thrombin_source_alignment/complete_evidence.zip`.
- Raw pre-execution protocol, numerical source snapshot and both declared
  cases remain under `runs/end-to-end-demo-20261019/`.
- The v2 compact report corrects report wording/scope metadata only; it uses
  exactly the same frozen model run. The initial report is retained.

The compact evidence excludes upstream structures. Source structures keep
their separate CC BY-NC 4.0 terms; the failed CCR5 case's data retain the
upstream CC BY-SA 4.0 attribution. Neither external program code nor an
external executable was used by the worked-example runner.

Verification: **87 selected workflow/geometry/export/evidence tests passed**,
including actual Streamlit runs of both primary profiles, followed by **64
related field/UI/interpretation/reproduction tests passed**. Targeted lint and
diff-whitespace checks pass. The
[testing record](../benchmarks/field_qsar/end-to-end-demo-testing-20261019/)
retains an initial example-switch test failure and the successful rerun;
this is not claimed as a fresh full-repository suite run.

```bash
.venv/bin/python -m benchmarks.validate_end_to_end_demo prepare \
  --archive /path/to/jm0497141_si_002.zip --out runs/demo-NEW
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.validate_end_to_end_demo run --out runs/demo-NEW
.venv/bin/python -m benchmarks.summarize_end_to_end_demo \
  --source runs/demo-NEW --out benchmarks/field_qsar/demo-report-NEW
```

Use fresh output directories; no frozen evidence is overwritten. The
separately licensed source archive is fetched only with explicit user
action in the app. New runs freeze current source hashes before execution.

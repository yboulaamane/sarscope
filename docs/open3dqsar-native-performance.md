# Native Open3DQSAR performance comparison

Executed 19 October 2026. **Open3DQSAR's native fields performed better on
these two historical held-out sets than SARscope's existing five-field
Gaussian profile.** Both standalone modelling and a common nested-modelling
comparison were run; these are distinct experiments.

No app backend, default, scientific threshold or manuscript was changed.
This comparison is not a new independent benchmark or a universal ranking.

## Original supplied alignments

All 88 thrombin and 111 AChE compounds were retained, with unchanged measured
activities and historical 59/29 and 74/37 training/test partitions.

| Field source and modelling workflow | Thrombin test R² | Thrombin RMSE | AChE test R² | AChE RMSE |
|---|---:|---:|---:|---:|
| SARscope Gaussian five-field profile + SARscope nested PLS | 0.5072 | 0.7716 | 0.4890 | 0.9223 |
| Native Open3DQSAR fields + native SD/BUW/three-component PLS | 0.5166 | 0.7642 | 0.5804 | 0.8358 |
| Native Open3DQSAR fields + SARscope nested PLS | 0.6298 | 0.6687 | 0.7455 | 0.6510 |

The third row reuses the same nested modelling engine and outer partitions as
the first row. It does not reuse SARscope-generated fields: Open3DQSAR itself
calculates the descriptors. The representations differ in equations, charge
parameters and number of fields (two versus five), so the comparison is not
an isolated test of one equation or an exact CoMFA/CoMSIA equivalence check.
The very small thrombin improvement in the native three-component workflow
must not be described as a significant advantage.

## What the external program computed

Official standalone Open3DQSAR 2.3 was executed as a black box, using the
public bundled user manual and MM sample command file, not algorithm source.

- Native MMFF94 **buffered 14-7 van der Waals** field with an alkyl-carbon
  (`CR`) probe, plus a native MMFF94 Coulomb field.
- Constant dielectric 1, no probe smoothing, positive/negative 30 kcal/mol
  cutoffs. These are **not** SARscope Gaussian fields or Tripos LJ 12-6 fields.
- A 1 Å grid with 4 Å margin, derived from training molecules only, using the
  same enclosing-grid rule as the previous SARscope experiments.
- Test membership was assigned **before** field pretreatment. Activities,
  molecular membership and input hashes were checked against the primary
  source and previous frozen cohorts.
- Native modelling: SD cutoff 2 kcal/mol, Block Unscaled Weighting (BUW),
  **three prespecified PLS components**. No held-out tuning or variable-search
  campaign was performed. Native cutoff/inactive-column treatment remains
  part of this workflow.
- Native leave-one-out CV was also executed. It is not nested CV; variance
  screening/scaling were set on the outer training set before invoking LOO.
  Its q² is therefore not interchangeable with SARscope's nested q².

## Common nested-modelling comparison

Clipped native fields were exported before native SD screening and BUW.
All field blocks, rows and grid dimensions were accounted for; nonfinite,
missing, constant-entire-field and out-of-cutoff matrices would be refused,
not repaired or silently filtered. ASCII export is limited to its printed
precision. The exported x-fast grid order was explicitly converted to
SARscope's z-fast matrix order and unit-tested on an asymmetric grid.

The downstream procedure was unchanged: three-fold nested PLS, training-fold
variance screening, per-field budgets with at most 2000 columns total,
training-derived block scaling, component candidates one through four, and
the n/5 component cap. Both native-field models selected four components
using training data, versus three in the original Gaussian-profile fits.
The two-field `comfa` setting in the adapter selects filtering bookkeeping;
it does **not** cause SARscope to recalculate the imported descriptors.

| Alignment and field profile | Thrombin nested q² | Thrombin test R² | AChE nested q² | AChE test R² |
|---|---:|---:|---:|---:|
| Original alignment, SARscope Gaussian fields | 0.6071 | 0.5072 | 0.0519 | 0.4890 |
| Original alignment, native Open3DQSAR fields | 0.6122 | 0.6298 | 0.4828 | 0.7455 |
| Open3DALIGN realignment, SARscope Gaussian fields | 0.5470 | 0.4406 | 0.1670 | 0.4794 |
| Open3DALIGN realignment, native Open3DQSAR fields | 0.3295 | 0.5942 | 0.4890 | 0.6544 |

All four native-field cases are reported. Original-alignment results are the
primary comparison, not a route selected after seeing test scores. Single-
conformer Open3DALIGN realignment was not consistently better. AChE's native-
field nested q² remains below 0.5 despite its improved test R²: do not turn
this experiment into a blanket validated-model claim.

No new permutations were run. The data and previous results were already
known, so this is retrospective sensitivity evidence, not prospective or
unseen-family validation.

## Standalone native workflow and runtime

| Cohort/alignment | Native LOO q² | Native test R² | Native RMSE | Wall time, one CPU |
|---|---:|---:|---:|---:|
| Thrombin/original | 0.5872 | 0.5166 | 0.7642 | 4.40 s |
| Thrombin/Open3DALIGN | 0.5456 | 0.5278 | 0.7553 | 4.30 s |
| AChE/original | 0.3714 | 0.5804 | 0.8358 | 6.41 s |
| AChE/Open3DALIGN | 0.3890 | 0.5367 | 0.8783 | 6.75 s |

These local wall times include import, native field calculation, ASCII
export, pretreatment, model fitting, LOO and prediction. Alignment, downloading,
environment setup, result parsing, the common nested model and the Streamlit
interface are excluded. Common-engine modelling alone added about 0.46–0.59 s
per case. These are single local measurements, **not** speed ratios against
SARscope, deployment timings or cold-start estimates.

## Interpretation

The results justify evaluating MMFF molecular-interaction fields as an
additional field profile or optional external backend. They do not show that
SARscope's existing equations are incorrectly implemented: different fields,
charges, filtering and component counts can produce different predictive
performance even when both solvers are numerically correct.

The earlier [same-descriptor PLS comparison](external-3dqsar-comparison.md)
already showed solver agreement. This experiment adds the missing native
field-generation performance test. Before changing defaults or making a
general performance claim, freeze these profiles and test additional unseen
series. The external legacy binary/OpenBabel runtime also needs a separate
packaging assessment before any Streamlit integration.

## Evidence and reproduction

The frozen driver is [open3dqsar_native.py](../benchmarks/open3dqsar_native.py).
Compact protocols, measurements, hashes and targeted test results are in
[open3dqsar-native-20261019](../benchmarks/field_qsar/open3dqsar-native-20261019/).
Full raw fields, commands, logs and per-compound predictions are under the
Git-ignored `runs/open3dqsar-native-20261019/` directory. A premature first
analysis was retained in `analysis-attempt1/`; analysis was repeated after the
fourth external job completed, without rerunning fields or changing settings.

The external binary and launch wrappers are the same hash-checked isolated
runtime used in the previous comparison. SARscope's relevant production
modules were checked unchanged against that earlier frozen source manifest.
Primary source data retain **CC-BY-NC-4.0** terms (Sutherland et al.,
doi:10.1021/jm0497141); they and their raw structural derivatives are not
relicensed as MIT. No GPL binary or algorithm source was added to this repo.

```bash
python -m benchmarks.open3dqsar_native prepare --out runs/new-native-check \
  --binary-dir /path/to/open3dtools/bin --babel-dir /path/to/babel-launchers
python -m benchmarks.open3dqsar_native generate --out runs/new-native-check \
  --binary-dir /path/to/open3dtools/bin --babel-dir /path/to/babel-launchers
# Wait for generate to finish before analysing.
python -m benchmarks.open3dqsar_native analyze --out runs/new-native-check
```

The adapter requires the locally retained, hash-verified primary archive and
previous fixed-reference alignment outputs; it does not fetch substitutes.
Reference: [official Open3DQSAR distribution and bundled manual](https://sourceforge.net/projects/open3dqsar/files/binaries/linux/).

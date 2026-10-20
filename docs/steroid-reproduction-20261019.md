# Published steroid benchmark repetition and alignment sensitivity

**Follow-up:** the [positive thrombin worked example](end-to-end-3d-demo.md)
adds full fixed-frame prediction checks. Failed constrained embedding now
reports an explicit absence of a conformer instead of the conformer-ID
exception documented in this earlier run. This error-handling correction
does not turn the failed steroid alignment into a success; its frozen
outputs below are unchanged.

SARscope independently repeats the published **SEH training/LOO** result on
the steroid/corticosteroid-binding globulin (CBG) benchmark. It does **not**
establish good external prediction or a successful generated-alignment demo.
This is a repetition of the historical benchmark already investigated in
October, with a new generated-alignment/native-pipeline comparison—not a new
independent dataset.

The primary comparison is [Haga et al., Pharmaceuticals 2025, 18:440,
Table 1](https://doi.org/10.3390/ph18030440). This dataset was chosen because
measured activities, deposited aligned structures, the 21/10 partition and
numerical targets are available. Selection did not depend on the new scores.

## Repeated published-protocol calculation

| Quantity | Published Py-CoMSIA SEH | SARscope-only repeat |
|---|---:|---:|
| LOO q² | 0.609 | 0.607851 |
| Training R² | 0.917 | 0.917438 |
| Selected PLS components | 3 | 3 |

Both numerical metrics meet the previously declared absolute tolerance of
0.005. Component selection evaluated the complete historical 1–12 search;
three components were not forced. All 31 valid source structures were
accounted for, alongside the source training file's known unreadable trailing
record. No valid compound was excluded by residual or model performance.
Source activities were used unchanged: no new pIC50 conversion or invented
measurement units.

This calculation uses SARscope fields, preprocessing and modelling; no
reference executable, implementation source, fitted model or processed
reference matrix is used. It preserves supplied explicit-H coordinates.
The historical scaling uses both subsets' descriptors and centres the test
subset separately. It is deliberately separate from production modelling.

The genuine historical test predictions give **R² = −1.124886** and
**RMSE = 0.897916** on the original ten test molecules. Adding the measured
test-mean offset reproduces the earlier reference-export statistic of
R² = 0.319005, but is an explicitly labelled oracle diagnostic—not a model
prediction. The ordinary prediction column and native workflow do not use
test activities. Matching the training result therefore must not be described
as reproducing useful held-out prediction.

## New fixed native-pipeline comparison

All settings were frozen before fitting. Two field profiles—Gaussian SEH and
MMFF94 steric/electrostatic—were tested with three alignment routes. Native
modelling used a training-derived 1 Å grid, block scaling, fold-local screening
and component selection, nested three-fold CV and the unchanged n/5 component
cap. Hydrogens were removed and regenerated consistently across native routes.
Neither field profile was selected using its test score. No permutations were
run: this is a sensitivity experiment, not model qualification.

| Supplied-alignment native profile | Nested q² | Test R² | Test RMSE |
|---|---:|---:|---:|
| Gaussian SEH | 0.511164 | −0.489285 | 0.751721 |
| MMFF94 SE | 0.707504 | −0.882666 | 0.845190 |

These results demonstrate why internal q² alone is insufficient. Native and
historical q² estimates have different preprocessing and selection protocols;
they are not interchangeable method rankings.

Neither generated route yielded an accepted complete series:

- **Rigid MCS:** preparation failed the whole-series common-core requirement
  (fewer than three compatible atoms). Steroid functional-group/aromaticity
  changes matter under the current strict mapping rules. This does not prove
  that no useful pharmacophore alignment exists; it means this automatic
  common-core route cannot accept this complete series as configured.
- **Constrained:** the second compound, androstanediol, reported `Bad Conformer
  Id`. A separate same-settings diagnostic found that RDKit's constrained
  embedding returned `[-1]` with no conformers, after which the code tried to
  retrieve a conformer at `align3d.py:169`. This is a confirmed error-handling
  weakness for failed embedding, not evidence of a successful alignment or a
  reason to exclude the compound. Its correct handling is a follow-up fix;
  no production code, stereochemistry, threshold or time budget was changed
  in this experiment.

All six declared outcomes, including the four field/alignment combinations
blocked by preparation, are retained. The earlier EGFR implementation checks
remain valid for their tested cohort; they do not establish universal
alignment success.

## Other candidate checks

The antiplasmodial benzopyrans ([2021](https://doi.org/10.3390/molecules26175249))
and HDAC1 benzamides ([2025](https://doi.org/10.3390/ijms26209970)) have public
supplement ZIPs containing PDFs, not the selected aligned molecular ensembles.
They cannot currently support exact alignment-dependent reproduction from
those files alone. The primary Sutherland GPB dataset has 66 records, a 44/22
partition, no duplicated encoded graphs and no radicals, but three structures
have unspecified stereochemistry. Those states were not guessed to produce
a clean-looking demonstration. None of these candidates was fitted here.

## Evidence and commands

The [compact evidence](../benchmarks/field_qsar/steroid-alignment-reproduction-20261019/)
contains the frozen protocol, source accounting, all outcomes, component
curves, per-compound predictions, pipeline checks and SHA-256 manifests.
Upstream SDFs and raw field arrays are not redistributed in that directory.
Raw inputs/outputs remain locally under the ignored `runs/` directory.

```bash
.venv/bin/python -m benchmarks.reproduce_steroid_alignment prepare \
  --out runs/steroid-repeat-NEW
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -m benchmarks.reproduce_steroid_alignment run \
  --out runs/steroid-repeat-NEW
```

Use a new output directory. Generated routes require the normal optional
preparation dependencies. The app's numerical defaults and validation gates
were not modified, and nothing was committed or pushed by this experiment.

**Decision:** retain the steroids as a numerical-reproduction fixture, not a
"good-results" predictive demo. Do not claim that generated alignment is
validated across this chemistry or that the entire 3D-QSAR package is
predictively validated by this result.

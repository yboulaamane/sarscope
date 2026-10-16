# Independent SARscope reproduction

**This run uses SARscope's own code from data to predictions.** It does not
call or import Py-CoMSIA, load its fitted models, or use its processed arrays
as calculation inputs. RDKit/NumPy and the existing scikit-learn-based
SARscope regression engine are the numerical dependencies.

The previous `fixed-split-api-*` runs were reference-software comparisons,
not this independent implementation. They remain frozen as comparators.

## What runs independently

`python -m sarscope.field_reproduction` fetches only two hash-pinned source
SDF data files. It performs SARscope's own coordinate/stereo checks, retains
the supplied explicit hydrogens, constructs its own rounded-extent grid,
calculates its own SEH fields, scales/centres them, searches PLS components
with the existing regression/CV implementation, fits and exports predictions.

The formulas are based on Haga et al. (2025),
[doi:10.3390/ph18030440](https://doi.org/10.3390/ph18030440), Methods 4.4–4.7.
No external implementation source was read or translated. Ambiguities in
the prose were checked against numerical reference outputs. The data are
already pre-aligned; this is not a reproduction of an independent alignment
generation protocol or experimental binding-mode validation.

```bash
.venv/bin/python -m sarscope.field_reproduction \
  --out benchmarks/field_qsar/independent-new-run
```

No PyVista, VTK or reference-program environment is needed. `--offline-data`
accepts a directory containing the two original SDF files with the same
hashes. Raw source structures are not redistributed here. `--include-native`
additionally runs the ordinary training-only workflow with the explicitly
recorded comparison settings; every outcome is retained.

## Own numerical results

The [independent assessment](independent-20261016-v3/assessment.json) uses
all 21 training and 10 test molecules. One unreadable trailing source record
is explicitly accounted for. No valid molecule was removed for its residual.

| Quantity | Published | SARscope only |
| --- | ---: | ---: |
| LOO q² | 0.609 | 0.607851 |
| Training R² | 0.917 | 0.917438 |
| Selected components | 3 | 3, selected from the full 1–12 search |

These meet the recorded absolute 0.005 tolerances. The entire component
curve, genuine fitted predictions, charge audit, alignment checks, scaler
state, coefficients, versions and code/data hashes are retained.

This is a **historical reproduction profile**, not unbiased validation:
the field-block SDs use all 31 molecules and train/test columns are centred
separately. The app does not offer this preprocessing. Historical uncapped
component search is also refused by the native app pipeline, which retains
its n/5 cap and fold-local preprocessing/selection.

## Important correction: exported historical test predictions

The frozen reference predictions are numerically equivalent to adding
`mean(measured test activities) - mean(training activities)` to the own
fitted predictions under the separately centred historical representation.
The offset is **0.7401761995 activity units**, constant across all 10 test
compounds. After that shift, maximum disagreement with the reference exports
is about 3 × 10⁻¹⁵.

| Historical evaluation | R² |
| --- | ---: |
| Genuine fitted predictions, before using test-label mean | −1.124886 |
| Diagnostic after adding the measured test-mean offset | 0.319005 |

**The second row uses test labels and is not a prospective model prediction.**
It must not be used to claim predictive validation. The CSV deliberately
separates the real `prediction` column from
`oracle_test_mean_diagnostic_NOT_prediction`. Model fitting and the ordinary
prediction column never receive test activities. This reconstructs the
behaviour of the pinned exports, not every workflow in the reference tool;
no claim about authors' intentions is made.

The native column-scaled sensitivity run in `independent-20261016-v2/`
also fails dramatically on test compounds despite high internal q². That
aggressive all-column setting is not the app's default; its failure is
retained, not converted into a validated model or silently excluded.

## Tests and limits

- A synthetic end-to-end run rejects reference-software imports and runs
  without any reference arrays/models.
- Shifting only test labels by +100 leaves fitted predictions and training
  q² unchanged. Only the explicitly labelled oracle diagnostic changes.
- Full capped `max_q2` selection evaluates every candidate rather than
  stopping at an intermediate plateau. The original early-stop option remains
  available and all settings are recorded.
- Native validation rejects historical uncapped mode. Field/scaling choices
  exposed in Streamlit remain training-only.

Source/geometry-dependent preprocessing interpretations were resolved openly:
the initial independent run used per-column scaling and did not match;
its outputs remain in `independent-20261016/`. The corrected interpretation
uses one SD over each complete field block, with separate subset centres,
matching the frozen intermediate arrays. No force-field parameter was tuned
to test R² and no empirically fitted field scale factor was inserted.

This establishes independent **SEH training/LOO benchmark agreement**. It
does not establish useful external prediction, complete CoMSIA hydrogen-bond
equivalence, CoMFA reproduction, contour correctness across conventions or
multi-target prospective validation. The app remains experimental.

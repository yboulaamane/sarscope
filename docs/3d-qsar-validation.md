# Qualification of SARscope's own field methods

**Reference direction updated:** the user subsequently requested following
published SYBYL methods using public documentation, without vendor access.
See the [component-level implementation and source audit](sybyl-reference.md).
The earlier own-method results below remain frozen evidence, not vendor
compatibility tests. A subsequent public-SE candidate has now consumed the
reserved ATA cohort and failed its predictive criteria. See the
[latest validation attempt](public-se-validation.md) before interpreting the
earlier evidence or planning a fresh-cohort evaluation.

Scope agreed on 16 October 2026: **SARscope's documented numerical methods**,
not exact legacy/proprietary CoMFA/CoMSIA compatibility. There is no universal
"fully validated" certificate covering every QSAR model a package can fit.
Implementation verification, evaluation of a workflow, and validation of an
individual endpoint/model are separate questions.

The [OECD validation guidance](https://doi.org/10.1787/9789264085442-en)
addresses defined endpoints, transparent algorithms, applicability domains,
statistical performance and interpretation. This is a useful reporting
framework, not an OECD endorsement or regulatory certification of SARscope.

## What the implementation actually promises

- LJ/Coulomb-style fields use RDKit van der Waals radii, UFF well depths,
  a +1e carbon probe, constant dielectric, an explicit minimum distance and
  steric truncation. Electrostatic overlaps remain undefined. No proprietary
  atom table or exact historical CoMFA equivalence is promised.
- Positive-sum Gaussian fields use atomic radius cubed, partial charges,
  Crippen atomic logP contributions and atom-centred Lipinski donor/acceptor
  SMARTS. These indicators are the specified feature definition, not a
  directional protein/ligand hydrogen-bond energy or a proprietary CoMSIA
  reconstruction. Protonation and tautomer choice remain input assumptions.
- Gasteiger and MMFF94 are explicit choices. Charge-model fallback is opt-in
  and counted. Generated geometry is checked for stereochemical consistency;
  this does not establish an experimental bound conformation.
- Train-derived filtering/scaling and nested component selection use the
  existing regression engine. The n/5 component cap is a heuristic, not proof
  that a fitted model will generalise. The historical reproduction profile
  is not the native validation pipeline.

## Numerical and statistical verification

The offline suite now includes independent scalar LJ/Coulomb/Gaussian oracles
for every documented field, with a small frozen RDKit/UFF/Crippen parameter
snapshot. It also checks common-frame translation/rotation covariance under
both built-in charge models, a closed-form one-component PLS oracle, charge
zero/sign controls, exact chunking, finite-coordinate/stereo safeguards,
training-only masks/scalers, raw-coordinate contour coefficients and export
ordering. The numerical oracles do not call production field/property helpers.

A deterministic synthetic signal has nested q² > 0.9; 19 full-pipeline
permutations have negative mean q² and none reaches the real signal. These
are software/statistical controls, not experimental activity measurements.
Existing tests also ensure altered held-out activities cannot alter fitted
historical-profile predictions, and bad calibration does not receive a pass.

Local compatibility checks pass on both tested stacks: **68 backend tests**
in a fresh Python 3.12 environment (RDKit 2024.09.6, NumPy 2.2.4, pandas 2.2.3,
scikit-learn 1.6.1), and **71 including Streamlit tests** on Python 3.13
(RDKit 2026.09.1, NumPy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1).
The [JUnit records](../benchmarks/field_qsar/compatibility-20261016/) are retained.
The 3.12 environment has no Streamlit installation; its missing browser tests
are not a second browser validation. Separate CI jobs now repeat backend
verification with the two recorded numerical constraint files. These constrain
the numerical stack, not every transitive dependency or every supported OS.

The entire native campaign was also rerun in the fresh 3.12 environment,
with [separate evidence](../benchmarks/field_qsar/native-validation-20261016-py312/).
Gaussian-field predictions, selected component counts, q², test metrics and
permutation p-values agree with the 3.13 run to the exported precision
(maximum prediction difference 0 in these runs). All LJ/Coulomb abstentions
recur. The RF baseline is not bitwise identical: its largest test RMSE
difference is about 0.00219 log units across scikit-learn versions. This is
same-machine cross-environment replication, not independent-laboratory review.

Run locally (no source-data download):

```bash
make verify-fields
# If make is unavailable:
.venv/bin/python -m pytest -q tests/test_field_numerics.py tests/test_field_campaign.py \
  tests/test_field_qsar.py tests/test_field_reproduction.py tests/test_field_reference.py \
  tests/test_app_field_qsar.py
```

## Native multi-family campaign: actual results

The [frozen protocol](../benchmarks/field_qsar/native-validation-protocol.json)
was recorded before scoring these cohorts in SARscope. It fixes source-file
hashes, the original train/test membership, grid, charges, filtering, capped
component search, CV seeds and 99 full-pipeline training-label permutations.
No replacement preset was chosen after inspecting test scores. There is no
joint train/test field scaling or measured-test-mean prediction adjustment.

Source SDFs are fetched by exact commit/hash, not bundled. The upstream
repository declares CC BY-SA 4.0; the source is attributed to
[Haga et al.](https://doi.org/10.3390/ph18030440), with antecedent datasets
from [Sutherland et al.](https://doi.org/10.1021/jm0497141) and
[Aher et al.](https://doi.org/10.1007/s00894-007-0173-z).
No external QSAR executable, model or processed matrix is used.

| Native Gaussian fields | Train/test | Nested q² | Test R² | Test RMSE | 2D RF RMSE | Permutation p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Acetylcholinesterase | 74/37 | 0.038 | 0.277 | 1.097 | 0.879 | 0.04 |
| CCR5 | 63/12 | 0.753 | −0.692 | 0.996 | 0.848 | 0.01 |
| Thrombin | 59/29 | 0.441 | 0.548 | 0.739 | 0.955 | 0.01 |

These are three separate historical endpoints, **not pooled activities**.
Thrombin's test R² has a conditional scaffold-block bootstrap 95% interval
of approximately 0.104–0.835. The 500 resamples quantify uncertainty for this
fitted model/test population; they do not establish robustness to retraining,
temporal shift or repeated test-set use.

All three native LJ/Coulomb cases **refuse scoring**: held-out molecules have
undefined electrostatic fields in retained columns. These are operational
abstentions under the documented policy, not R² = 0, successful predictive
models, or permission to mask columns using test molecules. This is a serious
usability limitation for general-purpose CoMFA-style prediction.

The [complete evidence](../benchmarks/field_qsar/native-validation-20261016-v2/)
includes every case, predictions, same-split RF/mean baselines, 99 permutations
per completed model, 500 bootstrap replicates, failed checks, source accounting,
graph/split audit, versions, code hashes and an SHA-256 manifest. Results show
that internal q² and a permutation p-value can coexist with poor held-out
performance. Neither metric certifies a useful model by itself.

Source screening found duplicate training graphs in ACE and thermolysin;
those require a grouped-CV/source-adjudication protocol. The campaign refuses
duplicates rather than silently removing compounds or leaking them across CV.
The selected three cohorts have unique encoded graphs and no exact encoded
graph train/test overlap. This does not eliminate close-analogue similarity,
cross-publication dependence, or all possible standardisation equivalences.

The initial `native-validation-20261016` artifacts are retained: they recorded
the noncanonical CV label `kfold`, which took the same runtime branch as
`default`. V2 fixes the configuration metadata/validation only and reproduces
the metrics; no scientific preset changed. No source bytes or valid rows were
repaired to improve outcomes.

Reproduce explicitly, outside the Streamlit request loop:

```bash
.venv/bin/python -m sarscope.field_campaign \
  --out benchmarks/field_qsar/native-validation-new-run
```

The command refuses to overwrite prior evidence. `--offline-data DIR` accepts
the six original source SDF files, still subject to hash verification.

## Release decision and remaining work

**Do not yet label the complete package fully validated.** The numerical
tests establish important implementation properties; mixed predictive results
do not automatically invalidate a correct field formula. However, a positive
literature training score is not a substitute for a documented, usable and
independently reproducible prediction workflow.

| Requirement for the agreed own-method scope | Current evidence / next acceptance step |
| --- | --- |
| Documented field/property/charge conventions | Explicit specification and scalar/invariance tests; no proprietary parity claim needed |
| Leakage and negative controls | Fold-local tests, synthetic controls and actual 99-permutation distributions retained |
| Native performance characterised on multiple families | Three frozen historical test cohorts completed for Gaussian fields; all LJ/Coulomb abstentions retained |
| Useful LJ/Coulomb prediction contract | Not established on these cohorts; resolve overlap/abstention behaviour without test-derived masks or silently redefining fields |
| Generated-alignment robustness | Pre-aligned campaign does not validate generated binding modes; evaluate fixed reference/core choices, charge and alignment perturbations without choosing by test scores |
| Reproducible tested environments | Numerical/backend checks pass on two Python/library stacks; retain the native cross-stack replication and run CI, packaging and live-host smoke tests before release |
| Independent final evaluation | These are public, now-inspected test splits. Freeze the final supported workflow and evaluate fresh untouched data; obtain external scientific review of methods and evidence |

An individual weak model must remain labelled weak even after the software
is qualified. Conversely, certification of an implementation must not require
discarding every dataset where a 2D baseline wins. The defensible eventual
claim is a **verified implementation with a specified validation scope and
per-model qualification**, not universal predictivity.

# Literature benchmark status — 16 October 2026

The [own-method qualification record](3d-qsar-validation.md) adds a frozen
native three-family evaluation. Native Gaussian test R²: AChE 0.277, CCR5
−0.692, thrombin 0.548, with 99 nested training-label permutations each.
All three LJ/Coulomb cases abstain on undefined held-out electrostatic
fields. This is actual native-pipeline evidence, not historical API emulation;
it does not qualify the complete package. Exact proprietary compatibility is
outside the user-agreed own-method scope.

**Experimental, with independent SEH training/LOO agreement established.**
SARscope now performs the full historical field-to-model calculation itself,
without importing/calling the reference API or reading its processed matrices
as calculation inputs. The independent run gives q² = 0.607851, training
R² = 0.917438 and three selected components. See the
[independent evidence and test-mean caveat](../benchmarks/field_qsar/INDEPENDENT.md).

The earlier API-only result below was a comparator, not the requested own
implementation. Its exported test predictions match a measured-test-mean
adjustment. That reproduces R² = 0.319 as an oracle diagnostic, not a genuine
prospective prediction. Our fitted prediction column and native app do not
perform that adjustment. General external validation, CoMFA and hydrogen-bond
field equivalence remain incomplete.

The [reference-reproduction record](../benchmarks/field_qsar/README.md)
contains source selection, frozen protocols, commands, record accounting,
actual results and remaining acceptance gates.

The following distinct experiments have been executed:

- **Explicit-split historical API:** using either reference or SARscope SEH
  fields, the author's 1–12 component search selects three components. On
  the fixed 21/10 cohort, q² = 0.607851 (published 0.609), training R² =
  0.917438 (published 0.917), test R² = 0.319005. Cross-source predictions
  agree to about 10⁻¹⁵ without manual sign/scale conversion. The reference's
  historical preprocessing is external to SARscope and is not copied into
  the app. Both sources use the same reference PLS API, not independent solvers.

- **SARscope comparison:** eight frozen steroid/CBG configurations, each with
  21 training and 10 external compounds. A three-field block-scaled setting
  reaches LOO q² = 0.665968 but external R² = 0.039756 using the paper's
  labels. Other configurations perform worse, including pathological
  extrapolation under column scaling. No test outlier was removed.
- **Independent reference execution:** the unmodified, pinned Py-CoMSIA CLI,
  outside SARscope, gives q² = 0.607851 versus published 0.609. Its printed
  training/test R² values belong to an internal 16/5 split. Scoring the ten
  supplied external predictions independently gives R² = −1.092875 and
  RMSE = 0.891127. This is not reproduction of the paper's full fixed 21/10
  evaluation and does not establish SARscope field equivalence.
- **Isolated raw-field comparison:** all 1626 Gasteiger charges agree exactly.
  On identical coordinates and grid, SEH spatial fields agree to numerical
  precision after empirically inferred global factors (−1, 337 and 25), but
  raw field equality fails. Hydrogen-bond fields differ spatially. These
  post-hoc factors were not added to production; this is a diagnostic,
  not full method reproduction or predictive validation.

The reference is [Haga et al. (2025)](https://doi.org/10.3390/ph18030440),
Tables 1–2 and Methods 4.4–4.7. The executable and its split files are frozen
at commit `0b30179ecc20aac5a003577b4437399a8fb20bca`. The earlier unsplit source
is frozen at `c64af25644619f942bf4aebf85a80dba8f378f06`; its SDF SHA-256 is
`32000e0fafb6b0f4ad2af42854038615962ebe1fb46801ebfd16a7743b85a26c`.

Remaining work includes native end-to-end reproduction, raw sign/scale and
contour conventions, hydrogen-bond fields and CoMFA. Reference arrays have
been obtained through the runtime API without inspecting its method source. Keep
historical preprocessing emulation distinct from production training-only
validation. A second independent dataset is needed before broader claims.

No GPL implementation source was consulted or copied. The external program
was executed unmodified, not incorporated into SARscope. Upstream structures
are not redistributed; source hashes and derived numerical evidence are
retained. Optional receptor/redocking/assembly/covalent-reference extensions
remain separate and are not validated by these experiments.

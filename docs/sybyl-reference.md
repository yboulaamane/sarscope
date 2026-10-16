# SYBYL as the method reference

Requested on 16 October 2026. The goal is a faithful, independently implemented
published method, not a novel replacement for CoMFA/CoMSIA. The public target
we can currently inspect is the **1999 SYBYL 6.6 QSAR manual**. This does not
establish the internals of every later SYBYL/SYBYL-X release.

**Implemented: documented method components, not full SYBYL equivalence.**
The existing SARscope variant remains available and unchanged by default.
The reference-components path is separate, records every setting and refuses
unsupported field definitions rather than renaming approximations as SYBYL.

**Latest validation attempt:** numerical/cross-environment checks pass, but
the additional ATA predictive evaluation fails. The cohort is no longer
reserved. See the [complete outcome and concrete problems found](public-se-validation.md).

## Reference contract

| Component | Implemented / remaining limitation |
| --- | --- |
| Organic-atom steric parameters | Public Tripos-5.2 radius/well-depth pairs replace RDKit/UFF in the separate SE profile. Explicit-H environment rules are our documented parameter selector, not the complete vendor atom typer. Unknown elements/environments fail, without fallback. |
| CoMFA potential controls | LJ 12-6, constant or distance-dependent Coulomb dielectric, signed electrostatic plateau, steric plateau and abrupt cutoff. Smooth transitions, region focusing and other optional vendor modes are not implemented. |
| Missing prediction descriptors | Opt-in fitting-column means, learned separately within every fitting fold. Per-row counts and overall/electrostatic missing fractions are exported. Strict refusal remains available and remains the original default. |
| CoMSIA SE | Positive Gaussian sum, radius-cubed steric weights, current partial charges, explicit attenuation and grid. No empirical parity-fitted multipliers are inserted. Vendor numerical equality has not been demonstrated. |
| Hydrophobic CoMSIA | Not implemented in the reference profile: the documented vendor property table is not replaced with Crippen values. |
| Donor/acceptor CoMSIA | Not implemented in the reference profile: atom-centred indicators are not silently substituted for directional extension sites. |
| Charges | RDKit Gasteiger–Marsili or MMFF94, explicitly recorded. Gasteiger–Marsili is not Gasteiger–Hückel; neither is relabelled as the latter. SYBYL's documentation permits fields based on the molecule's current charges. |
| Modelling/scaling | Existing SARscope nested PLS, train-derived grid, component cap and training-only block scaling. These are not represented as exact SYBYL preprocessing, scaling-order or PLS-solver parity. |

The parameter values are from the public
[Tripos-5.2 table in the rDock documentation](https://rdock.github.io/documentation/html_docs/reference-guide/appendix.html),
with [Clark et al. (1989)](https://doi.org/10.1002/jcc.540100804) as the primary
force-field reference. Only documented numeric facts were used, not GPL
implementation source. Carbon/nitrogen/oxygen table rows share R/K values
within their respective elements. Polar hydrogen and oxidised sulfur use
separate public entries. United-atom carbon parameters are not mixed with
explicit hydrogen contributions. This is a finite supported subset, not an
assertion that every SYBYL atom type is covered.

## The overlap-policy correction

The [SYBYL prediction manual, §4.3.6](https://life.nthu.edu.tw/~lslth/QSARandCoMFA/sybyl/sybyl/qsar/qsar_theory15.html)
specifies replacing unavailable prediction descriptors with their training
column means and notifying users of replaced terms. Therefore, refusing all
overlap cases was **our strict policy**, not an inherent impossibility of
CoMFA. The earlier refusal results remain valid records of that policy, not
evidence that SYBYL cannot score those molecules.

With `missing_field_policy="training_mean"`, field computation still emits
NaN for steric-overlap electrostatics. The modelling filter calculates means
only from fitting rows; all-missing fitting columns are dropped. Remaining
missing fitting/validation/test terms use those fitting means before variance
filtering/scaling. Infinite values are numerical failures, not imputable
overlaps. No test geometry redesigns the mask, and no test activity enters
the means. Default `"refuse"` behaviour is preserved.

This is **explicit published-policy following**, superseding the earlier
project prompt's blanket prohibition on mean replacement only for this
opt-in path. It is not covert imputation in the field calculator.

The default 10% warning limit is a configurable SARscope engineering flag,
checked both overall and within retained electrostatic terms. It is **not**
a published SYBYL validation cutoff. A mean-filled molecule may fall inside
the PCA box precisely because missing terms were replaced by averages;
the export and interface retain the missingness information separately.

## Why full five-field equivalence is still not a finished claim

The [vendor's CoMSIA description](https://life.nthu.edu.tw/~lslth/QSARandCoMFA/sybyl/sybyl/qsar/qsar_theory18.html)
identifies specific atomic-property tables and directional donor/acceptor
extension sites, including exclusion of sites within 1.8 Å of a real atom.
It also specifies the positive field-sign convention. Those details matter:
matching a Gaussian equation does not establish matching spatial descriptors.
Public inspection here did not supply a complete independently reproducible
hydrophobic atom-typing table and directional site-generation specification.
No vendor reference exports/executable are available from the user.

Consequently, the reference profile exposes **steric/electrostatic fields
only**. The original five-field SARscope variant remains explicitly labelled
as RDKit/UFF/Crippen/atom-centred, not as full SYBYL CoMSIA.

The [CoMFA controls](https://life.nthu.edu.tw/~lslth/QSARandCoMFA/sybyl/sybyl/qsar/qsar_gui11.html)
document configurable dielectric functions and cutoff behaviour. The
[scaling chapter](https://life.nthu.edu.tw/~lslth/QSARandCoMFA/sybyl/sybyl/qsar/qsar_theory7.html)
describes common within-field weighting, unlike independent pointwise
autoscaling. Our training-only `block_std` implementation is compatible with
that general idea but has not been independently verified against the
vendor's exact filtering/scaling order, normalisation and solver settings.
That gap is reported, not papered over by matching one q².

## Use and evidence

In Streamlit, under **Choose fields and validation → Field definition**, select
**Public Tripos 5.2 · steric/electrostatic reference components**. CoMFA exposes
the missingness policy, signed electrostatic plateau and dielectric function.
Changing these settings hides stale model results. Missing counts/fractions
appear next to held-out predictions and are included in the evidence ZIP;
the fitted column means are exported in `fields.npz`.

The [source audit](../benchmarks/field_qsar/sybyl-sources.json) records URLs and
verified source hashes without redistributing vendor manuals or proprietary
parameter files. The [component characterisation protocol](../benchmarks/field_qsar/tripos-components-protocol.json)
uses the already-inspected AChE/CCR5/thrombin cohorts. Results are method
characterisation, **not a new independent test, a vendor benchmark
reproduction or justification for choosing a winning preset**. All outcomes,
replacement counts and failed validation checks are retained.

The [frozen component run](../benchmarks/field_qsar/tripos-components-20261016-v3/summary.csv)
completed all six cases with 99 label permutations per case:

| Cohort | Gaussian SE held-out R² | CoMFA SE held-out R² |
| --- | ---: | ---: |
| AChE | 0.446 | 0.470 |
| CCR5 | −0.671 | −0.403 |
| Thrombin | 0.561 | −6.976 |

These are SARscope predictions, **not numerical comparisons against SYBYL**.
Completion resolves the strict-policy refusal, not the predictive limitations.
The poor thrombin CoMFA result is retained without tuning it away. Three CCR5
CoMFA held-out molecules exceed the configured 10% missing-field warning
within retained electrostatic terms. All six models remain labelled
experimental/not validated; favourable permutation tests do not override
poor external prediction or the remaining method-compatibility gaps.

The earlier five-field 1 Å Gaussian candidate was not promoted or evaluated.
A later, separately frozen **public SE** candidate passed its training-only
sampling gates and was evaluated once on ATA; both its primary and secondary
models failed predictive criteria. That dataset is now consumed and must not
be used to select a replacement preset. Details are in the
[new validation record](public-se-validation.md).

The next decisive verification resource is an independently citable complete
parameter/site definition or numerical field export for a specified SYBYL
version. Without that resource, a claim of full numerical SYBYL equivalence
would be unsupported, even if our own tests all pass.

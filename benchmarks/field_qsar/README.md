# Field-QSAR reference reproduction

**Status: SARscope now independently reproduces the SEH steroid training/LOO
benchmark, without calling the author's modelling API.** See the
[independent implementation and evidence](INDEPENDENT.md) first. The older
reference-API experiments below are retained as comparators, not substitutes
for our implementation. CoMFA, five-field equivalence and general external
validation remain incomplete.

**Correction to the earlier test-R² interpretation:** the exported reference
predictions reproduce R² = 0.319 only after a measured-test-mean offset under
the historical representation. That is an oracle diagnostic, not prospective
prediction. It is not copied into the native app or its prediction column.

## Retained campaign versions

Several campaigns have numbered reruns. Every version still present is kept
deliberately, because something cites it:

| Family | Kept | Why the earlier run is retained |
| --- | --- | --- |
| `independent-20261016` | v1, v2, v3 | INDEPENDENT.md cites all three: v3 is the current assessment, v2 the native column-scaled sensitivity run, and v1 the outputs behind the corrected interpretation. Deleting v1 would leave the correction unevidenced. |
| `reference-20261016` | v1, v3, v4 | Cited as successive comparators. |
| `field-parity-20261016` | v1, v3 | Both cited. |
| `fixed-split-api-20261016` | v1, v3 | Both cited. |
| `model-parity-20261016` | v1, v3 | Both cited. |
| `native-validation-20261016` | v1, v2 | Both cited. |
| `public-se-assessment-20261016` | v1, v2 | v2 is the current recalculated assessment. |
| `results-20261015` | v1, v4 | Both cited. |
| `sensitivity-20261016` | v1, v2 | `tests/test_field_sensitivity.py` verifies v2's manifest hashes. That test skips when the directory is absent, so removing it would silently turn a real check into a no-op. |
| `tripos-components-20261016` | v1, v3, py312 | v3 is the frozen campaign; py312 is its cross-environment repeat. |

Five orphaned reruns were removed on 16 October 2026 after checking that
nothing in the repository referenced them: `fixed-split-api-20261016-v2`,
`model-parity-20261016-v2`, `results-20261015-v2`, `results-20261015-v3` and
`tripos-components-20261016-v2`. They are recoverable from git history.

## Selected reference

Haga CL, Le CN, Yang XD, Phinney DG. *Py-CoMSIA: An Open-Source Implementation
of Comparative Molecular Similarity Indices Analysis in Python.*
Pharmaceuticals. 2025;18:440. [doi:10.3390/ph18030440](https://doi.org/10.3390/ph18030440).
The academic editor is not an author.

The [author repository](https://github.com/clhaga/pycomsia) supplies split
steroid SDFs and an executable program. The initial target is its three-field
steric/electrostatic/hydrophobic (SEH) model, not hydrogen-bond fields or CoMFA.
The publication specifies spacing 1 Å, padding 4 Å, attenuation 0.3 and
21 training / 10 external compounds. Table 1 reports q² = 0.609 and training
R² = 0.917 with three PLS components. Some narrative values are inconsistent;
they must not replace the declared table targets after seeing results.

`reference-selection.json` freezes the external commit, download SHA-256,
input hashes, CLI settings, package versions and diagnostic tolerances.
The selected commit adds the licence in July 2025; its algorithm files precede
the March 2025 publication. It is not asserted to be an archived publication
release or the original SYBYL implementation.

The user-supplied *Introduction to 3D-QSAR*, Chapter 8, §§8.2–8.3, is a
conceptual reference, not an exact numerical specification: the Markdown
extraction lacks equations and has corrupted symbols. Do not redistribute it
or silently substitute its probe parameters into a different reference method.

## Explicit-split historical API — limited benchmark reproduction

The reference's runtime-documented `src.PLSAnalysisTestSets.PLSAnalysis`
accepts separate training and test fields. Unlike the CLI, it fits all 21
training molecules and evaluates the original 10 test molecules. The frozen
`fixed-split-api-protocol.json` uses the documented default 1–12 component
search, rather than forcing three components. **Three is actually selected**
by the maximum LOO q² across that complete search.

The [fixed-split assessment](fixed-split-api-20261016-v3/assessment.json) uses
the published settings and separate reference/SARscope-generated SEH fields:

| Quantity | Published | Reference fields | SARscope fields |
| --- | ---: | ---: | ---: |
| LOO q² | 0.609 | 0.607851 | 0.607851 |
| Training R² | 0.917 | 0.917438 | 0.917438 |
| Selected components | 3 | 3 | 3 |
| Test R² | 0.319 in the later narrative | 0.319005 | 0.319005 |

q²/training R² meet the prespecified absolute 0.005 tolerances. Maximum
prediction differences between the field sources are 1.78 × 10⁻¹⁵ on
training compounds and 8.88 × 10⁻¹⁶ on test compounds. **No empirically fitted
sign/scale correction was applied in this experiment.** The external API's
field standardisation handles those conventions. Full component curves,
predictions, metrics, versions, processed matrices and hashes are retained.

This reproduces the three-field benchmark approximately **using SARscope
fields and the reference's historical modelling/preprocessing API**. It is
not independent verification of every line of that API or reproduction by
SARscope's native preprocessing. The paper's combined train/test preprocessing
is not copied into production, nor is the historical 1–12 search substituted
for the native component cap. These are already-inspected historical data,
not an unbiased external validation. H-bond fields and CoMFA remain pending.

```bash
.venv/bin/python benchmarks/field_qsar/fixed_split_api_probe.py \
  --python /path/to/reference-env/bin/python \
  --out benchmarks/field_qsar/fixed-split-api-new-run
```

The first API assessment completed its fit/CSV export but failed while
serialising a NumPy integer. The corrected wrapper changes metadata encoding
only; prior results remain retained with `FAILED.md`.

## Initial CLI probe — 16 October 2026

The unmodified CLI ran in a separate Python 3.12 environment. RDKit 2024.09.6,
NumPy 2.2.4, scikit-learn 1.6.1 and PyVista 0.44.2 match the paper's named
versions. Python differs from the paper's 3.11; this is recorded, not hidden.
No external implementation source was read or translated into SARscope;
only the paper, README, executable help, runtime API signatures/docstrings,
licence, repository metadata, SDF data and execution outputs were consulted.

The retained [assessment](reference-20261016-v4/assessment.json) distinguishes
three different quantities:

| Quantity | Actual result | Interpretation |
| --- | ---: | --- |
| Reference CLI LOO q² | 0.607851 | Within the prespecified 0.005 tolerance of Table 1's 0.609 |
| CLI training R² | 0.896074 | Its exported fit uses 16 compounds, not all 21 |
| CLI-labelled test R² | 0.964991 | Five compounds split from the supplied training file |
| Supplied external-set R², independently recalculated | −1.092875 | All 10 supplied external predictions; not the CLI-labelled test metric |
| Supplied external-set RMSE | 0.891127 | Calculated from the same 10 exported predictions |

The five internal test indices are 15, 14, 9, 2 and 5 (zero-based within
the original 21-row training file), as recorded by the external program.
The graph hashes of all 10 exported external predictions were checked against
their input identities before scoring. No compound was removed for a bad
residual. The source training SDF has 21 valid records and one unreadable
trailing record; the external SDF has 10 valid records. No source bytes were
repaired. Activities already use positive `Activity` values, unlike the
earlier unsplit SDF's negative `pKi_CBG` property.

**This initial CLI invocation approximately reproduces one q², not the
paper's complete fixed-split evaluation.** Its
printed R² = 0.965 must not be advertised as performance on the ten supplied
external compounds. This finding concerns this pinned CLI invocation, not
every workflow in the software or the publication.

## Raw-field comparison against the reference API

The public runtime API provides `generate_grid` and `calc_field`. Their
signatures/docstrings were inspected, not their implementation source. Two
separate subprocesses executed the reference and SARscope field calculators
using the same Python/RDKit environment and deposited explicit-hydrogen
coordinates. Production geometry preparation was bypassed to isolate fields
from alignment and hydrogen regeneration.

The grid is 23 × 16 × 14 = 5152 points at origin (−12, −8, −7) Å, spacing
1 Å. There are 159,712 values per field across all 31 compounds.
The [field-parity assessment](field-parity-20261016-v3/assessment.json) records:

| Quantity | Finding |
| --- | --- |
| Gasteiger charges | All 1626 charges identical; maximum difference 0 |
| Steric | Reference = −1 × SARscope; residual maximum 7.11 × 10⁻¹⁵ |
| Electrostatic | Reference = 337 × SARscope; residual maximum 2.56 × 10⁻¹³ |
| Hydrophobic | Reference = 25 × SARscope; residual maximum 2.49 × 10⁻¹⁴ |
| Hydrogen-bond donor | Different spatial shape; scalar adjustment does not reconcile it |
| Hydrogen-bond acceptor | Different spatial shape; scalar adjustment does not reconcile it |

**Raw fields do not match the declared rtol/atol = 10⁻¹⁰.** These factors
were inferred *after* observing the arrays to distinguish sign/unit conventions
from spatial disagreement. They are diagnostics, not newly declared passing
criteria or independently sourced literature parameters. No factor was added
to SARscope production.

This is strong evidence of matching SEH spatial kernels under identical
inputs, not matching raw numerical conventions or the end-to-end workflow.
The hydrogen-bond difference is consistent with atom-centred versus directional
pseudoatom definitions. This does not test MMFF94, CoMFA, preparation,
preprocessing, component selection or prediction.

```bash
.venv/bin/python benchmarks/field_qsar/field_parity_probe.py \
  --python /path/to/reference-env/bin/python \
  --out benchmarks/field_qsar/field-parity-new-run
```

The probe retains numeric arrays, errors and a hash manifest, not upstream
structures or program code. The initial launcher failed because it imported
an unnecessary HTTP client into the comparator worker; the correction keeps
that import in the launcher. No numerical setting changed to fix it.

## Earlier SARscope comparison

`protocol.json` and `results-20261015-v4/` retain eight prespecified SARscope
runs: four settings × two activity sources. The three-field, block-scaled,
fixed-three-component model using Table 2 labels has LOO q² = 0.665968 and
held-out R² = 0.039756. This does **not** reproduce Python SEH q² = 0.609.
Proximity to classic SEH q² = 0.665 is not equivalence: the classic model
reports four components and different parameterisation. Native defaults
perform worse externally; column-scaled variants can yield extreme predictions.

The old V4 summary accidentally associates the five-field native setting
with three-field published q² (0.609 instead of 0.630). This is a metadata
error, not a changed model result. The runner is corrected; the original
artifacts and manifest remain unchanged. Earlier preparation failures are
retained and do not count as completed models. In V3, `completed_cases: 0`
takes precedence over its incorrect generic conclusion text.

## Controlled model comparison in SARscope

`model-parity-protocol.json` freezes a separate 21/10, fixed-three-component
LOO comparison using the existing SARscope regression engine for both matrix
sources. It does not replace the historical API experiment above.
The [model assessment](model-parity-20261016-v3/assessment.json) shows:

- Unconverted block-scaled matrices give similar, but not identical results:
  q² = 0.598152 (reference fields) versus 0.598538 (SARscope fields). Retained
  columns differ (15346 versus 15224) because the variance floor is in raw
  field units. The declared strict parity test correctly fails.
- A subsequently added diagnostic converts reference field conventions using
  factors measured in the preceding geometry-only comparison, not fitted to
  activities. Both raw and block-scaled models then give matching LOO/test
  predictions (maximum difference 1.78 × 10⁻¹⁵), with identical columns.
- The native block-scaled case still has test R² = 0.152207. Matching code
  behaviour does not make a model predictive. The raw control has higher
  q² = 0.674281 but test R² = −0.295452; no test-based winner is selected.

These post-hoc conversion cases are explicitly labelled diagnostics, not
prespecified reproduction successes. Both matrices go through the **same**
regression engine, so this is not independent PLS solver validation. No
conversion factors or preprocessing changes were inserted into production.

## Remaining acceptance gates

1. Reproduce the full production preparation/preprocessing path independently,
   not just the isolated field generator plus an external historical API.
2. Resolve raw sign/scale conventions and directional hydrogen-bond definitions
   against primary documentation before claiming raw or five-field equivalence.
3. Compare contours/coefficient conventions and extend to CoMFA separately.
4. Evaluate a second independently chosen dataset. Previously inspected
   steroid test compounds are a reproduction fixture, not fresh validation.

Methods 4.6 describes combining training and prediction fields before
preprocessing. Any historical emulation must be separately labelled;
SARscope's production training-only preprocessing is deliberately retained.
SEH agreement would not establish CoMFA or hydrogen-bond-field reproduction.

## Rerunning the isolated probe

From this checkout, create a **separate** Python 3.12 environment and install
`reference-environment.txt` into it. Do not overwrite SARscope's working
environment or add PyVista/VTK to its application requirements.

```bash
/path/to/reference-env/bin/python -m pip install -r benchmarks/field_qsar/reference-environment.txt
.venv/bin/python benchmarks/field_qsar/reference_probe.py \
  --python /path/to/reference-env/bin/python \
  --out benchmarks/field_qsar/reference-new-run
```

The probe checks source hashes, executes unmodified code with one-thread BLAS
and a 180-second timeout, retains versions/numerical CSVs, removes structure
strings from retained CSVs, and writes a SHA-256 manifest. Nonempty output
directories are refused. The optional offline tests verify identity checking,
metric accounting, hashes and failure behaviour, not predictive validity.

`reference-20261016/` records a failed invocation using the README's field
syntax. Executable `--help` instead requires `--fields SEH`; this correction
is not model tuning. Successful sanitised runs are `reference-20261016-v3/`
and `reference-20261016-v4/`; V4 also records the runner/protocol hashes and
UTC timestamp and returns identical scores. Their raw predecessor was moved
outside the repository to temporary storage
because it contained upstream structure strings. The README's all-dataset
`comsiatest.py` example was also attempted separately but exceeded 180 seconds;
it is not treated as a completed reproduction.

The external repository's LICENSE is CC BY-SA 4.0; the paper is CC BY 4.0.
No upstream code or SDF is bundled into SARscope. The original unsplit data
repository lacks a discovered redistribution licence; the author repository's
licence alone does not adjudicate all third-party data rights. Numerical
reference outputs are attributed to Haga et al.; no endorsement is implied.
Open3DQSAR/Open3DALIGN source was not consulted.

## Other candidate investigated

Ragno R. *www.3d-qsar.com: a web portal that brings 3-D QSAR to all electronic
devices—the Py-CoMFA web application as tool to build models from pre-aligned
datasets.* J Comput Aided Mol Des. 2019;33:855–864.
[doi:10.1007/s10822-019-00231-x](https://doi.org/10.1007/s10822-019-00231-x).

Its abstract reports 30 pre-aligned datasets. A directly executable reference
and intermediate numerical outputs were not established in this search;
the institutional preprint download returned HTTP 403. This access failure
is not evidence that data do not exist. Do not guess force-field parameters.

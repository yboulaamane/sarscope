# Generated-alignment implementation validation — 19 October 2026

**Additional steroid-series check:** the later
[published-benchmark repetition](steroid-reproduction-20261019.md) exposed
failures outside the EGFR cohort below: rigid MCS refuses the strict
whole-series core, and constrained embedding fails for androstanediol with
an unhelpful `Bad Conformer Id` error. The passing checks here are
cohort-specific, not evidence that those failures have been fixed.

**Both generated routes pass the tested implementation contracts.** This is
evidence for rigid MCS and core-constrained conformer generation, not an
assertion that every generated conformer is a biological binding pose.
Pre-aligned coordinates remain a separate input route. No receptor or docking
is invoked by either tested generated route.

## What was executed

The [offline runner](../benchmarks/validate_generated_alignment.py) freezes its
protocol, source hashes and input membership before executing tests. It runs
43 targeted tests, including 26 newly added generated-alignment checks, and
then aligns **all 50** bundled measured EGFR compounds with each route. It
uses the previously documented CHEMBL29197 reference, source order, seed 42,
eight constrained conformers and unchanged 30-second job/180-second series
budgets. Activities are not supplied to the aligner. There is no selection by
potency, model performance or overlay outcome, and no model is fitted.

Two Linux environments were run independently:

| Runtime | Rigid MCS | Constrained | Default rigid quality warnings | Constrained warnings |
|---|---|---|---:|---:|
| Python 3.12 / RDKit 2024.9.6 | 50/50 completed; checks pass | 50/50 completed; checks pass | 17 | 0 |
| Python 3.13 / RDKit 2026.09.1 | 50/50 completed; checks pass | 50/50 completed; checks pass | 18 | 0 |

These are 50 unique measured molecules, evaluated in two routes and two
environments, **not 200 independent chemical examples**. The same source cohort
was already available; this is an implementation audit, not a fresh predictive
benchmark. Source activities and chemistry are unchanged.

The complete non-network regression suite also passed: **653 passed, six
skipped and two deselected**, with no failures. Its
[JUnit record](../benchmarks/alignment/full-suite-20261019.xml) is retained
separately from the frozen route-run manifests.

## Independent checks

- A NumPy SVD/Kabsch oracle checks proper-rotation least-squares fits without
  calling RDKit's alignment implementation for the reference answer.
- Known rotations/translations, atom-order reversal and ring symmetry test
  mapping and fit recovery. Mirrored stereoisomers cannot gain a zero RMSD by
  reflection; aromatic and saturated rings cannot be forced into a shared core.
- Direct bond-length measurements, signed tetrahedral volumes and E/Z geometry
  supplement coordinate-derived stereochemical checks. Aromatic, chiral, E/Z,
  fused-ring and bridged-ring fixtures are included.
- Rigid generation preserves all interatomic distances to the 0.0002 Å
  serialization tolerance. Constrained generation retains mapped reference
  coordinates to that tolerance. The measured constrained fixed-core RMSD was
  zero at exported coordinate precision in both environments.
- Molecular identity, finite coordinates, mapping element/aromaticity/
  hybridisation compatibility, and the reported RMSD/coverage are checked on
  every measured output. Broad bond-length/radius ratios (0.6–1.45) are geometry
  sanity checks, not an energy or binding-pose certificate.
- Reruns after activity changes and record reordering preserve coordinates
  bitwise **within each tested environment**. Cross-RDKit-version conformer
  identity is not required or claimed; sampling differs between versions.
- Malformed, nonfinite, unspecified or conflicting stereo inputs are refused.
  A real deliberately nonresponding spawned worker is terminated at its budget;
  other tests cover startup/job/series attribution and cleanup. No molecule or
  stereochemical condition is silently skipped after a failure.

## Why rigid warnings are retained

The maximum core RMSD was 1.596700 Å in RDKit 2024.9.6 and 1.709262 Å in
RDKit 2026.09.1; minimum core coverage was 0.72 in both. The fixed default
RMSD threshold is 1.0 Å, hence the visible 17/18 warnings. Rigid superposition
cannot change an independently generated conformation to make its entire core
identical to the reference. Correctly calculating and reporting that mismatch
passes the implementation contract; it does **not** pass a model's alignment
quality gate. Neither threshold was relaxed and no warnings were cleared.

For pre-aligned input, biological justification of the supplied frame is the
user's methodological responsibility. SARscope is responsible for preserving
that frame and implementing its numerical checks correctly. The evidence here
replaces an open-ended "generated alignment has not been implementation-tested"
caveat with specific passing checks; it does not prove a bioactive conformation.

## Frozen artifacts and reproducibility

- [Python 3.12 assessment](../benchmarks/alignment/generated-20261019-py312/assessment.json)
- [Python 3.13 assessment](../benchmarks/alignment/generated-20261019-py313/assessment.json)
- Each directory contains the pre-execution protocol, source/input hashes,
  JUnit/log, all 50 generated structures per route, per-molecule coordinate
  audit, unchanged warnings and an output SHA-256 manifest.
- [Fixture development record](../benchmarks/alignment/README.md) retains the
  initial wrongly assigned bridged control and the reason it was corrected.

```bash
python benchmarks/validate_generated_alignment.py --out runs/alignment-check
```

Use a new output directory. Install the ordinary development dependencies plus
the `sarscope[docking]` preparation extra to exercise the normal route gate.
The new tests are also included in the existing two-version field CI job;
that workflow edit has not been represented as an already completed remote CI run.

The production aligner did not require a numerical fix in this audit. The
tested source is identified by file hashes in each protocol; the added tests,
runner and evidence are separate from the older public commit until published.

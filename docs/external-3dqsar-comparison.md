# External alignment and PLS comparison — 19 October 2026

Follow-up: [native Open3DQSAR field-generation performance](open3dqsar-native-performance.md)
now evaluates its own MMFF fields, rather than only importing SARscope descriptors.

## Outcome

SARscope's fitted PLS predictions agree with **standalone Open3DQSAR 2.3** on
both original-source thrombin and acetylcholinesterase (AChE) cohorts when both
programs receive the same training-filtered/scaled descriptors, activities,
partitions and component count. Maximum absolute differences over **all 199
training and test predictions** are below 0.000050 activity units, consistent
with Open3DQSAR's four-decimal SDF output.

**Standalone Open3DALIGN 2.3 also ran successfully**, aligning all 50 EGFR,
88 thrombin and 111 AChE compounds. However, its single-conformer realignment
did not improve held-out predictive R² over the supplied historical alignments.
RDKit's MMFF- and Crippen-based O3A routes did not improve it either. These
results support retaining reviewed, supplied alignments and considering O3A as
an optional hypothesis—not replacing the app's routes automatically.

This is **PLS solver agreement and retrospective alignment sensitivity**, not
equivalence of SARscope's fields to Open3DQSAR, universal model validation,
experimental binding-pose confirmation, or prospective predictive validation.
No app default, scientific threshold, manuscript claim or deployment dependency
was changed by this experiment.

## Fixed comparison

Protocols were saved before their respective runs. Original data hashes,
membership, reference identities, settings, output hashes and failed import
attempts are retained. No compounds were removed or stereochemical constraints
relaxed. No reference or alignment route was selected using held-out scores.

- EGFR: all 50 measured compounds from the bundled single-assay example;
  reference CHEMBL29197, chosen in the existing example before this comparison.
  All generated starting conformers used canonical isomeric SMILES and the
  production generator with seed 42. Both RDKit routes and standalone
  Open3DALIGN used the same conformations; only rigid-body orientation changed.
- Thrombin: all 88 original-source compounds, historical 59/29 partition;
  reference compound 42, first member of the fixed training order.
- AChE: all 111 original-source compounds, historical 74/37 partition;
  reference 1-1, first member of the fixed training order.
- Model sensitivity: SARscope's five Gaussian fields, Gasteiger field charges,
  1 Å lattice, 4 Å margin, training-only screening and block scaling, nested
  three-fold modelling, component cap four and n/5 cap. Exact cohort settings
  are in the frozen protocols. The same training-derived grid construction rule
  was used throughout; changed molecular orientations can change grid bounds.
- MMFF properties and Crippen contributions were used **only for alignment**;
  they did not replace the field charge model.
- Permutations were set to zero for this bounded sensitivity experiment. The
  original permutation evidence is not replaced and no new validation status
  is granted. These historical cohorts had already been inspected previously;
  this is not a new independent held-out benchmark.

## Predictive sensitivity

Every route is reported, including inferior scores. “Original” means the
supplied historical alignment, **not** a newly generated SARscope MCS alignment.
The original baseline predictions reproduced the previous results to numerical
precision before realigned models were interpreted.

| Cohort | Alignment | Nested pooled q² | Held-out R² | Held-out RMSE |
|---|---|---:|---:|---:|
| Thrombin | Original supplied coordinates | 0.6071 | 0.5072 | 0.7716 |
| Thrombin | RDKit O3A–MMFF | 0.5485 | 0.4376 | 0.8243 |
| Thrombin | RDKit O3A–Crippen | 0.5493 | 0.4234 | 0.8346 |
| Thrombin | Standalone Open3DALIGN, atom-based | 0.5470 | 0.4406 | 0.8221 |
| AChE | Original supplied coordinates | 0.0519 | 0.4890 | 0.9223 |
| AChE | RDKit O3A–MMFF | 0.2198 | 0.4221 | 0.9809 |
| AChE | RDKit O3A–Crippen | 0.1653 | 0.4327 | 0.9718 |
| AChE | Standalone Open3DALIGN, atom-based | 0.1670 | 0.4794 | 0.9310 |

The 2D random-forest comparator's predictions stayed unchanged to <4×10⁻¹⁵
across all geometry changes, providing a control for identical graphs and
partitions. The AChE training-CV improvement did **not** translate into a
held-out improvement. No significance or general superiority is inferred from
these two cohorts.

## Geometry checks

Both RDKit O3A variants processed all 50 EGFR structures. Molecular identities,
assigned coordinate-derived stereochemistry and internal interatomic distances
were preserved; maximum internal-distance change was 5.4×10⁻¹⁵ Å.

All 249 standalone Open3DALIGN outputs passed the corresponding graph, stereo
and rigid-distance checks. Maximum distance changes were <0.000152 Å,
consistent with SDF coordinate rounding. EGFR common-core RMSD medians were
0.573 Å (RDKit MMFF), 0.580 Å (RDKit Crippen), and 0.520 Å (standalone
Open3DALIGN); maxima were 2.825, 2.922 and 2.803 Å, respectively.

These are geometry-preservation checks, not proof of a correct bioactive
alignment. Property-based overlays can intentionally differ from literal core
overlays. The earlier generated-MCS validation is separate; constrained-core
RMSD is enforced by construction and must not be used to declare a winner.

## Independent PLS check

The solver comparison used the original coordinates. SARscope calculated its
fields and fitted its descriptor filter/scaling on training data only. The
resulting 2000-column matrix was passed to Open3DQSAR without further field
selection or scaling. Both solvers used the same three components, already
selected using the original training procedure. Test compounds were explicitly
excluded from fitting. Predictions were compared compound by compound, not
only by a rounded correlation coefficient.

The columns were indexed by an artificial 10×10×20 single-field grid solely
for matrix interchange; it is not a physical molecular interaction field.
This check does not compare field equations, contour maps or nested-CV
implementations.

| Cohort | Predictions checked | Maximum absolute difference | Prespecified tolerance | Result |
|---|---:|---:|---:|---|
| Thrombin | 88 | 0.00004994 | 0.0005 | Pass |
| AChE | 111 | 0.00004972 | 0.0005 | Pass |

Open3DQSAR prints an `r2(pred)` statistic with a denominator different from
sklearn's test-centred R². We therefore calculated both tools' R² and RMSE
from their exported predictions using the **same definitions**. Thrombin
test R² was 0.507182 versus 0.507179; AChE was 0.489004 versus 0.489009.
Different printed metric names alone must not be treated as a solver failure.

### Import failures retained, not hidden

The first AChE `FREE_FORMAT` import produced a maximum prediction difference
of 0.04848. A full descriptor round trip located exactly one altered term:
the first value, 0.000003636955993897112, came back as 8.456003. The remaining
terms agreed within ASCII export precision after accounting for axis order.
Decimal notation and shorter lines did not resolve it. This observed mismatch
is documented; its internal cause was not established by inspecting source.

A lossless OpenDX export resolved the comparison. The old reader required
printf-style numbered filenames and conventional three-value data lines;
the rejected wildcard and one-value-per-line attempts were retained too.
The final 222,000-term AChE round trip had no differences >0.0001, with a
maximum of 0.000000705 from limited ASCII export precision. All original
float64 descriptor values, activities, partitions and modelling settings were
unchanged. No descriptor value or prediction was patched to obtain agreement.

## Runtime, provenance and tests

- SARscope: Python 3.13.13 / RDKit 2026.09.1, existing project environment.
- External programs: official SourceForge Linux Open3DALIGN/Open3DQSAR 2.3
  binaries; hashes recorded. Executed as standalone processes, one CPU each.
- The bundled 32-bit OpenBabel would not run directly on this host. Isolated
  conda OpenBabel 3.1.1 was used with launch aliases for legacy executable names
  (the binary reports 3.1.0). No system installation or app dependency changed.
- Only public API/user manuals and sample command files were consulted;
  no GPL algorithm source was read, copied or ported into SARscope.
- Targeted geometry/conformer tests: **50 passed**.
- Full non-network suite: **660 passed, 6 skipped, 2 deselected**, with two
  existing small-class CV warnings. JUnit evidence is retained.

Compact tables, protocols and integrity records:
[external-comparison-20261019](../benchmarks/alignment/external-comparison-20261019/).
Full local per-compound structures, fields, predictions and failed-run logs:
`runs/o3a-comparison-20261019/` (ignored by Git).

Primary structures and their derivatives retain their separate licences:
ChEMBL EGFR **CC-BY-SA-3.0**; publisher thrombin/AChE source **CC-BY-NC-4.0**.
They are not relicensed as SARscope's code. External GPL binaries were not
added to the repository.

## References and next use

- [RDKit public molecular-alignment API](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolAlign.html).
- [Open3DALIGN paper](https://pubmed.ncbi.nlm.nih.gov/21792629/) and
  [official binary distribution](https://sourceforge.net/projects/open3dalign/files/binaries/linux/).
- [Open3DQSAR paper](https://doi.org/10.1007/s00894-010-0684-x) and
  [official binary distribution](https://sourceforge.net/projects/open3dqsar/files/binaries/linux/).
- Primary historical cohorts: [Sutherland et al., DOI 10.1021/jm0497141](https://doi.org/10.1021/jm0497141),
  supporting archive SHA256 `e69edbb64048a243db5bce8700b198ce9a60b6f8fe5c87f30d863d1f99790ecf`.
- EGFR example: [Bridges et al., DOI 10.1021/jm9503613](https://doi.org/10.1021/jm9503613).

The defensible use is to add Open3DALIGN as an **optional external alignment
source** in a future change, preserving user review and provenance. This
experiment does not justify an automatic switch or the claim that all native
3D-QSAR field calculations are independently reproduced by Open3DQSAR.

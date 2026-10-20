# SARscope-native MMFF94 molecular-interaction fields

Added 19 October 2026. This is an **independent implementation of published
equations using RDKit's public MMFF94 parameter API**, not a port of GPL
Open3DQSAR/OpenBabel source. It needs no external executable, OpenBabel,
additional package, network request or proprietary licence at runtime.

In Streamlit, prepare/review the alignment, then choose **Field definition →
SARscope MMFF94 · buffered 14-7 and Coulomb energy fields**. The existing
Gaussian and Tripos-reference choices remain available; the default was not
silently changed. Users still choose their grid, included fields, validation,
component limit and permutations explicitly.

## Method and scope

For each molecular atom and a hypothetical MMFF type-1 alkyl-carbon probe,
RDKit supplies the mixed pair distance R* and well depth ε. SARscope evaluates
the published buffered 14-7 van der Waals interaction:

```text
E_vdw(r) = ε [1.07 R* / (r + 0.07 R*)]^7
             × [1.12 R*^7 / (r^7 + 0.12 R*^7) − 2]
```

Electrostatics use an independent +1e point probe, MMFF94 partial charges and
the published **0.05 Å distance buffer**:

```text
E_electrostatic = (332.0716 / D) Σ_i q_i / (r_i + 0.05)
```

Both sums include all explicit atoms, including hydrogens. Distances are Å;
fields are kcal/mol. This initial profile supports constant dielectric only.
It is not a full molecular-mechanics engine: no bonded terms, intermolecular
probe hydrogens or intramolecular 1–4 scaling enter the probe-field sum.

Summed steric and electrostatic energies are bounded **independently**, at
±30 kcal/mol by default. Unlike the older LJ-style profile, steric plateaus
do not make electrostatic descriptors NaN. There is no hidden mean filling,
charge fallback or force-field substitution; unsupported typing fails with
an explicit error. User molecules and supplied hydrogen coordinates are
preserved. Calculation remains chunked and respects existing workspace limits.

The internal `method="comfa"` value selects two energy-field blocks in the
existing pipeline; `field_definition="mmff94_mif"` distinguishes the actual
MMFF equations. This is **not Tripos CoMFA or CoMSIA**, nor a claim to reproduce
all Open3DQSAR preprocessing, BUW, variable masks or CV options. Contours are
labelled as MMFF probe-energy associations, not instructions to add bulk.

Parameters, field convention, RDKit version and per-molecule charge/atom-type
audit are retained in normal SARscope results. Nested validation, design
prediction, domain analysis, contours and downloads reuse the existing
pipeline; no separate regression implementation was added.
The existing charge/grid/pose training-stability UI remains Gaussian-only;
this energy profile does not silently enable an untested stability workflow.

## Numerical comparison against standalone Open3DQSAR

The external program's native outputs were generated and frozen before this
implementation. The comparison uses all 88 thrombin and 111 AChE compounds,
their original historical partitions, and both supplied and fixed-reference
Open3DALIGN frames. All four cases are reported. No compound, descriptor
outlier, stereochemical condition or inferior model was removed.

Across approximately **12.6 million field terms**, maximum absolute
differences from the external ASCII exports were below **0.000006 kcal/mol**
for both fields. This is consistent with their printed precision.

Both matrices then passed through the same existing SARscope nested-PLS
workflow, with the same partition, four-component cap, n/5 constraint,
training-fold screening and block scaling. This is not a comparison between
native Open3DQSAR's LOO procedure and SARscope's nested CV.

| Cohort/alignment | Own nested q² | External-field nested q² | Own test R² | External-field test R² |
|---|---:|---:|---:|---:|
| Thrombin/original | 0.61215945 | 0.61215945 | 0.62982987 | 0.62982987 |
| Thrombin/Open3DALIGN | 0.32949077 | 0.32949077 | 0.59418288 | 0.59418288 |
| AChE/original | 0.48283626 | 0.48283625 | 0.74546542 | 0.74546542 |
| AChE/Open3DALIGN | 0.48904243 | 0.48904243 | 0.65441718 | 0.65441717 |

For original alignments, own held-out RMSE was 0.66870311 (thrombin) and
0.65095473 (AChE). The same profile and settings were used throughout, not
selected per target using held-out labels. These historical sets had already
been inspected: agreement demonstrates implementation reproduction on these
cases, not a fresh independent predictive benchmark. No new permutations were
run in this bounded diagnostic. The app's ordinary validation gates remain
unchanged; AChE's nested q² remains below 0.5.

### Discrepancies discovered and retained

The first implementation used an unbuffered 1/r point probe based on the
external manual's schematic equation. Steric grids agreed, but electrostatics
did not. A descriptor-only diagnostic compared the two documented forms,
unbuffered and the original MMFF **fixed 0.05 Å** buffer; the latter reproduced
the grids. No fitted buffer search or test-score optimisation was performed.
The unbuffered results and pre-correction source snapshots remain under
`runs/mmff-mif-independent-20261019/`.

The public RDKit experimental MIF docstring says kJ/mol, but its installed
MMFF VDW evaluator numerically follows its kcal/mol pair parameters without
an additional 4.184 factor. An initial test-only unit conversion was removed
after checking the analytic well depth and direct API outputs. The production
energy equation was not rescaled to fit a benchmark.

Final evidence is from `runs/mmff-mif-independent-release-20261019/`.
Compact protocols, per-case results, source hashes and test records are in
[mmff-mif-independent-20261019](../benchmarks/field_qsar/mmff-mif-independent-20261019/).
The archive and raw structural derivatives retain the publisher's separate
CC-BY-NC-4.0 data terms. No GPL executable or algorithm source is bundled.

## Tests and reproducibility

The new tests cover the analytic VDW well depth and asymptote, finite origin,
exact zero-charge electrostatics, charge inversion, dielectric scaling,
chunk-size independence, no input mutation and explicit refusal of inconsistent
profiles. A separate public RDKit MIF evaluator checks neutral, aromatic,
charged and fluorinated molecules. All 13 field tests passed under both
Python 3.13/RDKit 2026.09.1 and Python 3.12/RDKit 2024.09.6.

The Streamlit test selects the new profile and verifies compatible fields,
charge model, scaling, electrostatic plateau and absence of inapplicable
missing-value controls. Existing profiles retain their previous options.
The final serial non-network suite passed: **679 passed, six skipped, two
deselected**, with two existing small-class CV warnings. A concurrent run's
five setup errors all came from one existing demo fixture reaching its
unchanged five-second MCS limit; that record and the successful serial rerun
are both retained. No alignment timeout or scientific threshold was relaxed.

```bash
python -m benchmarks.validate_mmff_mif prepare --out runs/new-mmff-check
python -m benchmarks.validate_mmff_mif run --out runs/new-mmff-check
```

This offline reproduction requires the locally retained, frozen native
Open3DQSAR outputs; the ordinary app does not. The runner freezes source and
input hashes before comparing grids or fitting models.

## Public scientific and API references

- [Halgren, MMFF94 intermolecular VDW/electrostatic parameters (1996)](https://doi.org/10.1002/%28SICI%291096-987X%28199604%2917%3A5/6%3C520%3A%3AAID-JCC2%3E3.0.CO%3B2-W).
- [Public MMFF functional forms](https://www.charmm-gui.org/?doc=docs&file=mmff).
- [RDKit's public MMFF parameter API](https://www.rdkit.org/docs/source/rdkit.ForceField.rdForceField.html).
- [RDKit molecular-interaction-field API](https://www.rdkit.org/docs/source/rdkit.Chem.rdMIF.html).
- [Open3DQSAR native workflow comparison](open3dqsar-native-performance.md).

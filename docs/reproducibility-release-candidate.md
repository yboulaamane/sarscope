# Paper reproducibility release candidate

Version **0.2.0rc1** is a local, unpublished candidate, not a public release or
archival deposit. It is based on public `main` at `a79be6a`, without rewriting
the original diverged checkout. It includes the independent MMFF94 field module,
its UI integration, the bounded conformer-ID correction, comparison drivers,
compact positive/negative evidence, tests and a scalar-count audit.

## Historical provenance

The earlier field baseline is publicly reachable at
`55dfeefb8baefcadf6f87f7e9a5b3a78c74d662b`. It has exactly the same tree as the
pre-rewrite identifier in the historical execution logs. The corresponding SAR
evaluation base has also been mapped by tree. See
`benchmarks/release-20261020/commit-provenance.json`. Original protocols, results
and their manifests have not been rewritten to look like new executions.

## Install and check

Use a new environment, not the running Streamlit deployment. For the tested
Python 3.13 numerical stack:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -e '.[dev,app]' \
  -c benchmarks/field_qsar/compatibility-constraints-py313.txt
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -m 'not network'
```

Python 3.12 has a separate constraint file. These numerical constraints are not
a universal transitive lock. Optional docking, SHAP and representation libraries
have their own dependencies; skipped integrations must not be counted as tested.
The candidate check record is in `benchmarks/release-20261020/`.

## Repeat the recorded measured-data example

Obtain the original publisher ZIP from
<https://ndownloader.figshare.com/files/5158327>, subject to the publisher's
CC BY-NC 4.0 terms. The preparation routine verifies its hash and preserves
the original 88 thrombin records and 59/29 partition. In a **new** output folder:

```bash
python -m benchmarks.validate_end_to_end_demo prepare \
  --archive /absolute/path/to/primary-data.zip --out runs/my-reproduction
python -m benchmarks.validate_end_to_end_demo run --out runs/my-reproduction
```

The driver freezes the current sources and inputs before execution, uses 99
training-label permutations and retains the declared unsuccessful CCR5 route.
It does not replace the original recorded evidence. Preparation and full fitting
are separate actions; the run may take minutes. No raw primary SD file, external
GPL algorithm source or third-party binary is distributed in this candidate.
The underlying Gaussian and MMFF94 numeric modules are unchanged by release
formatting; original frozen code snapshots remain the authority for historical
source hashes. Reformatted comparison drivers have new file hashes and require
a new protocol freeze, not reuse of an old frozen source-hash assertion.

## Independent scalar count

`benchmarks/field_qsar/mmff-mif-independent-20261019/scalar-count-audit.json`
records **12,607,200** compared scalar entries, counted once per entry across
four molecule × field-grid matrices. It checks the raw-array hashes against the
original protocol, then records matrix/grid dimensions and individual counts.
It is a post-hoc provenance audit, not a new field calculation or validation run.

With the original, separately licensed native arrays available:

```bash
python -m benchmarks.audit_mmff_field_counts \
  --arrays-root /absolute/path/to/open3dqsar-native-20261019 \
  --out runs/my-scalar-count-audit.json
```

The offline test suite independently checks the recorded total against frozen
protocol dimensions without bundling the raw arrays. Neither `results.csv` nor
the old manifests were changed to insert the count.

## Before submission

Review and publish the exact candidate revision, assign an appropriate release
tag, and archive the permitted code/evidence under a persistent identifier.
Update manuscript availability statements only after those records exist.
No push, public tag, DOI publication or journal submission is implied here.

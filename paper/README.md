# SARscope publication package

## Journal-tailored revision — 13 October 2026

The current target is **Journal of Cheminformatics, Software article**:

- [Journal-tailored manuscript](jcheminformatics-manuscript.md),
  [Word](SARscope_JCheminformatics_manuscript.docx) and
  [PDF](SARscope_JCheminformatics_manuscript.pdf).
- [Updated supplement](jcheminformatics-supplement.md),
  [Word](SARscope_JCheminformatics_supplement.docx) and
  [PDF](SARscope_JCheminformatics_supplement.pdf).
- [Journal fit and validation status](journal-target-and-validation.md).

The new revision includes the supplied affiliation, an approximately 800-word
introduction and 32 numbered references. The bibliography follows Springer Basic
numeric/bracket conventions and is generated from `references.csl.json`.
Import [RIS](reference-exports/references.ris) or
[BibTeX](reference-exports/references.bib) into Zotero/EndNote. Full author lists
are retained in the import files; Word citations link internally to reference
entries but are not live Zotero fields. See [the reference review](reference-review.md).
The offline current-workbench audit
reproduces all 263 scoped pairs; among 17 sign-stable repeated-comparison effects,
15 retain the full directional criterion in tested nonempty subsets. One assay
and publication per case means their omissions leave no evidence, not proof of
cross-source robustness. These cases remain locally verified, not independently
scientifically validated or submission-ready.

Reproduce the new implementation audit in a fresh output directory:

```bash
python paper/audit_current_workbench.py --out runs/workbench-check
```

Inputs, source modules and output CSV hashes are in `workbench-audit/provenance.json`.
The new journal Word/PDF package has separate layout/citation/hash QA in
`journal-package-verification.json`. Earlier manuscripts and original exports
are preserved below; they are not silently replaced by this revision.

This is a working software-paper package, **not a submission-ready manuscript**.
The intended contribution is interactive, assay-aware SAR interpretation with
traceable source evidence, not another model leaderboard or a docking mechanism.

Start with the revised [scientific manuscript](scientific-manuscript.md),
[Supplementary Information](supplementary-scientific-methods.md) and
[author notes](author-notes-scientific.md). The original manuscript and Word
document are preserved for comparison. The revised DRD2 analysis uses 23
stereochemically consistent compounds and 36 pairs, superseding the original
26-compound/46-pair positional results.

The revised [scientific manuscript](scientific-manuscript.md) and
[Supplementary Information](supplementary-scientific-methods.md) add positional
interpretation, a stereochemical-core audit and robustness diagnostics.
[Author notes](author-notes-scientific.md) explain the changed DRD2 counts and
remaining submission gates. The saved scientific documents describe the original
evaluation-side stereochemical restriction. A subsequent local implementation
now retains per-molecule mapped cores, separates core states in the browser and
scopes CLI maps to the largest state. Existing saved results and Word/PDF files
have not been regenerated or publicly released with that code change.

To reproduce the new analyses without downloads, docking or model training:

```bash
python paper/analyse_sar_evidence.py --results paper/results --out runs/scientific-paper-check
```

## What has actually been done

- Frozen, source-attributed case cohorts from three published papers: BRAF,
  DRD2 and an ER-alpha counter-assay from an ER-beta selectivity paper.
- Offline curation, largest-series decomposition, interactive-map SVGs,
  position-level comparisons and complete measurement exports.
- Exhaustive enumeration verifies all 273 position-pair rows across the three
  selected series. This is numerical verification, not independent SAR validation.
- An actual mmpdb 3.1.4 single-cut coverage comparison on the same curated inputs.
  All SARscope unique MMP pairs were also found by mmpdb; mmpdb found 56 additional
  unique pairs, all involving hydrogen substitutions in these cohorts.
- A scaffold-cluster bootstrap of the existing three-family held-out predictions.
  This reuses saved predictions: no retraining or new independent benchmark.
- A 41-task grade-stratified review pack, currently **unreviewed**.
- Fresh isolated-environment reproduction of all 26 case CSV outputs (excluding
  timing), and a hashed Linux/Python 3.13 runtime lock.

See `results/case-summary.csv`, `mmpdb-comparison/comparison.csv` and
`benchmark-audit/bootstrap-summary.csv`. Cold local case-analysis timings are
single-run observations, not a hosted-app or competitor speed claim.

## Reproduce the small published-data examples

From the repository root, in a Python 3.13 environment:

```bash
python -m pip install --require-hashes -r paper/requirements-linux-py313.lock
python -m pip install --no-deps -e .
python -m sarscope.publication evaluate --cases paper/cases --out runs/paper-check
```

The hashed lock is specific to Linux x86_64/Python 3.13. On other platforms,
start from `requirements-analysis.txt`, resolve compatible wheels and record the
result; those environments have not been tested here. This is an analysis runtime
lock, not a hosted-app/docking/comparator environment lock.

Outputs must be new/empty directories. The evaluator never accesses the network,
trains a model or runs docking. Source files are hash-checked before analysis.
Compare CSV contents excluding `elapsed_seconds`; timings and SVG metadata can
vary between runs. Algorithmic outputs can change with RDKit versions.

The paper cases are **not shipped as a Streamlit demo** and do not affect app
startup. To examine a case in the app, upload `results/<case>/molecules.csv` using
the SMILES, molecule_id and pactivity columns. That upload does not carry the
original assay evidence; use the offline case exports for the publication analysis.

## Reproduce the optional comparator

Use a separate environment so mmpdb cannot change hosted-app dependencies:

```bash
python -m pip install -r paper/requirements-comparison.txt
python -m pip install --no-deps -e .
python -m sarscope.publication compare-mmpdb --results paper/results --out runs/mmpdb-check
```

Commands, inputs, full pair lists and logs are saved. Compare compound-pair
membership, not attachment-point labels. mmpdb has capabilities outside this
single-cut experiment; this is not evidence SARscope is algorithmically superior.

## Existing QSAR audit and full source snapshot

The full frozen benchmark remains in local ignored `runs/benchmark-chembl37/`.
It is approximately 40 MB uncompressed and is **not in the Git repository**.
The local archive command below includes it explicitly, subject to ChEMBL's
separate data license. A live `benchmark-freeze` call would obtain current data,
not recreate ChEMBL 37 byte-for-byte; do not call that historical reproduction.

```bash
python -m sarscope.publication benchmark-audit \
  --predictions runs/benchmark-chembl37/results/held_out_predictions.csv \
  --out runs/bootstrap-check
python -m sarscope.publication bundle --root . \
  --benchmark runs/benchmark-chembl37 \
  --out /tmp/sarscope-publication.zip
```

The archive contains source code, tests, manuscript, small case snapshots and,
when requested, the full frozen benchmark and saved predictions. Each file has
a SHA-256 entry. It records the Git base and whether uncommitted files were
included. **A local archive is not a published release or a DOI deposit.**
After deposition, the full benchmark would be accessible at
`reproducibility/benchmark/` inside the archive; use its `frozen/` and `results/`
paths to reproduce/audit the original benchmark.

## Human review

Use [the protocol](evaluation-protocol.md) to distribute `review-tasks.csv`,
`review-pairs.csv`, per-case structures/measurements and `review-template.csv`.
Do **not** send reviewers the answer key, annotated maps or transformation
summaries before they submit their independent decisions.

```bash
python -m sarscope.publication score-reviews \
  --key paper/results/review-answer-key.csv \
  --reviews completed_reviews.csv --out runs/review-agreement
```

At least two different reviewers must agree on a task to produce a consensus.
Blank templates fail instead of producing invented validation results. Missing
tasks, single reviews and disagreements remain unresolved and are reported.

## Licenses

Code is MIT. ChEMBL-derived snapshots and numerical/chemical evidence in `cases/`,
`results/`, `benchmark-audit/` and `mmpdb-comparison/` are CC BY-SA 3.0: see
[DATA-LICENSE.md](DATA-LICENSE.md). Paper prose/protocol/graphics are CC BY 4.0.
Original publication text and figures are not reproduced or relicensed here.
An archive containing data is not entirely MIT.

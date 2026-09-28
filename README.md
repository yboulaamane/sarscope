# SARscope

**Target ID in, structure–activity report out.**

ChEMBL will give you a target's bioactivity records. It will not curate them,
tell you which scaffolds are enriched, find the activity cliffs, or build you a
model that is honestly validated. SARscope does that, from one command, for any
target.

```bash
sarscope run CHEMBL5145 --out braf_report/
```

11,017 records in, a report folder out: curation log, property profile, scaffold
enrichment, R-group SAR tables, activity cliffs, and a model bake-off — with
every structure drawn and every setting recorded.

## Why this exists

The analysis is not hard. It is just long, and the parts that are easy to get
wrong are invisible in the result.

- **Curation is where the dataset is decided, and it is usually undocumented.**
  Whether you keep censored values, pool mutant with wild-type assays, or trust
  ChEMBL's duplicate flag moves the BRAF dataset between roughly 2,900 and 6,700
  molecules. SARscope makes each of those an explicit setting and counts every
  record that leaves, under the step that removed it.

- **The usual validation order leaks, and the leak flatters you.** Oversampling
  before the train/test split puts copies of training molecules in the test set;
  selecting features on all the data picks them using the held-out rows. On
  labels with no signal at all, that ordering scores Extra Trees at MCC 0.99
  where the correct ordering scores 0.00. SARscope resamples inside training
  folds only — and will run the leaky order alongside it on request, so the gap
  is a number for *your* dataset rather than a warning in a README.

- **Random splits are too easy on congeneric series.** Scaffold splitting is the
  default, so a series cannot sit on both sides of the split.

- **A ranked list is not an insight.** The report draws the structures: the
  enriched scaffolds, the two molecules of each activity cliff side by side, and
  the substituents that move potency at each position.

## Install

```bash
pip install sarscope
```

Or from source:

```bash
git clone https://github.com/yboulaamane/sarscope.git
cd sarscope && make install
```

PubChem fingerprints need the extra: `pip install "sarscope[pubchem]"`. It
depends on scikit-fingerprints, which caps the RDKit version, so the install can
downgrade RDKit.

## Using it

```bash
sarscope target CHEMBL5145                    # confirm the ID is the protein you think
sarscope fetch  CHEMBL5145                    # download (cached) and summarise raw records
sarscope run    CHEMBL5145 --out report/      # curate, analyse, write the report
sarscope run --input my_data.csv --out report/  # your own SMILES + potency table
sarscope compare CHEMBL5145 CHEMBL279
sarscope predict --model report/ --input new_compounds.csv
```

Always `target` first. A wrong ID does not fail — it quietly fetches a different
protein, and every number after that is about that protein.

`fetch` shows the distribution of every field curation acts on, so you can see
what you are about to filter:

```text
CHEMBL5145  Serine/threonine-protein kinase B-raf  (Homo sapiens, SINGLE PROTEIN)  [ChEMBL_37]
11017 records, 6703 molecules

  standard_relation        =: 9115, <: 1008, >: 554, None: 172, <=: 114, >=: 54
  assay_variant_mutation   None: 6523, V600E: 4494
  potential_duplicate      0: 8581, 1: 2436
  bao_label                single protein format: 7437, cell-based format: 2011, ...
```

In the browser:

```bash
streamlit run streamlit_app.py
```

The browser workflow is deliberately staged rather than automatic: check the
target, curate it, then run chemical space, scaffolds/SAR, activity cliffs, ML
and descriptor explanation only when requested. The ML stage includes a small
MLP alongside the tree and neighbour models. The explanation stage uses named
RDKit physicochemical descriptors and reports held-out permutation importance;
Random Forest also reports impurity importance. Optional TreeSHAP support is
available locally with `pip install "sarscope[explain]"` and is lazy-loaded so
the hosted app does not inherit SHAP's startup cost.

See [`docs/deploying.md`](docs/deploying.md) to host it.

## What you get

| Section | Output |
|---|---|
| Overview | Headline numbers, most potent molecules and enriched scaffolds, drawn |
| Curation | Every record removed and the step that removed it |
| Properties | Six descriptors by activity group, Mann-Whitney tests, PCA |
| Scaffolds | Murcko diversity, enrichment factors with Wilson lower bounds |
| R-group SAR | Per-position substituent effects, with cores and substituents drawn |
| Matched pairs | Single-cut transformations without requiring a preselected scaffold |
| Landscape | Activity cliffs with both structures, SALI, cliff generators |
| Models | Classification and continuous pActivity regression, scaffold/time split, leakage audit, applicability domain |
| Model diagnostics | Held-out error on activity-cliff compounds versus the rest |

```text
report/
  report.html        self-contained: figures and structures inlined
  model.joblib       selected regressor, refitted on the complete curated dataset
  provenance.json    ChEMBL release, package versions, every parameter
  tables/            every table as CSV
  figures/           every figure as PNG
```

`provenance.json` is the point of the whole thing: any number in the report can
be traced back to the ChEMBL release, the package versions and the exact
settings that produced it.

## Using your own data

A CSV with an ID, a SMILES and a potency column. Values may already be on the
p-scale, or given as a concentration:

```bash
sarscope run --input actives.csv --out report/ \
  --input-value-col IC50_nM --input-unit nM
```

Everything after curation is identical, so the same report comes out.

If the table contains a document year, chronological validation trains only on
the earlier compounds and reserves what came later:

```bash
sarscope run --input actives.csv --input-year-col year --split time \
  --time-cutoff 2019 --out report/
```

ChEMBL runs retain each molecule's earliest document year automatically. To
pool the common potency endpoints explicitly, use `--pool-types`; every value
is converted to `-log10(molar)` before aggregation and the pooled types are
recorded in the curation log and provenance. Pooling different endpoint types
is a modelling choice, not an assertion that IC50, Ki, Kd and EC50 are
experimentally interchangeable.

The saved regression model can score a new CSV containing `molecule_id` and
`smiles`:

```bash
sarscope predict --model report/ --input new_compounds.csv
```

The output includes predicted pActivity, maximum fingerprint similarity to the
training set, and an applicability-domain flag. `model.joblib` is a Python
pickle-based artifact; load only reports you trust.

## Development

```bash
make check                    # ruff, mypy, and the test suite
pytest -m network             # live ChEMBL checks (excluded from CI)
```

Standardisation, scaffolds and ECFP come from
[Sorbent](https://github.com/yboulaamane/sorbent). The descriptors deliberately
do not — see `analysis/descriptors.py` for why.

## License

MIT. ChEMBL data is CC BY-SA 3.0; cite ChEMBL when you publish results built on
it.

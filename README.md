# SARscope

**Target ID in, structure–activity report out.** SARscope pulls a target's
bioactivity data from ChEMBL, curates it with every decision recorded, and runs
the standard analyses of a chemical-space paper: descriptor profiles, scaffold
diversity and enrichment, activity cliffs, and a QSAR model bake-off. It
produces a report folder you can open, email, or attach as supplementary data.

> **Status: working.** The full pipeline runs end to end. Verified on BRAF
> (CHEMBL5145): 11,017 ChEMBL records curated to 2,917 molecules, then every
> analysis below, in about three minutes.

## Why

The workflow reproduces Aouidate, *J. Mol. Liq.* 401 (2024) 124705, a BRAF
study. None of its steps are specific to BRAF, so any ChEMBL target gets the
same report. Following the paper step by step turned up three places where a
tool should do better than a manual run:

- **Curation is explicit.** On BRAF, choices the paper leaves unstated
  (censored values, V600E vs wild-type assays, duplicate flags, assay format)
  move the dataset between roughly 3,400 and 6,650 molecules. Each choice here
  is a named parameter, and each record that is dropped is counted under the
  step that dropped it.
- **Validation cannot leak.** Oversampling before the train/test split puts
  copies of training molecules in the test set. Measured on BRAF, that order
  inflates random-forest test accuracy from 0.723 to 0.877; on random labels it
  turns MCC 0.00 into 0.99. SARscope resamples inside training folds only and
  runs both orders side by side, so the gap is a number in the report rather
  than an assertion.
- **Splits are by scaffold by default.** Random splits scatter a congeneric
  series across train and test.

[`docs/reproduction-notes.md`](docs/reproduction-notes.md) records what was
checked against the paper and how.

## Install

```bash
git clone https://github.com/yboulaamane/sarscope.git
cd sarscope
make install    # .venv with sarscope[dev]
```

PubChem fingerprints (the paper's best model) need `pip install -e ".[pubchem]"`.
That extra depends on scikit-fingerprints, which caps the RDKit version, so the
install can downgrade RDKit.

## Using it

```bash
sarscope target CHEMBL5145                    # confirm the ID is the protein you think it is
sarscope fetch  CHEMBL5145                    # download (cached) and summarise the raw records
sarscope run    CHEMBL5145 --out braf_report/ # curate, analyse, write the report folder
sarscope run --input my_data.csv --out report/  # your own SMILES + potency table
```

`run` writes a folder you can open or attach as supplementary data:

```text
braf_report/
  report.html        self-contained, figures inlined
  provenance.json    ChEMBL release, package versions, every parameter
  tables/            36 CSVs: curation log, Tables 2/3/4/6, cliffs, R-groups
  figures/           7 PNGs: Figs. 4, 5, 8, 9, 10
```

In the browser:

```bash
streamlit run streamlit_app.py
```

`fetch` prints the distribution of every field curation acts on, which shows
what you are about to filter:

```text
CHEMBL5145  Serine/threonine-protein kinase B-raf  (Homo sapiens, SINGLE PROTEIN)  [ChEMBL_37]
11017 records, 6703 molecules

  standard_relation        =: 9115, <: 1008, >: 554, None: 172, <=: 114, >=: 54
  assay_variant_mutation   None: 6523, V600E: 4494
  potential_duplicate      0: 8581, 1: 2436
  bao_label                single protein format: 7437, cell-based format: 2011, ...
```

## What it produces

| Section | Output |
|---|---|
| Curation | Every record removed and the step that removed it |
| Properties | Six descriptors by activity group (Table 2), PCA (Table 3, Fig. 5) |
| Scaffolds | Murcko diversity (Table 4), enrichment factors with Wilson bounds |
| R-group SAR | Per-position substituent deltas — the input to a Table 5 read |
| Landscape | SAS maps, activity cliffs, cliff generators (Figs. 8, 9) |
| Models | 14-algorithm bake-off with the leakage audit (Table 6), applicability domain |

Responses are cached per ChEMBL release in `~/.cache/sarscope` (override with
`SARSCOPE_CACHE`), so a report can be regenerated offline, and a new release is
never served stale data.

## Development

```bash
make check          # ruff, mypy, and the core tests (what CI runs)
make test-science   # the analysis-layer specification; fails until implemented
.venv/bin/pytest -m network   # live ChEMBL checks
```

Standardisation, scaffolds and ECFP come from
[Sorbent](https://github.com/yboulaamane/sorbent). The six descriptors do not,
because the paper uses RDKit's `NumHAcceptors`, while Sorbent uses a different
acceptor count on purpose. See `analysis/descriptors.py` for details.

## License

MIT. ChEMBL data is CC BY-SA 3.0; cite ChEMBL when you publish results built on it.

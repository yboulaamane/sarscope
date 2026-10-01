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
target, curate it, then run chemical space, scaffolds/SAR, activity cliffs, ML,
and descriptor explanation only when requested. Assay IDs and minimum ChEMBL
target-confidence scores can narrow the curated evidence; the retained
measurements remain inspectable behind each molecule. The ML stage includes a
small MLP alongside tree and neighbour models. The explanation stage uses named
RDKit physicochemical descriptors and reports held-out permutation importance;
Random Forest also reports impurity importance. TreeSHAP is included in the
hosted requirements and runs only when selected (up to 100 held-out compounds).
Local installs can enable it with `pip install "sarscope[explain]"`.

ML also supports selected RDKit 2D descriptors or fingerprint–descriptor hybrids.
Presets span physicochemical, medicinal-chemistry and the full installed 2D
catalog (excluding Ipc/AvgIpc). Preprocessing is fitted within training folds
and saved with ordered feature names for prediction. Molfeat's RDKit 2D backend
is an optional `sarscope[molfeat]` extra, excluded from cloud defaults due to PyTorch.
Regression adds Ridge, a training-mean reference, MAE, parity/residual plots,
3-fold/10-fold error rates, novelty summaries and optional scaffold-block bootstrap
intervals. See [QSAR methodology](docs/qsar-methodology.md) for the Walters/Bjerrum
examples that informed this design, interpretation, exports and scientific limits.

Class cutoffs are editable in the sidebar, defaulting to pActivity 8 / 7 / 6
(10 / 100 / 1000 nM). Enrichment Group 1 means potent + active; Group 2 means
intermediate + inactive. ML classification uses the four labels, while
regression uses continuous potency. Changed cutoffs require rerunning curation.
The curation view shows all four class counts and their ranges, including zero
counts, with horizontal chart labels. Protein-variant filtering accepts an exact
ChEMBL mutation annotation for any target; it is not restricted to BRAF V600E.
The default keeps records without a mutation annotation, which does **not**
confirm that the assayed protein is wild-type.
The cliff viewer supports molecule-ID filtering, sorting and pagination across
all discovered pairs; applied cliff thresholds are shown beside the results.

Selected cliff pairs also have an **opt-in Vina + ProLIF docking panel**. Find
experimental PDB structures by the target's UniProt accession, review the protein
construct/chains, and prepare the receptor with Meeko without uploading files.
A selected bound ligand defines the box automatically; when none is suitable,
an explicit fpocket step predicts candidate sites for review. Prepared receptor
uploads and manual boxes remain available. Dock just the two molecules;
undefined ligand stereochemistry is reviewed first, with explicit docking-only
stereoisomer choices, 2D previews and unchanged source activities.
Optional 3D preparation checks flag incompatible states before docking; bounded
embedding retries preserve chirality and export per-ligand diagnostics.
Outputs include Vina scores and their difference,
experimental ΔpActivity, a 3D pose overlay, free local ProLIF 2D interaction
diagrams, and retained/lost/gained residue contacts. A separate ZIP records
poses, contacts, inputs, versions and protocol. No receptor or target is hardcoded.
This is hypothesis support, not an explanation proven by docking scores.
See [cliff-pair docking](docs/cliff-docking.md) for preparation, setup and limits.

Chemical space offers standardized descriptor PCA and a separate opt-in
ECFP4 UMAP with Jaccard distance (1 − binary Tanimoto). UMAP is lazy-loaded;
install `sarscope[space]` locally or use the hosted requirements. Above 2,000
compounds, the projection uses a clearly labeled uniform sample with seed 42.
Neighbor count and minimum distance are editable; coordinates and settings
are downloadable separately from the report ZIP. This unsupervised map is for
exploration, not a QSAR validation or an applicability-domain test.

For larger targets, the curation step shows activity-page and structure
progress, then reports separate record-loading and curation times. Raw ChEMBL
responses are cached by release and selected activity types; standardized
structures are reused across filter changes while the app process runs. Assay,
source, and year filters still apply after the raw download, so they reduce
curation work but not network transfer. On Streamlit Community Cloud, locally
generated cache files [are not guaranteed to persist](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data)
after an app restart.

The dark browser UI uses distinct colors for potency classes and staged
controls. In ML, **Exclude intermediate class** is an explicit opt-in choice:
it removes those compounds from all selected ML tasks *before* splitting and
cross-validation, while curation, chemical space, scaffolds and cliffs retain
them. Potent, active and inactive remain separate classifier labels; the
option does not silently create a binary endpoint. Continuous regression
normally benefits from retaining intermediate potencies, so it is off by
default. The choice and model compound count are recorded in report
provenance.

Two additional opt-in steps use those results: **Selectivity** compares a
second ChEMBL target and exposes both shared-compound potency differences and a
compound-by-target matrix in which an unmeasured target remains *unknown*;
**Prioritize** standardizes and scores an uploaded CSV/TSV of up to 500 SMILES,
then offers an editable property-filtered, diverse shortlist. Predictions show
the nearest measured training analogue and a fingerprint-similarity domain flag.
The 90% residual band is an empirical training-CV diagnostic, not a guaranteed
prediction interval; held-out coverage and error versus novelty are displayed.

Target discovery accepts a **ChEMBL ID**, **gene/protein name**, or **disease**.
Name search uses ChEMBL and defaults to human targets; each candidate displays
its ID, organism, and target type for explicit confirmation. Disease search
uses Open Targets: choose an ontology term, load up to 25 direct gene
associations, then resolve a chosen gene through reviewed UniProt accessions
to human single-protein ChEMBL targets. Association scores summarize evidence;
they are not treatment-success probabilities or potency measurements. Evidence
from descendant disease terms is excluded. Search results are capped, show the
total available count, and can be narrowed by a more specific query.

All discovery calls are button-triggered and cached for one hour, with bounded
cache sizes. Disease discovery adds no dependencies or startup API calls, and
direct ChEMBL lookup remains available if Open Targets is unavailable. Reports
record the discovery path and, for disease discovery, the selected disease,
gene, association score, mapping accessions, and retrieval timestamp.

See [the product priorities](docs/product-priorities.md) for the implemented
scope, scientific limitations, and remaining validation work.

See [`docs/deploying.md`](docs/deploying.md) to host it.

## What you get

| Section | Output |
|---|---|
| Overview | Headline numbers, most potent molecules and enriched scaffolds, drawn |
| Curation | Every record removed, retained measurements, assay context, original source and cross-source disagreement |
| Properties | Six descriptors by activity group, Mann-Whitney tests, PCA |
| Scaffolds | Murcko diversity, enrichment factors with Wilson lower bounds |
| R-group SAR | Per-position substituent effects, with cores and substituents drawn |
| Matched pairs | Single-cut transformations and directional summaries with support and assay-context flags |
| Landscape | Activity cliffs with both structures, SALI, cliff generators |
| Models | Classification with held-out ROC AUC, average precision (PR AUC), per-class precision/recall/F1 and confusion matrix; continuous pActivity regression; scaffold/time/source-origin split, leakage audit, applicability domain and empirical error band |
| Model diagnostics | Held-out error on activity-cliff compounds versus the rest |
| Comparison | Shared-compound selectivity, shared scaffolds and explicit missing target measurements |

```text
report/
  report.html        self-contained: figures and structures inlined
  model.joblib       selected regressor, refitted on the modelling dataset
  provenance.json    ChEMBL release, package versions, every parameter
  tables/            every table as CSV
  figures/           every figure as PNG
```

`provenance.json` is the point of the whole thing: any number in the report can
be traced back to the ChEMBL release, the package versions and the exact
settings that produced it.

## Using your own data

### Evidence origins without duplicate downloads

ChEMBL already includes [BindingDB (source 37) and PubChem BioAssay (source 7)
records](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/document-and-data-source-questions).
SARscope shows these original source IDs, counts compounds appearing in multiple
origins, and flags cross-origin potency disagreement. It does not append a
second copy from the upstream APIs. ChEMBL's PubChem import covers only a
[subset of confirmatory assay data](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions),
so this is not a claim that all PubChem screens or all BindingDB records are
present. The source filter can focus an analysis on an integrated origin:

```bash
sarscope run CHEMBL5145 --source-ids 7 37 --out report/
```

To test transfer across origins, reserve one as an external-origin holdout:

```bash
sarscope run CHEMBL5145 --split source --source-test-id 37 \
  --cv-folds 5 --out report/
```

Measurements are split before aggregation. A structure also measured in another
origin remains in training and is excluded from the held-out test. This tests
source transfer only where enough origin-unique compounds exist; it is not an
independent prospective experiment. Inside training, model selection uses
scaffold cross-validation. Do not combine `--source-ids` with a filter that
removes every record of the intended training or test origin.

### Qualitative PubChem screens (separate endpoint)

```bash
sarscope screen 1000 --out pubchem_aid1000/
sarscope screen 1000 --snapshot pubchem_aid1000/source_snapshot.json \
  --out pubchem_aid1000_replay/
```

This workflow reads one [PubChem BioAssay AID](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)'s
**Active/Inactive** [outcome calls](https://pubchem.ncbi.nlm.nih.gov/docs/bioassay-tag-names) and
compound SMILES. It excludes inconclusive/unspecified calls and structures
with conflicting calls. It reports held-out scaffold ROC AUC, PR AUC, MCC,
precision/enrichment at the top 10%, and the test-set prevalence baseline.
The original calls and structures are saved in `source_snapshot.json` for
offline replay. Qualitative labels are **never converted to pIC50** or mixed
with the ChEMBL potency pipeline. The direct API route refuses assays at the
10,000-row concise-response safety limit rather than silently truncating them.
For a larger screen, a complete, versioned assay-export importer is still
needed. AID 1000 is a small smoke test, not a general performance claim.

### Frozen multi-target-family benchmark

```bash
sarscope benchmark-freeze --out frozen_chembl_panel/
sarscope benchmark-run --frozen frozen_chembl_panel/ --out benchmark_results/
sarscope benchmark-run --frozen frozen_chembl_panel/ --validation origin \
  --out origin_holdout_results/
```

The fixed panel is human BRAF (kinase), DRD2 (class A GPCR), and ESR1
(nuclear receptor). Freezing records the ChEMBL release, exact target identity,
raw IC50 records and SHA-256 hashes. The offline runner rejects modified
snapshots or protocol/panel drift. The predeclared analysis uses exact nM
wild-type/unannotated binding IC50, median aggregation, ECFP4, random-forest
regression and scaffold holdout; each target is scored against a training-median
baseline with R²/RMSE/Spearman and molecule-level test predictions. This is
cross-family **coverage of within-target tests**, not cross-target transfer or
prospective validation. Frozen ChEMBL records are local artifacts, not bundled
in this repository; review licensing before redistributing them. A benchmark
claim requires running the frozen panel, checking per-target results and
repeating it in a locked environment.
The optional origin test holds out compounds unique to ChEMBL's integrated
BindingDB source 37, partitioning measurements before molecule aggregation and
using scaffold CV inside training. It tests evidence-origin transfer, not
truly independent biology; shared assay designs or chemistry series can remain.
The [first local ChEMBL 37 benchmark](docs/benchmark-chembl37.md) reports
per-family held-out results and limitations; it is not an external validation.

### Local tables

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

ChEMBL runs retain measurement-level document years. For a time split, SARscope
aggregates pre-cutoff records into training labels and only newly measured
structures enter the later test set; later measurements cannot revise an early
training label. Model selection also uses expanding, earlier-to-later year folds
within the training period, requiring at least one more distinct training year
than CV folds. Within those folds, potency labels still reflect all retained
pre-cutoff measurements, so repeated measurements published after a fold's
validation year can influence its earlier training labels. The optional
random-split leakage audit is omitted for time validation because
its test score would not be comparable to a chronological holdout.

To pool the common potency endpoints explicitly, use `--pool-types`; every value
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
training set, a nearest measured analogue, an applicability-domain flag, and
an empirical residual band when available. The band is not a formal
distribution-shift guarantee. `model.joblib` is a Python
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

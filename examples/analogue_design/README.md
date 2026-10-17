# Rebuild the optional measured example

The packaged CSV, source records, licence and walkthrough are in
[`src/sarscope/data/analogue_design`](../../src/sarscope/data/analogue_design/).
`selection_audit.json` here records the four examined assays, including the
unsuccessful RF demonstrations. This is deliberately not a benchmark campaign.

To rebuild into a **fresh output directory**, download the four JSON responses
listed in the packaged `source.json`, using these exact local filenames:

- `CHEMBL944276.json`: the full activity response, limit 1000.
- `metadata-944276.json`: assay CHEMBL944276 metadata.
- `document-944276.json`: document CHEMBL1140899 metadata.
- `status.json`: ChEMBL release status.

Then run from the repository root in the SARscope environment:

```bash
python examples/analogue_design/build_example.py \
  --source-dir /absolute/path/to/downloads \
  --output-dir /absolute/path/to/new-output-directory
```

The builder checks that the complete assay was supplied, applies existing
curation, writes the three-column CSV, and verifies the **serialized upload**
using one unchanged default RF configuration. It also checks reconstruction
and scoring for the recommended parent. It refuses to overwrite frozen output
files or produce a supposedly successful demonstration when gates fail.

The live API can change between releases. Compare response/data SHA256 hashes
with the frozen source metadata before expecting identical results. Original
download JSON is not bundled, but retained per-record input values and IDs are.
Rebuilding is developer work; the deployed example needs no API request,
dependency installation, fitting or proposal generation on page load.

The builder is MIT-licensed software; bundled ChEMBL-derived data have the
separate CC BY-SA 3.0 licence stated in the data directory.

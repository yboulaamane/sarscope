# Verification of the revised scientific manuscript

13 October 2026. Local checks, not external scientific validation or GitHub CI.

- Stereochemical audit recovered the source R-position mapping without
  mismatches. DRD2 had 23 dominant-state compounds, two oppositely assigned
  and one unassigned core; the latter three were excluded from the paper study.
- The revised study has 92 compounds, 263 positional comparisons and 191
  transformation summaries. Original input files and earlier exports remain intact.
- Threshold sensitivity, compound-omission diagnostics, 5,000 permutations per
  series and complete-cycle enumeration were executed from the retained data.
- Re-execution reproduced every saved numerical CSV exactly. The provenance
  records final analysis-script and input-file SHA-256 hashes.
- No new model training, docking or data acquisition was performed for the study.
- Nineteen targeted tests passed: scientific diagnostics, publication integrity
  checks and SAR-map tests. Repo lint/format checks and diff whitespace checks passed.
- The Word documents passed ZIP integrity and exact table-cell comparison against
  the Markdown sources. The main manuscript has one results table and three
  figures; the supplement has six tables and one supporting figure.
- Final Word documents were rendered locally with LibreOffice, producing a
  12-page main manuscript and five-page supplement. Contact sheets were reviewed,
  detailed chemical/effect figures inspected and text-block page bounds checked.
  This is local rendering QA; exact pagination can differ in Microsoft Word.
- The original compound structures are embedded as aligned vector drawings,
  with high-resolution raster fallbacks. Figure concentrations and directional
  effects are calculated from the source records rather than manually supplied.
- Source-publication and correction bibliographic metadata were checked against
  primary records. No expert review or original-table SAR adjudication is implied.

The paper's stereochemical restriction is evaluation-side only; the app has not
been patched to apply it automatically. No transformed source data or findings
were pushed, publicly deposited or submitted. Author details, independent
review/workflow comparisons and public archive/DOI deposition remain pending.

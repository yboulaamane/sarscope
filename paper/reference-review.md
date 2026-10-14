# Reference revision — Journal of Cheminformatics

13 October 2026. Bibliographic and citation QA, not independent validation of
the case studies. Original reference and document versions are preserved in
local run archives.

## Style and reproducibility

The journal's [Software instructions](https://link.springer.com/journal/13321/submission-guidelines/software)
show numbered Basic Springer references and require separately identified web
resources with access dates. The authoring convention used here is
[Springer Basic (numeric, brackets)](https://www.zotero.org/styles/springer-basic-brackets):
for five or more authors display the first three followed by “et al”; otherwise
display all authors. Full author metadata remain in the CSL-JSON, RIS and BibTeX
files. The convention also agrees with the style supplied in the journal's
[Markdown template](https://github.com/jcheminform/markdown-jcheminf).

The single source of truth is `references.csl.json`, containing 32 ordered
entries: 29 unique publication DOIs and three web/software resources. The
bibliography in the manuscript is generated from that library, not maintained
as an independent hand-typed list. Citation numbering follows first appearance.
The Word document links numbered citations to reference bookmarks. Those links
are navigation aids, **not live Zotero/EndNote citation fields**; import the
library into the author's chosen reference manager for that workflow.

## Corrections made

- Replaced inconsistent one-author/three-author “et al” abbreviations with one
  convention and retained the complete publisher author records in exports.
- Restored publication titles from DOI metadata, removed markup/embedded line
  breaks, retained chemical identifiers and Greek characters, and repaired
  punctuation lost in the indexed Maggiora title.
- Standardized journal abbreviations, years, page ranges, article numbers and
  DOI links. The ChEMBL article is cited using its 2024 issue year rather than
  its 2023 online-publication date. Duplicate single-page ranges are collapsed.
- Removed the ChEMBL 2026 copyright-only correction from the scientific
  bibliography. It does not support the assay-annotation claim; the updated
  original article remains cited. The MoleculeACE scientific correction is
  retained separately and cited with the original study.
- Split the SARscope code repository and browser application into distinct web
  entries. Kept release-readiness warnings in Availability, not in references.
- Added SAR maps (2007), Activity Landscape Plotter (2017) and SPARFlow (2026)
  where their published capabilities are discussed. These are relevant prior
  work, not extra references added to increase a count. Missing analogue/cell
  identification and integrated SAR workflows are not claimed as new ideas.
- Added importable `reference-exports/references.bib`, `references.ris` and the
  complete CSL-JSON library.

## What the source checks establish

The library was initialized from the 26 retained publication records and three
new Crossref records. A fresh DOI/title audit then retrieved all 29 publication
entries successfully; the manuscript source hash is recorded with that audit.
Publisher/PubMed records and available abstracts were checked for the
added comparator context. The DOI audit explicitly distinguishes metadata checks
from full-text or original-table adjudication.

This revision does **not** certify recovery of the original case-paper authors'
SAR conclusions, demonstrate equivalence/superiority to SPARFlow or other tools,
or remove the pending independent reviewer assessment. Claims about numerical
results still rest on the accompanying source data and analysis, not bibliography
formatting. The manuscript continues to state those limits.

Before submission, replace the development-repository citation with an additional
versioned software/data archive citation once a genuine release DOI exists. No
DOI or release date is invented in this revision.

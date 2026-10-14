# Release and submission gates

## Prepared locally

- [x] Source code under MIT; draft citation metadata without an invented DOI.
- [x] Frozen small assay datasets with original records, references and hashes.
- [x] Numerical reference check and actual mmpdb coverage comparison.
- [x] Machine-readable results, maps and unfilled human-review templates.
- [x] Direct dependency pins and environment/version metadata.
- [x] Small-case reproduction in a fresh isolated Linux/Python 3.13 environment;
  all 26 CSV outputs match apart from elapsed time. Hashed runtime lock supplied.
- [x] Archive builder with file hashes; optional full benchmark inclusion.
- [x] Manuscript describes negative results and currently missing evidence.

## Required before submission

- [ ] Confirm author list, affiliations, contributions, funding, competing
  interests and AI-assistance disclosure. Do not invent “no funding/no conflicts”.
- [ ] Obtain/read all relevant full texts and tables; freeze independent SAR
  expectations before scoring them. Document compound-ID mapping discrepancies.
- [ ] Two independent medicinal-chemist reviews plus adjudication where needed.
- [ ] Execute the counterbalanced DataWarrior/RDKit/SARscope workflow comparison.
- [ ] Independently reproduce from the final downloaded release archive on
  another machine. Extend environment verification if other platforms are claimed.
- [ ] Complete code QA and store exact test/CI outcomes for the release revision.
- [ ] Render and inspect the submission document/figures, check citations,
  captions, accessibility, supplementary files and journal format.
- [ ] Commit the reviewed state, choose a release version and create a release
  tag consistent with package version and CITATION.cff. Current version is development.
- [ ] Review the archive for private files; include full ChEMBL 37 snapshots,
  saved predictions and licenses if the QSAR results remain in the paper.
- [ ] Publish code/data anonymously accessible and archive the exact release
  on Zenodo or another accepted repository. A local ZIP is not a DOI deposit.
- [ ] Add the actual software/data DOI and public URLs only after successful
  deposition; verify downloads and hashes from outside the author's environment.
- [ ] Update manuscript availability, final figures and declarations; remove
  working-draft warnings only when the checklist is genuinely satisfied.

Target format: Journal of Cheminformatics Software article. Its
[current guidance](https://link.springer.com/journal/13321/submission-guidelines/software)
emphasizes reproducibility, open source and a material advance over existing
tools. Acceptance is not guaranteed by feature count or this checklist.

No external submission, release publication, reviewer messaging or DOI deposit
is performed by the evaluation commands.

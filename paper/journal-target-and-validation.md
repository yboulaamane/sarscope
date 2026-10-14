# Journal target and validation audit

Prepared 13 October 2026. This is an author-facing decision note, not a claim of
submission readiness. Original scientific manuscript and Word/PDF files are preserved.

## Recommended target

**Journal of Cheminformatics — Software article.** It is the closest fit for an
open, reusable cheminformatics workbench with scientific case studies. This is
a scope-fit recommendation, not a prediction of acceptance. The main editorial
risk is whether SARscope demonstrates a meaningful advance over existing tools,
not whether its introduction has enough references.

Current [Software guidance](https://link.springer.com/journal/13321/submission-guidelines/software)
was checked. The tailored draft uses Introduction, Implementation, Results,
Discussion, Conclusions, Availability and requirements, and separate declarations.
Its abstract contains a Scientific Contribution subsection of two sentences.
The journal emphasizes anonymous access to source/data, reproducibility and
comparison with existing software. The source revision needs a public archive;
the current development homepage alone is insufficient.

Check current publication charges and institutional funding/waiver eligibility
before committing to the journal. No fee amount or eligibility is assumed here.

## What changed in the journal draft

- Expanded the introduction to approximately 800 words: medicinal-chemistry
  reasoning, established matched-pair/series methods, assay variability,
  stereochemistry, context dependence, activity cliffs and software reuse.
- Expanded the bibliography to 32 numbered entries, including recent 2025
  inter-assay work and 2026 ChEMBL assay annotations, with relevant corrections.
  Every entry is cited; references were not added merely to increase the count.
- Centered the contribution on traceable evidence stress testing and actionable
  evidence gaps, rather than new descriptors, docking or algorithmic novelty.
- Added a current implementation audit, a workbench schematic and an explicit
  distinction between median-sign retention and the full directional criterion.
- Kept the negative permutation result and the mmpdb hydrogen-pair limitation.
- Used the author's supplied affiliation and preserved working-draft declarations
  rather than inventing a
  funding statement, conflict declaration or public DOI.

## Are the case studies validated?

**Partly. They are numerically verified, not fully independently scientifically
validated.** These are separate claims:

| Evidence question | Current status | What it supports |
|---|---|---|
| Frozen source records and hashes | Complete locally | Traceability of the supplied ChEMBL snapshots |
| Pair identities and signed effects | All 263 match exhaustive enumeration and the saved scientific cases | Implementation consistency; shared RDKit chemistry is not an independent chemical reference |
| Stereochemical core restriction | DRD2 retains 23 and excludes the same 3 other-state compounds | Encoded structure comparability, not experimentally verified stereochemical purity |
| Descriptive robustness and permutation diagnostics | Executed; 17 sign-stable claims, 15 retaining the full tested criterion; none of 191 adjusted permutation tests below 0.05 | Within-dataset sensitivity, not calibrated confidence |
| Assay/publication omission robustness | Implemented, but every case has one assay/publication | Omissions leave no evidence; cross-source persistence is not demonstrated |
| Original-paper compound/table adjudication | Pending | Needed before claiming independently established recovery of author SAR conclusions |
| Medicinal-chemist review and task comparison | Pending | No proven interpretation accuracy or time-saving advantage |
| Experimental value of suggested measurements | Pending | Suggestions remain hypotheses, not validated design recommendations |
| Anonymous archived release and external reproduction | Pending | No final public code/data release or DOI yet |

The preceding full local suite passed 414 tests (3 skipped; 2 network tests
deselected). This is not peer review or experimental validation. The current
offline audit is in `workbench-audit/`, with source/input/output hashes. The revised
bibliography has 29 unique publication DOIs and three separately identified
software/web resources. Metadata are recorded in the structured reference
library and DOI audit; references were numbered in first-appearance order. This
bibliographic check is not verification of claims against original article tables.

## Practical path to submission

1. Independently check the original publications and supporting tables, including
   compound-number/ChEMBL mappings and assay assignments. Freeze expectations
   before scoring the output; do not ask reviewers to confirm the tool's labels.
2. Regenerate the historical review pack using the restricted DRD2 core, then
   obtain independent chemist decisions. The existing 41 tasks were generated
   before this restriction and remain unreviewed.
3. Execute the prespecified comparator workflow using the same inputs, cached
   data and tasks. Distinguish this project's review/usability protocol from
   journal-mandated sample sizes; the journal does not prescribe our six-user pilot.
4. Freeze and archive the tested release with all required source data, scripts,
   environment metadata and licenses. Reproduce from the downloaded archive on
   another machine, then add only the genuine DOI and verified availability URLs.
5. Confirm authorship, funding, conflicts and AI-assistance disclosure; the supplied
   affiliation is included on the title page.
   Review the final Word/PDF and any journal editorial feedback before submission.

A temporal/publication-disjoint, multi-family evaluation is necessary if claiming
improved SAR transferability or decision quality. It is not supplied by the
existing scaffold-split QSAR results or the three selected retrospective cases.
Wet-lab experiments are not being claimed or invented to strengthen a software
paper. The paper can remain a software contribution, with benefits stated at the
level actually tested.

## Prepared files

- `jcheminformatics-manuscript.md` and the matching journal-tailored Word/PDF.
- `jcheminformatics-supplement.md` and its matching Word/PDF.
- `audit_current_workbench.py` and `workbench-audit/` for the current offline audit.
- `figures/sar-stress-workflow.svg`, a 920×300 white-background vector schematic
  suitable for consideration as a graphical abstract; it does not depict results
  of unperformed experiments.

No manuscript submission, public release, DOI deposition or external reviewer
communication was performed during this revision.

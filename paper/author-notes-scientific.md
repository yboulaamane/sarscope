# Author notes for the revised scientific manuscript

Prepared 13 October 2026. This file is an editorial handoff, not manuscript text.

## What changed

- Reframed the article around positional SAR rather than a software feature list.
- Added a stereochemical core audit, single-compound sensitivity, a fixed
  effect-threshold grid, 5,000 label permutations per series and complete
  four-compound substitution cycles.
- Replaced the DRD2 positional results with the stereochemically restricted
  23-compound analysis. Its 36 positional pairs replace the previous 46.
- Added individual-effect and substitution-cycle figures; moved dependency,
  execution and QSAR benchmark details to supplementary material.
- Kept negative results, incomplete validation and AI-assistance disclosure.

## Important interpretation limits

The original application scaffold grouping did not separate two oppositely
assigned and one unassigned DRD2 core from the dominant state. The paper analysis
does. A subsequent local implementation on 13 October now preserves mapped
per-molecule core identities, provides a browser core-state selector and uses
the largest identical state for CLI maps. Its DRD2 regression check retains
23 compounds and excludes the same three other-state compounds. This does not
retroactively change the original saved results or the delivered Word/PDF
documents, which still describe the evaluation-side restriction. Freeze and
review the updated code before updating the manuscript or public release.

None of the 191 transformation permutation tests passed the adjusted 0.05
cutoff. This does not make the measurements false, but it rules out presenting
the descriptive “strong/moderate” labels as statistically validated confidence.
The paper does not claim to discover a binding mechanism, infer ER-beta selectivity,
validate new potency predictions or explain the prior mutant-BRAF docking pair.

## Required author decisions and work

1. Confirm affiliation, corresponding-author details, final authorship,
   contributions, funding, competing interests and assistance disclosure.
2. Verify the source publications and compound mappings against original tables.
   The current cases are analyses of ChEMBL measurements, not independently
   certified recovery of the original authors' SAR conclusions.
3. Review the permutation null, dependence assumptions and reporting with a
   statistician/cheminformatician. It is a diagnostic, not assay-noise calibration.
4. Arrange independent medicinal-chemist review and execute the planned workflow
   comparison. Do not replace these with automated tests or invented reviews.
5. Review and release the locally implemented stereochemical safeguard and SAR
   stress-test workflow. Do not claim the new evidence-gap suggestions are
   independently validated or that later-paper/temporal SAR validation was run.
6. Freeze the final source/data/environment release and deposit it publicly with
   a real DOI. Existing local archives are not a public release.

The rewrite improves the manuscript and adds actual analyses. It does not
remove the remaining submission gates or guarantee journal acceptance.

The previously supplied 12 October reproducibility ZIP contains the earlier
manuscript package, not this revised study. Regenerate the source/data archive
after reviewing the new documents and before public deposition.

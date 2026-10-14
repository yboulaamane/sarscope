# SARscope: interactive, assay-aware structure–activity interpretation with activity-cliff diagnostics

**Working software-article draft. Not for submission yet.** Independent literature
annotations, expert review and the user-level comparator study remain pending.
Author list/affiliations/declarations require confirmation. Local artifacts are
not a publicly archived release. Paper text/protocol/graphics: CC BY 4.0;
ChEMBL-derived evidence: CC BY-SA 3.0; software: MIT.

Draft corresponding author: Yassir Boulaamane, yassirxmail@gmail.com, from project
metadata. This does not establish final authorship, affiliation or contributions.

## Abstract

### Background

Public bioactivity records are useful for examining medicinal-chemistry series,
but interpretation requires maintaining experimental context, identifying
comparable structural changes and distinguishing observed effects from
unsupported generalizations. SARscope provides a staged browser and command-line
workflow that links curated measurements to interactive position-level SAR,
activity-cliff inspection and optional retrospective QSAR.

### Implementation and results

The software combines established RDKit structure analysis with measurement
provenance, matched single-position comparisons and explicit descriptive support
grades. Three frozen published-data examples cover BRAF, DRD2 and estrogen
receptor alpha. Their largest scaffold series contain 16, 26 and 53 molecules,
yielding 29, 46 and 198 position-pair records respectively. Exhaustive reference
enumeration reproduced all pair identities and numerical differences. A
single-cut comparison on identical curated inputs found all SARscope compound
pairs within mmpdb's output, while mmpdb identified 56 additional pairs involving
hydrogen substitutions. Supporting retrospective scaffold-held-out QSAR results
vary substantially by target. Expert agreement and user-level comparative benefit
have not yet been measured.

### Scientific Contribution

SARscope links interactive position-level SAR statements to exact source
measurements and distinguishes matched-assay effects from pooled evidence.
Its staged workflow combines interpretable support counts, conflicting examples
and activity-cliff diagnostics without equating them with causal mechanisms.
The current contribution is an auditable interpretation workflow built on
established cheminformatics methods, not a new fingerprint, pairing algorithm
or experimentally validated optimization strategy.

### Conclusions

The implementation and examples establish feasibility and numerical consistency.
Independent review and comparative evaluation are required to establish the
workflow's incremental practical value and support publication claims.

Keywords: structure–activity relationships; activity cliffs; medicinal chemistry;
ChEMBL; assay provenance; interactive cheminformatics; QSAR.

## Introduction

Bioactivity databases such as ChEMBL collect structures and measurements with
assay and publication context [1]. A target-level potency table is not, by
itself, a coherent SAR series: measurements can differ in endpoint, assay
format, protein construct, reporting precision and source provenance. Structural
similarity can help identify local changes, but a large potency difference
requires inspection of the measurements and chemical context before a medicinal
chemist can act on it.

Established tools already address important parts of this problem. DataWarrior
provides R-group decomposition, chemical-space visualization, activity-cliff
analysis and separation by experimental categories [2,3]. mmpdb offers matched
molecular-pair analysis, transformation environments and property-change
statistics [4,5]. RDKit supplies core molecular representations and decomposition
methods [6]. SARscope therefore does not claim these capabilities are novel.

The intended contribution is a browser-accessible, staged interpretation workflow:
a user selects the target and experimental subset, explicitly curates it,
chooses an analysis, and follows an annotated structural change back to exact
source records. Directional statements remain separate from mechanistic
explanations and prospective predictions. The contribution must ultimately be
judged by traceability, interpretation quality, reproducibility and user task
performance relative to existing tools, rather than by feature count.

## Implementation

### Architecture and staged interaction

SARscope is an MIT-licensed Python package with a Streamlit interface and a CLI.
The analysis layer is independent of the browser. Structure standardization
uses Sorbent/RDKit; tabular operations use pandas; numerical/statistical analyses
use NumPy/SciPy; optional QSAR uses scikit-learn. An embedded SVG component
supports clickable and keyboard-accessible attachment positions without a
remote graphics dependency. The browser does not run every analysis upon loading:
curation, chemical space, scaffolds/SAR, activity cliffs, modeling and explanation
are explicit user actions.

Figure 1 depicts the scope and evidence boundaries. Selecting a position displays
fragment changes, supporting compounds, measured concentrations, effect sizes
and source assay records. Exports contain structures, retained measurements,
pair-level evidence, transformation summaries, an annotated SVG and analysis
scope. The map is an evidence-navigation interface, not an automatic biological
conclusion generator.

### Curation and provenance

The default quantitative workflow retains exact molar measurements for the
selected endpoint/assay type, excludes source validity flags and likely
duplicates, standardizes structures and aggregates repeated measurements by
standardized structure. Every filter has an explicit count and recorded setting.
Raw records and retained measurement IDs are preserved. Quantitative conversion
to -log10(molar) is a unit conversion, **not** a conversion of IC50 to Ki or of
qualitative screening activity to potency. PubChem qualitative calls use a
separate classification workflow.

Structured variant annotations are normalized without inferring mutations from
free-text descriptions. Missing annotations remain unreported, not confirmed
wild type. Raw assay descriptions are retained for review, since discrepancies
between structured fields and prose can remain. A shared assay identifier is a
compatibility safeguard, not proof that all measurements are experimentally
interchangeable.

### Position-level SAR

Within a chosen Murcko scaffold series, symmetry-aware RDKit R-group decomposition
defines a labeled core and variable positions. A comparison is admitted at a
position only when both compounds agree at every other mapped variable position.
For shared assay, endpoint and normalized variant context, per-compound
measurement medians determine the signed potency difference. If no compatible
context exists, pooled molecule values are labeled as unmatched and are not
promoted to supported assay-matched evidence.

Transformation summaries preserve endpoint and variant strata. Each molecular
pair contributes one median effect across its compatible assay measurements;
repeated assays do not multiply pair support. The summary displays the number
of molecular pairs, distinct other-position contexts, assays, and a conservative
deterministic count of disjoint pairs sharing no compounds. Disjoint pairs are
not called independent biological experiments. Attachment labels refer to the
RDKit decomposition, not medicinal-chemistry atom numbering.

Descriptive grades use explicit heuristics. At the fixed 0.5-log-unit threshold,
subthreshold changes, less than 80% sign agreement or opposing assay effects
are mixed. A coherent assay-matched transformation with at least two disjoint
pairs is moderate; at least three disjoint pairs across at least two other-site
contexts is strong. Remaining effects are limited. A reached pair cap prevents
promotion to moderate/strong. Grades are not calibrated confidence levels or
probabilities; the effect threshold and support rules require independent review.
Fixed positions, linker changes, coupled multisite changes and some symmetry
ambiguities remain outside this map's representational scope.

### Activity cliffs and optional modeling

Activity-cliff exploration exposes multiple pairs under user-selected fingerprint
similarity and potency-gap thresholds. Cliffs motivate inspection of measurements
and local transformations; they do not identify the cause of the activity change.
Optional classification/regression supports explicit random, scaffold, temporal
and source-based validation settings, descriptor/fingerprint representations,
baselines, held-out diagnostics and applicability-domain summaries. Regression
retains the continuous endpoint. A favorable random split cannot establish
prospective performance, and an applicability-domain flag is not assurance of
prediction accuracy. The main article focuses on SAR interpretation rather
than asserting predictive superiority of these standard models.

## Evaluation methods

### Frozen published-data examples

Three literature assays were chosen from the already frozen ChEMBL 37 family
panel for populated, chemically related series [7–9]. This was retrospective
selection, not preregistration. Cases are BRAF/CHEMBL5046398,
DRD2/CHEMBL671074 and ER-alpha/CHEMBL829540. The ER-alpha case is a counter-assay
from a paper on ER-beta selectivity; its measurements must not be relabeled ER-beta.
Within each assay, series selection uses the largest post-curation Murcko group,
with lexical tie-breaking and no potency-based selection.

The small source cohorts, complete assay IDs, publication references, original
records and hashes are included in the repository. Curation retains literature
source records, binding IC50 in nM and unannotated structured variants. These
examples overlap the benchmark source pool and are not independent QSAR tests.
An exhaustive reference separately enumerates all compound combinations within
the same RDKit-derived series, checking exactly one differing position and
recalculating deltas from assay medians. This is numerical verification using
shared chemistry infrastructure, not independent biological SAR validation.

### Comparator and review design

An actual mmpdb 3.1.4 comparison uses identical curated inputs, one cut, a common
cut SMARTS and 10-heavy-atom variable-fragment limit. It compares unordered
compound-pair membership, with complete command logs and discrepancies exported.
The algorithms differ in implicit-hydrogen handling and context enumeration, so
pair-count differences alone cannot be labeled errors. Process timings include
startup/I/O for mmpdb but only in-process execution for SARscope and are therefore
not used to claim a speed advantage.

A 41-task grade-stratified expert review pack has been generated, but no reviews
have been performed. Reviewers receive directed transformations and raw evidence
without the tool's grade or summary. The protocol requires at least two independent
reviewers, explicit unresolved cases, source citations and adjudication records.
Separately, original-paper SAR expectations must be annotated before assessing
trend recovery. A counterbalanced DataWarrior/RDKit/SARscope user-task comparison
is specified but not run. Full procedures and empty templates are supplied in
`evaluation-protocol.md`; they are not presented as completed experiments.

### Supporting QSAR uncertainty audit

The existing panel uses one 200-tree random-forest regressor with ECFP4 features,
fold-local filtering, three-fold scaffold-group training CV and a 20% outer
scaffold holdout, seed 42. A training-set median is the baseline. The previous
run is described in `docs/benchmark-chembl37.md`. The new audit resamples the
saved test scaffold clusters 1,000 times, treating acyclic molecules as separate
groups, and obtains percentile intervals for RMSE and paired baseline improvement.
No model is retrained. These intervals condition on the existing split and fitted
model; they are not prospective or repeated-training uncertainty estimates.

## Results

### Case coverage and numerical consistency

| Case | Raw records | Retained measurements | Curated molecules | Scaffolds | Selected series | Position pairs | Transformations |
|---|---:|---:|---:|---:|---:|---:|---:|
| BRAF | 46 | 44 | 44 | 8 | 16 | 29 | 15 |
| DRD2 | 34 | 31 | 30 | 5 | 26 | 46 | 21 |
| ER-alpha | 107 | 100 | 98 | 11 | 53 | 198 | 155 |

Selected-series coverage was 36.4%, 86.7% and 54.1% of curated molecules
respectively. No selected-series compound failed decomposition and no pair cap
was reached. Reference enumeration agreed with all 273 position-pair records
to an absolute tolerance of 1e-10 log units. Cases retain all scaffold counts
so coverage omissions remain visible. Local case-analysis times were a few
seconds per case in one run (recorded in the CSV); these do not measure Streamlit
cold starts. A fresh isolated Linux/Python 3.13 environment reproduced all 26 CSV
outputs exactly apart from elapsed times. A transitive, wheel-hashed analysis
runtime lock is supplied; other platforms and external release-archive
reproduction have not been evaluated.

BRAF produced one strong, four moderate, nine mixed and one limited transformation.
An illustrative methoxy-to-hydroxyl change at mapped R4 had median delta +1.43
across three disjoint pairs and three other-site contexts. This describes about
27-fold lower IC50 within those comparisons, not a general design rule or a
claim of reproducing the authors' interpretation. The annotated map is Figure 2.

DRD2 produced two strong, three moderate, thirteen mixed and three limited
transformations. Removing chlorine at mapped R2 had median delta +0.82, whereas
removing it at mapped R4 had delta -0.73. Thus a generic “chlorine is beneficial”
statement would conceal position dependence in this selected series. These
observed D2 IC50 effects do not establish D2/D4 selectivity or antipsychotic efficacy.

ER-alpha produced two strong, five moderate, eighty-three mixed and sixty-five
limited transformations. Most transformations were mixed or sparsely supported,
illustrating why a map should not turn every numerical delta into a confident
recommendation. No independent medicinal-chemist judgment is yet available for
any of these statements or grades.

### mmpdb comparison and a concrete limitation

| Case | SARscope unique MMP pairs | mmpdb unique MMP pairs | Shared | mmpdb-only |
|---|---:|---:|---:|---:|
| BRAF | 114 | 116 | 114 | 2 |
| DRD2 | 205 | 205 | 205 | 0 |
| ER-alpha | 441 | 495 | 441 | 54 |

Every unique SARscope pair was found by mmpdb in this experiment. Inspection of
all 56 mmpdb-only pairs found an explicit-H variable fragment in each. The
current SARscope single-cut MMP workflow therefore misses these implicit-H
substitution pairs, although its scaffold R-group map does represent H alternatives.
Multiple context rows per pair explain differences in raw row counts. This is
a useful negative comparison: the potential publication value lies in evidence
navigation, not stronger pair enumeration. DataWarrior and human workflow
comparisons remain pending.

### Supporting retrospective QSAR result

| Target | Test compounds | R² | RMSE | Conditional 95% RMSE interval | Spearman | Median-baseline RMSE |
|---|---:|---:|---:|---:|---:|---:|
| BRAF | 584 | 0.762 | 0.646 | 0.578–0.720 | 0.859 | 1.342 |
| DRD2 | 99 | 0.310 | 0.902 | 0.756–1.041 | 0.574 | 1.087 |
| ER-alpha | 651 | 0.696 | 0.828 | 0.744–0.925 | 0.836 | 1.503 |

The weakest case is DRD2, which must not be hidden by a panel average. Paired
bootstrap intervals for baseline-minus-model RMSE were positive for these saved
predictions; this supports improvement over a constant baseline on the particular
held-out splits, not external reliability. Figure 3 displays per-target RMSE.
Assay/document dependencies can remain across scaffold splits. The panel contains
only three selected targets and cannot establish general cross-family utility.

## Discussion and limitations

The principal intended use is to help computational and medicinal chemists
inspect a series with explicit provenance and controlled local comparisons.
The interactive map makes positional context, effect sizes and supporting records
available together. Mixed and limited evidence should guide review rather than
be hidden by a single summary or model score.

However, established comparators already provide SAR and activity-cliff analysis.
The current numerical examples do not establish that SARscope materially improves
interpretation or user efficiency. The evidence-grade heuristics have no independent
calibration or expert agreement study. These are decisive evaluation gaps, not
minor reporting omissions. Case selection is retrospective and limited to the
largest Murcko series; substructure/symmetry choices and chemical standardization
can affect which transformations are represented. Binding IC50 is assay-dependent
and is not affinity, selectivity or cellular efficacy. Structured annotations can
be incomplete, and source text requires review without silently changing labels.

The optional docking workflow is deliberately excluded from the central contribution.
Rigid-receptor docking scores are hypotheses; unstable rankings cannot explain
an activity cliff. The previous exploratory BRAF pair reversed ranking with Vina
seed changes. It should not be used to claim a fluorine-dependent mechanism or
improved experimental potency prediction. No docking protocol or experimental
mechanistic validation was newly conducted for this paper package.

The lowest-cost next falsification test is independent chemist review of the
prepared transformations: unsupported moderate/strong directional claims or
systematic missed/conflicting trends would challenge the current grading and
interpretation approach. A workflow-comparison pilot then tests whether the
interface yields an advantage beyond existing tools. These evaluations should
precede expanding the manuscript's scope or adding more modeling features.

## Conclusions

SARscope implements a staged, auditable SAR workflow with interactive structural
annotations, assay-context comparisons and activity-cliff exploration. Frozen
examples establish numerical consistency and expose a concrete MMP coverage
limitation. The software is a plausible publication candidate, but independent
SAR review, completed workflow comparisons and publicly archived reproducibility
assets are required before claiming a demonstrated advance.

## Availability and requirements

Project: SARscope. Home page/source: https://github.com/yboulaamane/sarscope.
Browser application: https://sarscope.streamlit.app/. Language: Python; supported
minimum Python 3.12, with this evaluation observed on Python 3.13.13/Linux WSL2.
Other requirements: RDKit, Sorbent, NumPy, pandas, SciPy, scikit-learn, matplotlib;
Streamlit for the browser. See package metadata for optional dependencies. Core
software license: MIT, with no additional restriction on nonacademic use.
ChEMBL data have a separate CC BY-SA 3.0 license. Current development version:
0.1.0.dev0. A release DOI has not been assigned.

## Declarations — author confirmation required

### Availability of data and materials

Small frozen case snapshots, protocols, exports and tests are in `paper/`.
Full ChEMBL 37 benchmark snapshots and saved predictions are currently local
ignored artifacts and can be included by the archive builder. Until a versioned
archive is publicly deposited, the complete QSAR result is **not independently
reproducible from the Git repository alone**. The manuscript must be updated with
actual anonymous-access URLs/DOIs before submission. Source-data attribution and
transformations are documented in `DATA-LICENSE.md`.

### Competing interests

Not yet supplied by the author(s); requires confirmation before submission.

### Funding

Not yet supplied by the author(s); requires confirmation before submission.

### Authors' contributions

Final roles and author list must be confirmed by the human author(s).

### Acknowledgements and AI assistance

Codex was used to assist software implementation, evaluation-code preparation and
drafting this working manuscript. Human authors must verify the code, numerical
results, citations, interpretations and final text and approve the exact disclosure
before submission. No AI system is listed as an author. Additional acknowledgements
require permission from those named.

## Figures and supplementary files

Figure 1: `figures/workflow.svg`. Original workflow schematic; no mechanistic claim.

Figure 2: `results/braf/sar-map.svg`. BRAF selected-series map. Labels show
observed potency differences/support heuristics, not causal or universal effects.
Full supporting pairs, structures and measurement IDs are exported alongside it.

Figure 3: `benchmark-audit/benchmark.svg`. Existing scaffold-held-out RMSE versus
training-median baseline. Vertical RF intervals are conditional scaffold-cluster
bootstrap percentiles; no prospective or repeated-training claim is implied.

Additional files: source case cohorts/manifest; case CSVs/scope; mmpdb commands,
inputs/pair membership/logs; bootstrap metrics; unreviewed review tasks/template;
evaluation and release protocols. Full benchmark assets must accompany public
deposition if the QSAR section is retained.

## References

1. Zdrazil B et al. The ChEMBL Database in 2023: a drug discovery platform spanning
   multiple bioactivity data types and time periods. Nucleic Acids Res. 2024;52:D1180–D1192.
   https://doi.org/10.1093/nar/gkad1004.
2. Sander T, Freyss J, von Korff M, Rufener C. DataWarrior: an open-source program
   for chemistry aware data visualization and analysis. J Chem Inf Model.
   2015;55:460–473. https://doi.org/10.1021/ci500588j.
3. DataWarrior chemistry documentation. https://openmolecules.org/help/chemistry.html.
   Accessed 12 October 2026.
4. Dalke A, Hert J, Kramer C. mmpdb: An Open-Source Matched Molecular Pair Platform
   for Large Multiproperty Data Sets. J Chem Inf Model. 2018;58:902–910.
   https://doi.org/10.1021/acs.jcim.8b00173.
5. mmpdb source and documentation. https://github.com/rdkit/mmpdb.
   Accessed 12 October 2026.
6. RDKit: Open-source cheminformatics. https://www.rdkit.org/.
   Accessed 12 October 2026. Exact evaluation version is recorded in the outputs.
7. Abdel-Maksoud MS et al. Discovery of New Imidazo[2,1-b]thiazole Derivatives as
   Potent Pan-RAF Inhibitors with Promising In Vitro and In Vivo Anti-melanoma
   Activity. J Med Chem. 2021;64:6877–6901. https://doi.org/10.1021/acs.jmedchem.1c00230.
8. Zhang X et al. trans-1-[(2-Phenylcyclopropyl)methyl]-4-arylpiperazines: mixed
   dopamine D(2)/D(4) receptor antagonists as potential antipsychotic agents.
   J Med Chem. 2000;43:3923–3932. https://doi.org/10.1021/jm990562x.
9. Malamas MS et al. Design and synthesis of aryl diphenolic azoles as potent and
   selective estrogen receptor-beta ligands. J Med Chem. 2004.
   https://doi.org/10.1021/jm049719y.

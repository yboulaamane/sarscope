# SARscope: an interactive workbench for positional structure–activity relationships and evidence stress testing

Yassir Boulaamane

Laboratory of Innovative Technologies, National School of Applied Sciences of Tangier,
Abdelmalek Essaadi University, Tetouan 93000, Morocco

Correspondence: yassirxmail@gmail.com

Molecular Informatics · software and methods draft.
Author declarations require confirmation before submission.

## Abstract

Medicinal chemists interpret structural changes through their measured effects on potency, and those effects depend on attachment position, the remaining structure and experimental context. We present SARscope, an interactive workbench that couples these controls to inspection of the underlying measurements. Three published assay datasets—BRAF, dopamine D2 receptor and estrogen receptor alpha—were reconstructed from frozen ChEMBL release 37 records. Positional comparisons required an identical mapped isomeric core, unchanged substituents elsewhere and a shared assay and endpoint. The selected series contained 16, 23 and 53 compounds, yielding 263 positional comparisons. BRAF hydroxyl addition and DRD2 chlorine removal showed opposite effects at different positions, and a complete four-compound cycle placed both DRD2 positions in one chemical background. All 17 repeated-comparison transformations retained the sign of their median effect after compound omissions, but only 15 retained the configured directional criterion, and none of 191 transformation-level permutation tests met the adjusted 0.05 cutoff. SARscope therefore links positional comparisons to source measurements and exposes their dependence on retained compounds, chemical contexts and practical thresholds, identifying missing-cycle measurements and conflicting pairs for follow-up without predicting potency or assigning calibrated confidence. The cases verify the implementation and illustrate local structure–activity relationships; they do not establish prospective transferability, binding mechanisms or improved chemist performance.

**Keywords:** structure–activity relationships · matched molecular pairs · activity cliffs · chemoinformatics · assay comparability

## 1. Introduction

Lead optimization depends on deciding which changes to a chemical series are
worth pursuing. Medicinal chemists make these decisions by comparing close
analogues, considering whether a measured difference is consistent with other
members of the series and whether the compounds were tested under comparable
conditions. Matched molecular pair analysis provides a formal representation
of this reasoning: a defined structural replacement is associated with the
change in a measured property while the shared structure is retained [1].
Efficient fragmentation algorithms have made these comparisons practical for
large collections [2], and platforms such as mmpdb organize transformations
together with the chemical environments in which they occur [3]. Matched
molecular series extend this view beyond individual pairs and can support
hypothesis generation or prediction when the transfer of activity trends is
evaluated explicitly [4].

Public bioactivity resources offer an opportunity to examine such relationships
outside a single project. ChEMBL supplies structures, standardized activity
measurements, assay descriptions, target assignments and source-document links
[5]. Its curation procedures already address unit consistency, activity validity,
target mapping and potential duplication [6]. Recent work has further enriched
bioassay descriptions and experimental-method annotations through manual and
AI-assisted curation [7]. The remaining task for an analysis tool is therefore
not to recreate assay-linked data, but to make the comparisons supported by
those data easier to inspect. In particular, a target-level collection is not
automatically a coherent chemical series or a single experimental setting.

The experimental context matters because reported potency is not a property
of the molecular structure alone. Investigations of public binding data have
documented uncertainty, undifferentiated stereoisomers and repeated citations
of individual measurements [8]. Mixing heterogeneous IC50 measurements can
introduce additional variability [9]. Paired differences may be less sensitive
to systematic assay shifts than absolute activities, but their agreement still
depends on curation and the comparison being made [10]. An assay identifier is
thus a useful control, not proof of identical conditions or independent
replication. Repeated records also need to remain distinguishable from distinct
chemical comparisons, and missing protein-variant annotations cannot be treated
as confirmation of a wild-type construct.

Chemical context introduces a separate difficulty. The same replacement may
have different effects at different attachment positions, or when another
substituent changes. Four-compound substitution cycles provide a way to inspect
whether two measured changes are additive in a defined chemical background.
Their interpretation requires attention to assay uncertainty; a nonzero contrast
does not by itself establish a binding interaction or a change in mechanism [11].
Activity cliffs draw attention to another form of local behavior, in which
similar compounds differ substantially in potency [12]. Benchmarking has shown
why model evaluation around such compounds cannot be replaced by a global
prediction score [13, 14]. Yet a cliff label remains conditional on the chosen
structural representation and numerical cutoffs. The compounds and measurements
behind the label still need to be examined.

Established software already addresses much of this analysis. DataWarrior
combines chemical visualization with SAR tables and activity-cliff inspection
[15]. SAR maps organize R-group combinations for medicinal-chemistry interpretation,
including the identification of missing analogues and screening measurements
[16]. Activity Landscape Plotter provides web-based examination of activity
landscapes [17]. More recently, SPARFlow integrated chemical curation, similarity
networks, R-group decomposition, activity cliffs and modelability assessment
within KNIME [18]. OOMMPPAA links matched transformations to structural information
for compound-design questions [19]. Nonadditivity analysis examines complete
substitution cycles and their statistical behavior [20], with subsequent work
investigating their chemical drivers and modeling implications [21].

R-group decomposition, cliff visualization and missing-analogue identification
are therefore established capabilities, not novelty claims for SARscope.
The present evaluation focuses on the connection between a positional SAR
statement, its assay-linked source measurements and the evidence-omission tests
applied to it. Comparisons with existing software need to address that workflow,
rather than the number of analysis functions available.

For the end user, the practical question is often narrower: does a particular
substitution conclusion survive removal of the evidence on which it relies?
A chemist may want to know whether one compound determines the median, whether
an effect changes between assays, or whether a missing fourth analogue prevents
a comparison of two chemical backgrounds. These questions are not answered by
adding another descriptor or model. They require a traceable connection between
the proposed SAR statement, its eligible molecular comparisons and the records
retained in the analysis. Machine-readable evidence and executable procedures
also make those statements easier to revisit, consistent with the goals of
FAIR data stewardship and reusable scientific software [22, 23].

SARscope was developed as an interactive workbench for these tasks. It links
mapped substitution positions to the compounds, potency differences and assay
records behind them, then lets the user challenge a selected effect through
evidence omissions and alternative practical thresholds. Context-specific views
and complete cycles help identify where a pooled conclusion becomes misleading.
Evidence-gap suggestions identify measurements that could clarify the comparison;
they are not potency forecasts. Here we evaluate the implementation in three
published chemical series from different target families, examine its agreement
with explicit numerical references and established pair analysis, and assess the
limits of descriptive support labels. The study is a retrospective evaluation
of an interpretation workflow, not an independent validation of SAR transferability
or a demonstration of improved experimental outcomes.

The numerical analysis is separated from the browser interface. Molecular
frameworks [24] and mapped R-groups are calculated using RDKit [25]; record handling
and numerical summaries use pandas and NumPy, with SciPy and scikit-learn used
where required by the broader analysis. The Streamlit interface exposes curation,
chemical-series preparation, activity-cliff analysis and modeling as separate,
user-initiated stages. The SAR workspace therefore does not train models or run
docking merely because a series is opened.

### 1.1. Software design

The core map connects a structural position to its substitution distributions
and controlled pair evidence. The stress-test panel operates on a user-selected
transformation and the curated measurements already retained in memory (Figure 1).
It does not fetch additional records or fit a predictive model. Content-based
signatures prevent an evidence card from remaining current when the selected
transformation, records or test settings change. A comparison limit and bounded
omission/cycle searches are reported explicitly rather than silently treated as
complete analyses.

The implementation preserves each compound's mapped isomeric core. The browser
lets the user select a core state, with the largest state offered first by
compound count, not potency. Command-line maps default to that largest state
and record exclusions. Oppositely assigned and unassigned core states are
separate; CIP labels alone are not used to establish identity because substituent
changes can alter their priorities. Legacy tables lacking per-compound core
metadata are identified as unaudited, rather than used to claim a stereo control.

## 2. Results and Discussion

### Assay and stereochemical restrictions defined the comparison sets

The three assay cohorts contained 187 raw activity records and 172 standardized
compounds after curation. The largest scaffold groups contained 16 BRAF, 26 DRD2
and 53 ER-alpha compounds. The stereochemical audit identified three core states
within the DRD2 group: 23 compounds with one assigned configuration, two with
the opposite configuration and one with unassigned core stereochemistry. The
23-compound state, corresponding to the reported (S,S) cyclopropyl configuration,
was retained. The other two series each contained one mapped core state.

| Series | Curated assay compounds | Selected compounds | Positional comparisons | Distinct transformations |
|---|---:|---:|---:|---:|
| BRAF | 44 | 16 | 29 | 15 |
| DRD2 | 30 | 23 | 36 | 21 |
| ER-alpha | 98 | 53 | 198 | 155 |

The selected series covered 36.4%, 76.7% and 54.1% of their respective curated
assay cohorts. Together they contained 92 compounds and 263 positional
comparisons. Restricting the DRD2 core removed ten comparisons from the original
scaffold-based set. This is a substantive control: a shared scaffold and assay
do not establish that an observed difference is attributable to the displayed
substituent alone.

### BRAF hydroxyl effects depended on attachment position

On the pendant phenyl ring of the BRAF series, adding hydroxyl at the para
position gave a median ΔpIC50 of −1.63 across two matched comparisons. Adding
the same group at the meta position gave +1.57 across two comparisons. Thus the
observed effects differed in direction despite involving the same nominal
chemical change. Their magnitudes correspond to approximately a 43-fold loss
and a 37-fold gain in potency, respectively, when calculated from the median
logarithmic effects.

Replacing meta methoxy with hydroxyl gave a median increase of 1.43 units across
three disjoint pairs in three remaining-substituent contexts, corresponding to
about a 27-fold potency increase. The positional map is shown in Figure 2 and
individual changes with their spread in Figure 3. Removing individual
compounds left the median for this transformation between +1.31 and +1.86.

These observations support a local distinction between the tested phenyl
positions. They do not identify the origin of the difference. In particular,
the data do not distinguish direct protein contacts from changes in solvation,
conformation or other properties, and they do not establish a general advantage
for hydroxyl substitution outside this series.

### DRD2 chlorine substitution showed opposite positional effects

Within the stereochemically restricted DRD2 series, chlorine removal from the
para position of the N-aryl ring increased pIC50 by a median of 0.68 units across
four disjoint pairs. Removal from the ortho position decreased pIC50 by 0.87 units
across three disjoint pairs. The signs persisted after individual-compound
omissions. The compound-omission median ranges were +0.53 to +0.82 for para
removal and −1.37 to −0.68 for ortho removal (Figure 3).

A complete four-compound cycle placed both positions in the same chemical
background (Figure 4). The unsubstituted analogue, CHEMBL341497, had an IC50 of
37 nM. Para chlorine alone, CHEMBL419610, gave 246 nM, whereas ortho chlorine
alone, CHEMBL126633, gave 5 nM. The doubly substituted compound, CHEMBL302075,
had two retained values of 65 and 93 nM; their median pIC50 corresponds to
77.75 nM after back-conversion.

Para chlorination reduced potency in both ortho backgrounds, while ortho
chlorination increased potency in both para backgrounds. The difference between
the conditional effects was −0.369 pIC50 units. This is a modest departure from
additivity, not evidence of a cooperative binding mechanism. Experimental
replicate assignments and assay uncertainty were insufficient to distinguish
such an effect from measurement variation. The cycle nevertheless provides a
clearer account of the positional SAR than a pooled statement about chlorine.

### ER-alpha transformations were dominated by sparse or mixed evidence

The ER-alpha series yielded the most transformations, but only seven of its
155 transformations met the moderate- or strong-support criteria. Eighty-three
were mixed or subthreshold and 65 had limited support at the default threshold.
At mapped R3, replacing bromine with vinyl or allyl gave median changes of −0.68
and −0.69 pIC50 units across four and three disjoint pairs, respectively. Their
directions persisted after compound omission (Figure 3).

These effects describe the ER-alpha counter-assay. They provide no estimate of
ER-beta selectivity or receptor agonism. The contrast between the large number
of displayed transformations and the small number with repeated comparison
support illustrates why transformation coverage should not be equated with
confidence in a medicinal-chemistry recommendation.

### Repeated observations did not establish statistical confidence

Seventeen transformations met the default repeated-comparison criteria across
the three series, and every one retained the sign of its median after each
valid compound omission. The number meeting those criteria fell from 25 at
a 0.25-unit threshold to 17 at 0.5, 11 at 0.75 and seven at 1.0. The threshold
therefore materially affected how much of the dataset appeared to support a
directional interpretation.

The permutation diagnostic provided a different perspective. At the default
threshold, shuffled assignments produced means of 4.36, 2.89 and 5.90 qualifying
transformations in BRAF, DRD2 and ER-alpha, compared with observed counts of
five, five and seven. The probabilities of a shuffled count at least as large
as the observed count were 0.453, 0.136 and 0.389. None of the 191 transformation
P values remained below 0.05 after multiple-comparison adjustment.

The observed directions and the omission checks remain useful descriptions of
these measurements. They are not independent corroboration of a generalizable
substituent effect. In these examples, pair count, consistency and effect size
were insufficient to make the software's descriptive support grades calibrated
statistical evidence.

### Stress tests separated sign retention from a persistent conclusion

The current workbench reproduced all 263 scoped pair identities and potency
differences against both the saved scientific results and the exhaustive
reference. In the DRD2 case it retained the same 23-compound core state and
excluded the same three opposite or unassigned-state compounds. This establishes
consistency between the tested implementation and the case-study calculation,
not external validation of the source measurements.

The 17 repeated-comparison transformations were then examined with the full
directional criterion, rather than median sign alone. Fifteen retained that
criterion in every tested nonempty evidence subset. The other two retained their
sign but became smaller than the 0.5-unit practical threshold after specific
compound omissions.

| Series | Evaluated transformations | Median sign retained after compound omissions | Full directional criterion retained in tested nonempty subsets |
|---|---:|---:|---:|
| BRAF | 5 | 5 | 4 |
| DRD2 | 5 | 5 | 5 |
| ER-alpha | 7 | 7 | 6 |
| Total | 17 | 17 | 15 |

For BRAF, fluorine removal at R1 had a baseline median of −0.684 units. Omitting
CHEMBL5077573 or CHEMBL5083036 changed the median to −0.256: the direction was
unchanged, but the configured evidence no longer supported a directional
statement above the practical threshold. In ER-alpha, the R3 bromovinyl-to-cyano
comparison changed from +0.603 to +0.152 after omitting CHEMBL186513 or
CHEMBL188900. Chemical-context omissions exposed the same dependencies.

Each case contained one source assay and one publication. Omitting either left
no eligible evidence for all 17 transformations. These examples therefore
cannot validate persistence across assays or publications, even though the
software supports those checks. No omission or cycle-search bound was reached
in this audit. Complete-cycle views and bounded missing-cell suggestions were
produced from the observed analogue patterns, but the proposed measurements
were not performed. Their experimental value remains to be evaluated.

The full software test run completed 420 tests with three skips and two network
tests deselected. The browser tests included selection of core states, a real
BRAF four-compound cycle display, stale-result invalidation and hashed evidence
exports. These tests are evidence of implementation behavior, not independent
medicinal-chemistry review or a usability comparison.

### Activity-cliff calls were sensitive to the chosen definition

Three BRAF positional comparisons exceeded a 100-fold potency difference. Only
one also exceeded the reference fingerprint-similarity threshold. It linked
CHEMBL5092007 and CHEMBL5076831, with reported IC50 values of 97 and 9,710 nM,
Tanimoto similarity 0.918 and a potency difference of 2.00045 pIC50 units. No
mapped DRD2 or ER-alpha pair satisfied the reference criterion.

The BRAF call disappeared when the similarity cutoff was increased to 0.95 or
the potency-gap cutoff to 2.5 units. Its gap was only slightly above the chosen
2.0-unit boundary. Two other BRAF comparisons had larger potency gaps but
similarities of 0.800 and 0.826, so they were excluded by the fingerprint cutoff.
The practical implication is that a cliff flag should direct attention to a
pair, not determine whether its measured difference deserves inspection.

### Comparison with mmpdb identified a coverage limitation

On the complete curated assay cohorts, SARscope's single-cut MMP routine returned
760 distinct within-cohort compound pairs. All were present among the 816 pairs
found by mmpdb. The 56 additional mmpdb pairs involved hydrogen substitution:
two in BRAF and 54 in ER-alpha. The current single-cut MMP routine therefore
does not have equivalent coverage for implicit-hydrogen changes. The separate
R-group analysis does represent hydrogen alternatives, as the examples above show.

This result supports consistency within the overlapping pair set and identifies
a concrete limitation. It does not demonstrate greater analytical coverage,
faster interpretation or improved prediction compared with mmpdb or DataWarrior.

The BRAF and DRD2 examples show why a substituent effect must be attached to a
defined position. Hydroxyl addition and chlorine removal each changed direction
between tested positions. These distinctions were evident in assay-matched
comparisons without proposing a binding mechanism. The four-compound DRD2 cycle
made the chemical context explicit and showed how the direction of each
chlorination could be examined in both backgrounds.

The stereochemical audit was equally consequential. A scaffold grouping that
looked chemically coherent contained opposite and unassigned core configurations.
Removing those comparisons changed the DRD2 median effects and the number of
eligible pairs. This is a limitation of relying on scaffold membership and a
representative core drawing without inspecting per-compound stereochemistry.
The present analysis applied an additional mapped-core restriction; that
restriction must be made explicit whenever these results are reproduced or
used to assess the software.

The robustness checks also place useful limits on interpretation. A sign that
survives omission of individual compounds is less dependent on one example,
but it is not evidence that the effect transfers to a new chemical background.
The label permutations showed that the configured support criteria can also
be met by shuffled potency assignments. Descriptive grades should therefore
be read as summaries of what is present in the dataset, not as probabilities
that a design change will succeed. Assay precision and independent chemical
examples remain necessary when a decision depends on a small effect.

Activity cliffs pose a related problem. Their identification depends on a
structural representation and numerical cutoffs. In the BRAF example, the
reference call was near the potency boundary, while larger measured differences
were excluded by the similarity criterion. These findings favor showing
similarity, potency difference and the underlying compounds together. They
also support assessing prediction error locally rather than relying only on
a global model score, as emphasized by activity-cliff benchmarking studies [13, 14].
The present study did not perform an independent cliff-specific prediction trial.

SARscope's practical contribution is the ability to move from a structural
annotation to its supporting measurements within the same analysis. The chemist
can compare positions, inspect contradictory examples, change the experimental
subset and retain the evidence with the exported result. Established tools
already implement the underlying scaffold, R-group and matched-pair concepts.
The available comparison therefore supports a complementary interpretation
workflow, not superiority of the pairing algorithms. The additional stress-test
audit also distinguishes an effect retaining its sign from one retaining the
configured practical interpretation. In two examples, that distinction exposed
dependence on specific compounds that a sign-only stability check would miss.

The follow-up suggestions are intended to make an evidence gap actionable.
Completing a cycle or retesting a conflicting pair is a way to challenge a local
interpretation, not a guarantee that the suggested compound will improve potency.
Because the present cases contain one assay and publication each, the analysis
cannot establish that the conclusions survive new experimental protocols or
independent source documents. A frozen temporal or publication-disjoint evaluation,
with simple and context-aware matched-pair baselines, is needed before the
workbench can be presented as improving SAR transferability or design decisions.

Several limitations remain. The series were selected retrospectively and cover
only three target families. The largest-scaffold restriction omitted substantial
fractions of the assay cohorts. Compounds and transformations share chemical
backgrounds, and the source records do not establish independent experimental
replication. The permutation diagnostic assumes exchangeability and does not
account for all structure-dependent sources of variability. Complete cycles
were defined on one R-group mapping rather than an exhaustive nonadditivity
search. The source papers were not independently annotated to provide an external
SAR reference, and no blinded medicinal-chemist or usability comparison has
yet been completed. The supporting QSAR results are retrospective and do not
address prospective or cross-target prediction.

## 3. Conclusions

Assay-matched, stereochemically consistent comparisons revealed clear positional
differences in three published chemical series. Their interpretation changed
when stereochemistry, effect-size thresholds and the limits of descriptive
support were examined. SARscope provides an interactive way to inspect those
relationships and the measurements behind them. Its present role is to support
local SAR reasoning and the choice of follow-up experiments, rather than to
replace experimental validation or establish universal substituent rules.

## 4. Computational Methods

### Data selection and experimental scope

Activity records were taken from frozen ChEMBL release 37 datasets for BRAF
(CHEMBL5145), DRD2 (CHEMBL217) and estrogen receptor alpha (CHEMBL206). One
published assay was selected for each target: CHEMBL5046398, CHEMBL671074 and
CHEMBL829540, respectively [26–28]. Selection was retrospective and based on the
availability of populated chemical series, not on predictive-model performance.
These examples are illustrative rather than a representative sample of medicinal
chemistry projects.

The BRAF data describe an enzyme inhibition assay in a study of imidazothiazole
pan-RAF inhibitors [26]. The DRD2 data describe radioligand competition by
trans-phenylcyclopropylmethyl arylpiperazines [27]. The estrogen data are the
ER-alpha counter-assay from a study of ER-beta-selective ligands [28]. They are
not ER-beta measurements, and selectivity cannot be inferred from this assay alone.

We retained exact IC50 values in nM and removed records flagged for validity
concerns or potential duplication. Structures were standardized, and records
for the same standardized structure were combined using the median on the
logarithmic potency scale. Potency was expressed as pIC50 = −log10(IC50 in mol/L).
No conversion from IC50 to Ki was made. Records lacking a structured protein
variant annotation were retained as unannotated, not assumed to be wild type.
Assay descriptions and original activity identifiers were retained for inspection.

### Structural controls and positional comparisons

The largest Murcko scaffold group was selected within each assay, with lexical
tie-breaking and no use of potency in the selection. RDKit R-group decomposition
identified the variable positions [25]. An additional audit compared the
stereochemically annotated mapped core returned for every compound. Where more
than one core state was present, the largest state was retained before positional
effects were calculated. Unassigned and oppositely assigned core stereochemistry
were not pooled. The original scaffold grouping alone did not enforce that restriction.
The current workbench now applies the same core-state control, and its selected
cohorts were checked against the saved case-study analysis.

A pair was admitted when the two compounds had the same selected core and
identical substituents at every mapped position except one. Effects were
calculated as ΔpIC50 = pIC50(to) − pIC50(from), so a positive value indicates
increased potency. Fragment order was fixed before examining the activities.
Repeated measurements of a compound were combined within its assay. Copies
of the same identified record/value were deduplicated, but equal rounded values
from distinct record IDs were not assumed to be duplicates. Each molecular pair
contributed once to a transformation summary. We report pair
counts and the number of disjoint pairs, meaning pairs that do not share a
compound. These are distinct chemical comparisons, not independent biological
replicates.

Attachment labels refer to the specific decomposition. For the BRAF pendant
phenyl ring, R3 and R4 correspond to para and meta substitution relative to its
attachment to the heterocyclic core. On the DRD2 N-aryl ring, R2 and R4 correspond
to para and ortho substitution relative to the piperazine attachment. These
relationships were checked on the molecular graphs; the numerical R-labels are
not conventional atom numbering.

### Effect summaries and robustness checks

The default practical effect-size threshold was 0.5 pIC50 units, equivalent to
approximately a 3.2-fold concentration difference. It was not treated as an
estimate of assay precision. A directional summary required an absolute median
effect at least this large, at least 80% agreement in sign and no pair effects
exceeding the threshold in opposite directions. Two disjoint pairs were required
for the software's moderate-support label; three disjoint pairs across at least
two remaining-substituent contexts were required for its strong-support label.
These labels describe data support and consistency, not statistical confidence.

We repeated the summaries at thresholds of 0.25, 0.75 and 1.0 units. For
individual-compound sensitivity, all pairs involving an omitted compound were
removed before recalculating the median. The resulting range was used to assess
dependence on a single compound; it is not a confidence interval.

An exploratory null diagnostic shuffled the compound pIC50 values 5,000 times
within each selected series while leaving the pair network unchanged. A shuffled
compound received the same value in all pairs containing it, preserving the
dependence caused by compound reuse. Two-sided permutation P values for median
effects were adjusted across all 191 transformations using the Benjamini–Hochberg
procedure [29]. The number of transformations meeting the descriptive support
criteria was also compared with the shuffled counts. This retrospective diagnostic
assumes exchangeable potency labels; it does not model assay error or provide
guaranteed false-discovery control for dependent chemical comparisons.

### Complete substitution cycles and activity cliffs

For two variable positions, complete four-compound substitution cycles were
identified when every combination was represented by one compound with the
same selected core and unchanged substituents elsewhere. Ambiguous cells were
excluded. Differences between conditional substituent effects were expressed
as ΔΔpIC50 [11]. These IC50 contrasts were not converted into binding free energies.

Activity-cliff calls were examined for the already mapped pairs using Morgan
fingerprints of radius 2, 2,048 bits and chirality disabled, using a circular
fingerprint representation [30]. The reference criterion
required Tanimoto similarity above 0.90 and an absolute potency difference above
2.0 pIC50 units. The effect of changing either cutoff was inspected. This was
not an exhaustive activity-cliff survey of the complete target datasets.

### Software comparison and supporting modeling results

Single-cut pair coverage was compared with mmpdb 3.1.4 on identical curated assay
cohorts [3]. The comparison used the same cut pattern and a ten-heavy-atom limit
for variable fragments. Compound-pair membership, rather than fragment-label
spelling, was compared. Timing was not used as an outcome because the mmpdb run
included process startup and file operations whereas SARscope was called within
an existing process.

An earlier random-forest regression benchmark on the full three-target datasets
was retained as supporting analysis. It used one scaffold-held-out split per
target and a training-set median baseline. Bootstrap intervals were calculated
from the saved predictions by resampling test scaffold groups. Those results
describe the particular retrospective splits, not prospective performance or
independent validation of the SAR examples. Details are provided in the
Supporting Information.

### Evidence stress tests and follow-up measurements

A stress test retains the selected position, directed fragments, endpoint and
variant-annotation stratum. One compound omission removes every comparison
incident to that compound. A chemical-context omission removes a combination of
the unchanged R-groups. Assay and publication omissions remove the relevant raw
records and recompute surviving shared-assay medians; there is no fallback to
pooled compound potency. Omissions leaving no eligible pairs are recorded as
unevaluable and are not counted as evidence that the effect persists.

The panel repeats the descriptive directional criterion over user-selected
effect thresholds and reports individual assay/context strata. Default bounds
are 100 omissions per axis, 5,000 cycle-candidate visits and 12 measurement
suggestions. The omission bound is adjustable in the interface. Any truncated
omission/cycle search or source map is labeled partial; displayed suggestions
are a bounded set, not an exhaustive synthesis plan.

For complete four-compound cycles, each analogue cell must contain one compound
and all four must share an assay, endpoint and recorded variant context.
Three measured corners can instead identify an existing analogue needing a
measurement in that assay or a missing structural combination. A proposed
structure is assembled only when the source core and mapped fragments exactly
reconstruct its isomeric SMILES; otherwise only the substitution pattern is shown.
The procedure does not enumerate unspecified stereochemistry, predict potency
or assess synthetic feasibility. Opposing assay effects for the same molecular
pair can prompt a recommendation to remeasure both compounds together in a
defined protocol.

The current workbench was evaluated from the hash-checked frozen assay snapshots,
not from live downloads. Its scoped pair identities and deltas were compared
with the saved scientific case-study results and a separate exhaustive enumeration
of all two-compound combinations. The enumeration shares RDKit's decomposition
and therefore verifies implementation, not independent chemical truth. The
stress-test audit examined all 17 transformations satisfying the default
repeated-comparison criteria. Tests also covered copied records, mismatched
assay/variant contexts, ambiguous cycle cells, bounded computation and invalidation
of stale browser results.

AI-assisted drafting and programming tools were used in software and manuscript
preparation. These tools were not expert reviewers or sources of experimental
validation. The author is responsible for reviewing the code, calculations,
references and final interpretation.

## Acknowledgements

Any acknowledgements require author confirmation and permission from the people
named. AI assistance is disclosed in the Computational Methods section; no AI
system is credited with authorship.

## Conflict of Interest

Author confirmation required before submission.

## Author Contributions

The final author list and contributions require author confirmation before
submission.

## Funding

Author confirmation required before submission.

## Data Availability Statement

The source code [31] and browser application [32] are available at the cited
locations. The manuscript package contains frozen assay snapshots, original
activity identifiers, analysis scripts, machine-readable results and
input/source/output hashes, together with the current workbench audit and the
comparator outputs.

SARscope is distributed under the MIT licence and runs on Python-compatible
platforms; numerical reproduction was checked on Linux x86_64 with Python 3.13,
and cross-platform numerical equivalence was not evaluated.

A public, versioned archive and DOI for the tested revision are not yet
assigned. The full benchmark snapshots and saved predictions exist only in a
prepared local reproducibility archive and must be publicly deposited if the
supporting modelling results are retained at submission. Redistributed ChEMBL
records remain under CC BY-SA 3.0, separate from the MIT code licence.

## Figure Legends

Figure 1: `figures/sar-stress-workflow.svg`. Workflow for challenging an observed
positional SAR conclusion. The chemist selects a transformation, inspects the
controlled assay evidence, tests sensitivity to retained records and thresholds,
and examines contexts or missing-cycle measurements. The exported evidence card
preserves the observations, decisions and limitations. Follow-up suggestions are
unmeasured hypotheses, not predicted potency gains.

Figure 2: `results/braf/sar-map.svg`. Interactive positional SAR map for the BRAF
series. The shared structure, variable positions and example transformations
are shown together. Labels describe the measured comparisons and the software's
support criteria, not statistical significance or a binding mechanism. Selecting
a position in the application opens its supporting compounds and assay records.

Figure 3: `scientific-results/position-effects.svg`. Individual pair effects for
selected BRAF, DRD2 and ER-alpha transformations. Circles show measured pair
differences; diamonds show their medians. Thick pale segments show the range of
medians after omitting individual compounds and all their incident pairs, not
confidence intervals. The grey band marks changes smaller than 0.5 pIC50 units.
The number of pairs is shown for each transformation. All displayed DRD2 pairs
have the same selected stereochemical core. The displayed examples were chosen
to illustrate positional contrasts and repeated comparisons, not as an exhaustive
ranking of transformations.

Figure 4: `scientific-results/drd2-chlorine-structures.svg`. Complete DRD2 para/ortho
chlorination cycle within the selected (S,S) core. Other mapped positions are
unchanged and all measurements come from CHEMBL671074. Structures and original
IC50 values are shown; the doubly chlorinated compound has retained values of
65 and 93 nM. Arrows show differences calculated from median pIC50. The
−0.369-unit contrast describes a modest departure
from additivity, not a binding free energy or evidence of biological cooperativity.

## References

[1] E. Griffen, A. G. Leach, G. R. Robb, D. J. Warner, *J. Med. Chem.* **2011**, *54*, 7739–7750.
[2] J. Hussain, C. Rea, *J. Chem. Inf. Model.* **2010**, *50*, 339–348.
[3] A. Dalke, J. Hert, C. Kramer, *J. Chem. Inf. Model.* **2018**, *58*, 902–910.
[4] N. M. O’Boyle, J. Boström, R. A. Sayle, A. Gill, *J. Med. Chem.* **2014**, *57*, 2704–2713.
[5] B. Zdrazil, E. Felix, F. Hunter, E. J. Manners, J. Blackshaw, S. Corbett, M. de Veij, H. Ioannidis, D. M. Lopez, J. F. Mosquera, M. P. Magarinos, N. Bosc, R. Arcila, T. Kizilören, A. Gaulton, A. P. Bento, M. F. Adasme, P. Monecke, G. A. Landrum, A. R. Leach, *Nucleic Acids Res.* **2024**, *52*, D1180–D1192.
[6] G. Papadatos, A. Gaulton, A. Hersey, J. P. Overington, *J. Comput.-Aided Mol. Des.* **2015**, *29*, 885–896.
[7] I. Smit, M. F. Adasme, E. Manners, S. Corbett, N. Bosc, H.-M.-A. Do, A. R. Leach, N. M. O’Boyle, B. Zdrazil, *J. Cheminform.* **2026**, *18*.
[8] C. Kramer, T. Kalliokoski, P. Gedeck, A. Vulpetti, *J. Med. Chem.* **2012**, *55*, 5165–5173.
[9] T. Kalliokoski, C. Kramer, A. Vulpetti, P. Gedeck, *PLoS One* **2013**, *8*, e61007.
[10] J. Nelen, H. Pérez-Sánchez, H. De Winter, D. Van Rompaey, *J. Cheminform.* **2025**, *17*.
[11] C. Kramer, J. E. Fuchs, K. R. Liedl, *J. Chem. Inf. Model.* **2015**, *55*, 483–494.
[12] G. M. Maggiora, *J. Chem. Inf. Model.* **2006**, *46*, 1535–1535.
[13] D. van Tilborg, A. Alenicheva, F. Grisoni, *J. Chem. Inf. Model.* **2022**, *62*, 5938–5951.
[14] D. van Tilborg, A. Alenicheva, F. Grisoni, *J. Chem. Inf. Model.* **2023**, *63*, 2266–2266.
[15] T. Sander, J. Freyss, M. von Korff, C. Rufener, *J. Chem. Inf. Model.* **2015**, *55*, 460–473.
[16] D. K. Agrafiotis, M. Shemanarev, P. J. Connolly, M. Farnum, V. S. Lobanov, *J. Med. Chem.* **2007**, *50*, 5926–5937.
[17] M. González-Medina, O. Méndez-Lucio, J. L. Medina-Franco, *J. Chem. Inf. Model.* **2017**, *57*, 397–402.
[18] E. E. Abreu-Martínez, K. Martinez-Mayorga, G. Merino, *J. Cheminform.* **2026**, *18*.
[19] A. R. Bradley, I. D. Wall, D. V. S. Green, C. M. Deane, B. D. Marsden, *J. Chem. Inf. Model.* **2014**, *54*, 2636–2646.
[20] C. Kramer, *J. Chem. Inf. Model.* **2019**, *59*, 4034–4042.
[21] L. Guasch, N. Maeder, J. G. Cumming, C. Kramer, *J. Comput.-Aided Mol. Des.* **2024**, *38*.
[22] M. D. Wilkinson, M. Dumontier, I. J. Aalbersberg, G. Appleton, M. Axton, A. Baak, N. Blomberg, J.-W. Boiten, L. B. da Silva Santos, P. E. Bourne, J. Bouwman, A. J. Brookes, T. Clark, M. Crosas, I. Dillo, O. Dumon, S. Edmunds, C. T. Evelo, R. Finkers, A. Gonzalez-Beltran, A. J. G. Gray, P. Groth, C. Goble, J. S. Grethe, J. Heringa, P. A. C. ’t Hoen, R. Hooft, T. Kuhn, R. Kok, J. Kok, S. J. Lusher, M. E. Martone, A. Mons, A. L. Packer, B. Persson, P. Rocca-Serra, M. Roos, R. van Schaik, S.-A. Sansone, E. Schultes, T. Sengstag, T. Slater, G. Strawn, M. A. Swertz, M. Thompson, J. van der Lei, E. van Mulligen, J. Velterop, A. Waagmeester, P. Wittenburg, K. Wolstencroft, J. Zhao, B. Mons, *Sci. Data* **2016**, *3*.
[23] C. T. Hoyt, B. Zdrazil, R. Guha, N. Jeliazkova, K. Martinez-Mayorga, E. Nittinger, *J. Cheminform.* **2023**, *15*.
[24] G. W. Bemis, M. A. Murcko, *J. Med. Chem.* **1996**, *39*, 2887–2893.
[25] RDKit: open-source cheminformatics, https://www.rdkit.org/, accessed 13 October 2026.
[26] M. S. Abdel-Maksoud, M. I. El-Gamal, B. S. Lee, M. M. Gamal El-Din, H. R. Jeon, D. Kwon, U. M. Ammar, K. I. Mersal, E. M. H. Ali, K.-T. Lee, K. H. Yoo, D. K. Han, J. K. Lee, G. Kim, H. S. Choi, Y. J. Kwon, K. H. Lee, C. H. Oh, *J. Med. Chem.* **2021**, *64*, 6877–6901.
[27] X. Zhang, K. Hodgetts, S. Rachwal, H. Zhao, J. W. F. Wasley, K. Craven, R. Brodbeck, A. Kieltyka, D. Hoffman, M. D. Bacolod, B. Girard, J. Tran, A. Thurkauf, *J. Med. Chem.* **2000**, *43*, 3923–3932.
[28] M. S. Malamas, E. S. Manas, R. E. McDevitt, I. Gunawan, Z. B. Xu, M. D. Collini, C. P. Miller, T. Dinh, R. A. Henderson, J. C. Keith, H. A. Harris, *J. Med. Chem.* **2004**, *47*, 5021–5040.
[29] Y. Benjamini, Y. Hochberg, *J. R. Stat. Soc. B* **1995**, *57*, 289–300.
[30] D. Rogers, M. Hahn, *J. Chem. Inf. Model.* **2010**, *50*, 742–754.
[31] Y. Boulaamane, SARscope: source-code repository, https://github.com/yboulaamane/sarscope, accessed 13 October 2026.
[32] Y. Boulaamane, SARscope: browser application, https://sarscope.streamlit.app/, accessed 13 October 2026.

# SARscope: interactive analysis of position-dependent structure–activity relationships

Yassir Boulaamane

Correspondence: yassirxmail@gmail.com

Manuscript draft. Affiliation and author declarations require confirmation.

## Abstract

### Background

Comparisons between close analogues help medicinal chemists choose modifications
during lead optimization. Their interpretation depends on the attachment
position, the remaining structure and the experimental conditions. We developed
SARscope to examine these factors together and make the measurements behind a
structure–activity relationship (SAR) accessible through an interactive interface.

### Methods

Three published IC50 datasets were reconstructed from a frozen ChEMBL release:
BRAF inhibitors, dopamine D2 receptor ligands and an estrogen receptor alpha
counter-assay. Comparisons were restricted to one assay, an identical
stereochemically annotated core and a change at one mapped position. We examined
individual pair effects, sensitivity to the effect-size threshold and omission
of individual compounds. Potency-label permutations assessed whether descriptive
support criteria alone distinguished the observed data from shuffled assignments.

### Results

The selected series contained 16, 23 and 53 compounds and yielded 263 positional
comparisons. Hydroxyl addition to the BRAF series was unfavorable at the para
position but favorable at the meta position. Chlorine removal from the DRD2
N-aryl ring was favorable at the para position and unfavorable at the ortho
position. Seventeen transformations met the configured repeated-comparison
criteria and retained their direction after individual-compound omissions.
However, their number was not unusual under the permutation diagnostic, and
none of the 191 transformation-level tests remained below 0.05 after adjustment
for multiple comparisons. These results distinguish observed local SAR from
statistically established, transferable design rules.

### Conclusions

SARscope links substituent effects to attachment position, stereochemistry and
assay evidence. The examples demonstrate the value of inspecting local
comparisons rather than interpreting a fragment or an activity-cliff flag in
isolation. The current evidence supports exploratory SAR analysis, not prospective
potency prediction or mechanistic conclusions.

Keywords: structure–activity relationships; medicinal chemistry; matched molecular
pairs; activity cliffs; stereochemistry; ChEMBL.

## Introduction

Medicinal chemists use comparisons between related compounds to decide which
chemical changes to retain during lead optimization. A well-controlled comparison
changes one part of the molecule while keeping the remainder of the structure
and the assay conditions constant. Matched molecular pair analysis formalizes
this reasoning and allows recurring transformations to be examined across larger
collections [1]. Public resources such as ChEMBL make these comparisons possible
well beyond a single project [2].

The main difficulty is not calculating a potency difference. It is deciding
what that difference means. A substituent can be favorable at one attachment
position and unfavorable at another. Its effect may also depend on groups
elsewhere in the molecule. Measurements from different assays, endpoints or
stereoisomers can obscure these distinctions. Nonadditivity studies have shown
why substituent contributions and experimental uncertainty must be considered
together [3]. A short statement such as “chlorine improves potency” can therefore
be less informative than the individual comparisons from which it was derived.

DataWarrior already provides chemical-space visualization, R-group analysis and
activity-cliff inspection, while mmpdb offers extensive transformation analysis
[1,4]. SARscope builds on these established ideas. Its purpose is to place the
shared structure, changed position, potency measurements and source assay in one
interactive view. A chemist can select a position, inspect the compounds behind
an apparent trend and distinguish repeated observations from isolated examples.
Scaffold analysis, activity-cliff inspection and modeling remain separate choices
rather than steps run automatically on every dataset.

Here we examine whether this approach produces interpretable local SAR in three
published chemical series. The analysis focuses on positional effects, the
comparability of the underlying compounds and the stability of descriptive
summaries. We also assess the limits of threshold-based support labels and
activity-cliff calls. The study does not introduce a new pairing algorithm or
claim superiority in prospective compound design.

## Materials and methods

### Data selection and experimental scope

Activity records were taken from frozen ChEMBL release 37 datasets for BRAF
(CHEMBL5145), DRD2 (CHEMBL217) and estrogen receptor alpha (CHEMBL206). One
published assay was selected for each target: CHEMBL5046398, CHEMBL671074 and
CHEMBL829540, respectively [5–7]. Selection was retrospective and based on the
availability of populated chemical series, not on predictive-model performance.
These examples are illustrative rather than a representative sample of medicinal
chemistry projects.

The BRAF data describe an enzyme inhibition assay in a study of imidazothiazole
pan-RAF inhibitors [5]. The DRD2 data describe radioligand competition by
trans-phenylcyclopropylmethyl arylpiperazines [6]. The estrogen data are the
ER-alpha counter-assay from a study of ER-beta-selective ligands [7]. They are
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
identified the variable positions [9]. An additional audit compared the
stereochemically annotated mapped core returned for every compound. Where more
than one core state was present, the largest state was retained before positional
effects were calculated. Unassigned and oppositely assigned core stereochemistry
were not pooled. This audit was part of the present case-study analysis; the
original scaffold grouping alone did not enforce that restriction.

A pair was admitted when the two compounds had the same selected core and
identical substituents at every mapped position except one. Effects were
calculated as ΔpIC50 = pIC50(to) − pIC50(from), so a positive value indicates
increased potency. Fragment order was fixed before examining the activities.
Repeated measurements of a compound were combined within its assay, and each
molecular pair contributed once to a transformation summary. We report pair
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
procedure [10]. The number of transformations meeting the descriptive support
criteria was also compared with the shuffled counts. This retrospective diagnostic
assumes exchangeable potency labels; it does not model assay error or provide
guaranteed false-discovery control for dependent chemical comparisons.

### Complete substitution cycles and activity cliffs

For two variable positions, complete four-compound substitution cycles were
identified when every combination was represented by one compound with the
same selected core and unchanged substituents elsewhere. Ambiguous cells were
excluded. Differences between conditional substituent effects were expressed
as ΔΔpIC50 [3]. These IC50 contrasts were not converted into binding free energies.

Activity-cliff calls were examined for the already mapped pairs using Morgan
fingerprints of radius 2, 2,048 bits and chirality disabled. The reference criterion
required Tanimoto similarity above 0.90 and an absolute potency difference above
2.0 pIC50 units. The effect of changing either cutoff was inspected. This was
not an exhaustive activity-cliff survey of the complete target datasets.

### Software comparison and supporting modeling results

Single-cut pair coverage was compared with mmpdb 3.1.4 on identical curated assay
cohorts [1]. The comparison used the same cut pattern and a ten-heavy-atom limit
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
Supplementary Information.

## Results

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
about a 27-fold potency increase. The individual changes and their spread are
shown in Figure 2, alongside the positional map in Figure 1. Removing individual
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
removal and −1.37 to −0.68 for ortho removal (Figure 2).

A complete four-compound cycle placed both positions in the same chemical
background (Figure 3). The unsubstituted analogue, CHEMBL341497, had an IC50 of
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
directions persisted after compound omission (Figure 2).

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

## Discussion

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
a global model score, as emphasized by activity-cliff benchmarking studies [8].
The present study did not perform an independent cliff-specific prediction trial.

SARscope's practical contribution is the ability to move from a structural
annotation to its supporting measurements within the same analysis. The chemist
can compare positions, inspect contradictory examples, change the experimental
subset and retain the evidence with the exported result. Established tools
already implement the underlying scaffold, R-group and matched-pair concepts.
The available comparison therefore supports a complementary interpretation
workflow, not superiority of the pairing algorithms.

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

## Conclusions

Assay-matched, stereochemically consistent comparisons recovered clear positional
differences in three published chemical series. Their interpretation changed
when stereochemistry, effect-size thresholds and the limits of descriptive
support were examined. SARscope provides an interactive way to inspect those
relationships and the measurements behind them. Its present role is to support
local SAR reasoning and the choice of follow-up experiments, rather than to
replace experimental validation or establish universal substituent rules.

## Scientific contribution

This study connects interactive positional SAR with explicit assay and
stereochemical controls. It demonstrates, using published measurements, that
consistent local effects and threshold-defined activity cliffs are not equivalent
to calibrated evidence for a transferable design rule. The contribution is an
evidence-linked interpretation workflow and its evaluated limits, not a new
pairing or prediction algorithm.

## Data and software availability

SARscope source code is available at https://github.com/yboulaamane/sarscope
under the MIT license; the browser application is at https://sarscope.streamlit.app/.
The frozen assay snapshots, original identifiers, analysis scripts and
machine-readable results are supplied with this manuscript package. ChEMBL-derived
data carry the CC BY-SA 3.0 license. The full benchmark snapshots and predictions
are included in the locally prepared reproducibility archive, but a publicly
deposited versioned archive and DOI are not yet available. Public deposition
is required before submission. Detailed procedures and software versions are
given in the Supplementary Information.

## Declarations

### Funding and competing interests

To be supplied and confirmed by the author before submission.

### Author contributions

The final author list, affiliation and contributions require confirmation.

### Assistance in manuscript and software preparation

AI-assisted drafting and programming tools were used during preparation.
The final text, calculations, references and interpretations require author
review and approval; no AI system is credited with authorship.

## Figure legends

Figure 1: `results/braf/sar-map.svg`. Interactive positional SAR map for the BRAF
series. The shared structure, variable positions and example transformations
are shown together. Labels describe the measured comparisons and the software's
support criteria, not statistical significance or a binding mechanism. Selecting
a position in the application opens its supporting compounds and assay records.

Figure 2: `scientific-results/position-effects.svg`. Individual pair effects for
selected BRAF, DRD2 and ER-alpha transformations. Circles show measured pair
differences; diamonds show their medians. Thick pale segments show the range of
medians after omitting individual compounds and all their incident pairs, not
confidence intervals. The grey band marks changes smaller than 0.5 pIC50 units.
The number of pairs is shown for each transformation. All displayed DRD2 pairs
have the same selected stereochemical core. The displayed examples were chosen
to illustrate positional contrasts and repeated comparisons, not as an exhaustive
ranking of transformations.

Figure 3: `scientific-results/drd2-chlorine-structures.svg`. Complete DRD2 para/ortho
chlorination cycle within the selected (S,S) core. Other mapped positions are
unchanged and all measurements come from CHEMBL671074. Structures and original
IC50 values are shown; the doubly chlorinated compound has retained values of
65 and 93 nM. Arrows show differences calculated from median pIC50. The
−0.369-unit contrast describes a modest departure
from additivity, not a binding free energy or evidence of biological cooperativity.

## References

1. Dalke A, Hert J, Kramer C. mmpdb: An Open-Source Matched Molecular Pair Platform
   for Large Multiproperty Data Sets. J Chem Inf Model. 2018;58:902–910.
   https://doi.org/10.1021/acs.jcim.8b00173.
2. Zdrazil B et al. The ChEMBL Database in 2023: a drug discovery platform spanning
   multiple bioactivity data types and time periods. Nucleic Acids Res.
   2024;52:D1180–D1192. https://doi.org/10.1093/nar/gkad1004.
3. Kramer C, Fuchs JE, Liedl KR. Strong Nonadditivity as a Key Structure–Activity
   Relationship Feature: Distinguishing Structural Changes from Assay Artifacts.
   J Chem Inf Model. 2015;55:483–494. https://doi.org/10.1021/acs.jcim.5b00018.
4. Sander T, Freyss J, von Korff M, Rufener C. DataWarrior: an open-source program
   for chemistry aware data visualization and analysis. J Chem Inf Model.
   2015;55:460–473. https://doi.org/10.1021/ci500588j.
5. Abdel-Maksoud MS et al. Discovery of New Imidazo[2,1-b]thiazole Derivatives as
   Potent Pan-RAF Inhibitors with Promising In Vitro and In Vivo Anti-melanoma
   Activity. J Med Chem. 2021;64:6877–6901.
   https://doi.org/10.1021/acs.jmedchem.1c00230.
6. Zhang X et al. trans-1-[(2-Phenylcyclopropyl)methyl]-4-arylpiperazines: mixed
   dopamine D(2)/D(4) receptor antagonists as potential antipsychotic agents.
   J Med Chem. 2000;43:3923–3932. https://doi.org/10.1021/jm990562x.
7. Malamas MS et al. Design and synthesis of aryl diphenolic azoles as potent and
   selective estrogen receptor-beta ligands. J Med Chem. 2004;47:5021–5040.
   https://doi.org/10.1021/jm049719y.
8. van Tilborg D, Alenicheva A, Grisoni F. Exposing the Limitations of Molecular
   Machine Learning with Activity Cliffs. J Chem Inf Model. 2022;62:5938–5951.
   https://doi.org/10.1021/acs.jcim.2c01073. Correction: 2023;63:2266.
   https://doi.org/10.1021/acs.jcim.3c00423.
9. RDKit: Open-source cheminformatics. https://www.rdkit.org/.
   Accessed 13 October 2026.
10. Benjamini Y, Hochberg Y. Controlling the False Discovery Rate: A Practical
    and Powerful Approach to Multiple Testing. J R Stat Soc B. 1995;57:289–300.
    https://doi.org/10.1111/j.2517-6161.1995.tb02031.x.

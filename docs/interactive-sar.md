# Interactive SAR interpretation

The SAR map summarizes the scaffold series produced by RDKit R-group
decomposition. It links a drawing to substitutions, potency distributions,
matched compounds, property changes and retained ChEMBL measurements.

## Workflow

1. Curate a coherent target/endpoint subset.
2. Click **Explore SAR**, then **Prepare SAR series**. This explicitly prepares
   the shared scaffolds and R-groups together. Series setup options let you
   change the minimum size (default eight compounds) and the number of series
   (default ten). The separate analysis buttons remain available in an expander.
3. Select a series and, when available, its **Core stereochemical state**.
   States are ranked by compound count, never potency; their drawings identify
   structures, not activity classes. Under **Refine the SAR evidence**, select endpoints, assays,
   origins, publication years and variant annotations. Choose the minimum effect
   threshold, default 0.5 log units.
4. Press **Build interactive SAR map**. Filters recompute measurement medians
   while retaining the original core/position labels. Changing filters makes the
   previous map stale until rebuilt.
5. Click a core R-site or side annotation, or use **Inspect SAR position**.
   The strongest supported comparison is selected first. A plain-language
   summary translates its median log-unit change into a concentration ratio.
   Select another transformation to inspect its deltas and underlying records.
6. Download the SVG or the evidence ZIP. The ZIP includes retained measurements,
   structures, pairs, summaries, annotations, selected filters and record IDs.
7. Under a chosen transformation, click **Run SAR stress test**. Inspect
   **Does the effect persist?**, **Where does it change?**, and **What to measure
   next**. The separate evidence-card ZIP includes the selected claim, matched
   pairs, sensitivity checks, contexts, cycles, suggested measurements, retained
   records, structures, settings, RDKit version and SHA-256 file hashes.

## Comparison rules

For each R-position, pairs must share the identical mapped isomeric core and
every other R-group in the decomposition. Assigned enantiomers and unassigned
core stereochemistry are separate states; matching CIP labels alone is not
used, since substituents can change CIP priorities. The browser lets users
choose the state. CLI maps default to the largest state and record exclusions.
Fragment order is canonical, so the displayed delta is always **to minus from**.
The shared-assay tier requires exact `=` measurements for both molecules within
the same ChEMBL assay ID, endpoint and normalized variant annotation. Its delta
uses the median of each compound's exact measurements in that context, removing
repeated copies of the same record ID/value. Equal rounded values from different
record IDs are not automatically discarded: equality does not prove duplication.
All original records remain traceable, and extra records or assay measurements
do not give a molecular pair extra votes in the support summary. Different record
IDs are not proof of independent experiments. Missing
annotation stays unreported; neither PubChem text nor assay prose is interpreted
as a structured variant identifier here.

When no shared context exists, the pair uses curated molecule medians and is
marked **pooled / unmatched**. These comparisons cannot receive moderate or
strong evidence grades. Descriptive substituent medians can mix other R-groups
and must be read separately from controlled comparisons. The user can select
one endpoint and assay to make their distributions more coherent.

Summary tables keep endpoint and variant strata separate. When a molecular pair
has several shared assays, its assay deltas are reduced to one median for the
pair, giving it one vote. Opposing assay deltas beyond the effect threshold
mark the transformation mixed. All assay rows remain in the pair table and ZIP.

## Colors and evidence grades

These are descriptive heuristics for the observed subset, not statistical
significance, causal attribution or predictions.

| Grade | Rule |
|---|---|
| Mixed | Median magnitude below threshold, less than 80% sign agreement, or opposing assay deltas beyond threshold |
| Limited | Sparse evidence, unmatched assays, or a truncated comparison run |
| Moderate | Shared assay evidence, consistent direction and at least two pairs sharing no compounds |
| Strong | Moderate conditions, at least three pairs sharing no compounds and at least two other-R-group contexts |

Pairs sharing no compounds are counted using a deterministic greedy matching:
the count is conservative and may be lower than the maximum possible matching.
Assay count is reported separately and does not imply independent experiments
or laboratories. Green/red indicate positive/negative moderate or strong
effects; amber indicates mixed/small effects; grey indicates limited evidence.
One representative transformation is annotated at each position; the panel
retains all comparisons. Middle-50% delta ranges are distribution summaries,
not confidence intervals. A 20,000-row limit bounds each map; hitting it caps
otherwise positive evidence grades at limited and is recorded in the export.

## Structural scope

These support grades are descriptive heuristics, not calibrated probabilities or
expert-validated recommendations. Green/red mean supported measured changes,
not independent biological replication. See the publication evaluation protocol
for the pending blinded review. The separate single-cut MMP explorer currently
omits implicit-H substitution pairs, unlike the scaffold map's R-group comparisons.

RDKit assigns the R-labels; symmetry can make chemically equivalent sites
exchangeable. A core shared by all molecules cannot establish which invariant
atoms are essential. Fixed Murcko series do not directly test changes that
alter the core, ring size or linker topology. The separate single-cut matched
molecular-pair explorer covers eligible transformations across series, with
compound filters and pagination. It shows pooled molecule deltas and the assay
records needed to assess comparability.

The map does not infer electron-donation mechanisms, binding interactions,
linker-length optima, or essential pharmacophores without measured comparisons.
CLI reports include maps and controlled comparison tables for full curated
series with one core state per map; browser-specific filter choices are exported
in the separate map ZIP.

## SAR stress-test workbench

Stress tests are opt-in and use the curated records already in memory. They
make no network requests, fit no models and perform no docking.

- **Leave one compound out:** remove every incident comparison. Pair reuse is
  retained in the graph, not treated as independent replication.
- **Leave one chemical context out:** remove a combination of unchanged R-groups.
- **Leave one assay/publication out:** remove its underlying records, then
  recompute medians and surviving shared-assay pairs. There is no fallback to
  pooled potency. A single-assay series becomes unevaluable on assay omission.
  Missing document IDs are reported; publication year is not a document ID.
- **Threshold sensitivity:** rerun the descriptive directional criterion over
  user-selected effect thresholds, always including the map's current threshold.
- **Context effects:** show individual assay / unchanged-R-group strata instead
  of claiming that a pooled median transfers to every context.
- **Matched cycles:** identify four unique analogue cells differing at two
  positions, within one assay/endpoint/variant stratum. Report the second
  substitution effect minus the first, with explicit context ordering. Ambiguous
  multi-compound cells are excluded rather than averaged.

The verdict reports direction retention in the tested subsets, dependence on
the retained evidence, mixed/small effects, sparse support or insufficient
shared-assay evidence. It is not a p-value, calibrated probability, biological
replication claim or validated recommendation. Empty omissions are reported
separately and never counted as evidence that an effect persists.

The default bound is 100 omissions per axis (user-selectable 25–200), 5,000
cycle-candidate visits and 12 displayed measurement suggestions. Any truncated
omission/cycle search or source map is explicitly labeled partial. Charts are
display-bounded; tables and exports retain all computed rows.

### What to measure next

Three observed, same-assay corners can suggest a missing fourth analogue to
test whether neighbouring substitutions change the observed effect. An existing
series member missing a measurement in that assay is proposed for measurement,
not called a new molecule. For structurally missing cells, RDKit can join mapped
core/R-group fragments only after the source molecule round-trips exactly with
its isomeric SMILES; otherwise only the proposed substitution pattern is shown.
Opposing assay effects for the same pair prioritize retesting that pair together
in a defined protocol. A sparse pair can prompt a reproducibility check.

Suggestions include supporting compound and record IDs but **no predicted
potency or synthetic feasibility score**. A missing analogue may already exist
outside the selected series; it is not a global novelty claim. Neither cycle
interaction deltas nor threshold sensitivity establish effects beyond assay
noise or binding mechanisms. The scientific validation still needed is a frozen,
publication-disjoint/temporal evaluation against later evidence, with simple and
context-aware matched-pair baselines and explicit abstention/coverage reporting.

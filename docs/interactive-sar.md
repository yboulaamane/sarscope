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
3. Select a series. Under **Refine the SAR evidence**, select endpoints, assays,
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

## Comparison rules

For each R-position, pairs must share every other R-group in the decomposition.
Fragment order is canonical, so the displayed delta is always **to minus from**.
The shared-assay tier requires exact `=` measurements for both molecules within
the same ChEMBL assay ID, endpoint and normalized variant annotation. Its delta
uses the median of each compound's measurements in that context. Missing
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
series; browser-specific filter choices are exported in the separate map ZIP.

# Product priorities: complementing ChEMBL

SARscope should help a scientist answer: **Which chemical series should I
pursue, what evidence supports that choice, and what experiment would reduce
the uncertainty?** Treat ChEMBL as the evidence source and Open Targets as
biological context, with links back to their records.

ChEMBL already supports name search, compound similarity/substructure search,
and extensive assay metadata. Search makes SARscope easier to enter; its
competitive value must come from the analysis and decisions after retrieval.
Open Targets already supplies disease evidence, tractability, and target
prioritisation. SARscope should connect these capabilities to chemistry rather
than recreate a comprehensive biological knowledge portal.

## Available now

- ID and gene/protein discovery, with explicit species/type confirmation.
- Disease term → direct associated gene → reviewed UniProt accession → human single-protein ChEMBL target.
- User-triggered curation, chemical space, scaffolds, R-groups, matched pairs, cliffs, ML, and descriptor explanation.
- Continuous regression and classification, scaffold/time/source-origin/random validation, held-out cliff-compound errors, ROC/PR curves, per-class diagnostics, and confusion matrices.
- Original ChEMBL source IDs for integrated PubChem BioAssay and BindingDB rows,
  with cross-origin compound overlap and potency-disagreement diagnostics.
- Assay-context drill-down, optional assay-ID and target-confidence filtering, and
  a measurement-level time split that keeps post-cutoff records out of early labels.
- Streamlit selectivity comparison with explicit missingness and assay-context
  labels; CLI comparison remains available.
- Uploaded-compound prediction and editable, property-filtered diverse shortlists,
  with invalid-row reasons, nearest training analogues and domain flags.
- Directional matched-pair summaries with support and shared-assay indicators.
- Training-CV empirical residual bands, observed held-out coverage, error versus
  novelty, split manifests, and retained-measurement exports.

Disease discovery is deliberately bounded to 10 search terms and 25 associated
genes. Counts expose truncation. It is a starting point, not an exhaustive
target ranking or a claim that an associated gene has suitable compound data.

## Delivery status and remaining work

| Priority | Deliverable | Status and next validation step |
|---|---|---|
| 1 | Assay comparability and dataset readiness | Measurement context, confidence filtering and drill-down are implemented. Next: assay-homogeneous subset presets and coverage warnings. |
| 2 | Selectivity workspace | Comparison, matrix missingness and context labels are implemented. Next: paired-assay-aware uncertainty and explicit experimental design prompts. |
| 3 | SAR transformations | Directional support, context counts and contradictory effects are implemented. Next: document-level independence and direct links for each supporting pair. |
| 4 | Upload and prioritize | Standardisation, reasons for invalid rows, nearest analogues and an editable diverse shortlist are implemented. Next: stronger batch-level deduplication and alternative scoring objectives. |
| 5 | Uncertainty and prospective evaluation | A training-only empirical band, held-out coverage, novelty/error and leakage-safe time split are implemented. The band is not calibrated conformal uncertainty; validate coverage by target family and split before stronger claims. |
| 6 | Reproducibility and benchmarks | Provenance, measurement exports and split manifests are implemented. A fixed BRAF/DRD2/ESR1 snapshot-and-hash benchmark runner completed one [local ChEMBL 37 run](benchmark-chembl37.md). Independent-origin results, repeated-seed evidence and publication remain outstanding. |

### Scientific issues to address before strong predictive claims

Current aggregation pools retained measurements into a single molecular
potency, although measurement identity is retained separately and assay subsets
can be selected. Converting Ki, Kd, IC50, and EC50 to a common negative-log
molar scale does not make their biological meanings identical. Pooling remains
an explicit analysis choice.

The time split now partitions retained measurements before aggregation and has
a regression test for a compound measured on both sides of the cutoff. It uses
*document year*, not actual compound disclosure date, and remains retrospective.
Compare descriptor, fingerprint, nearest-neighbour and simple baseline models
on identical frozen splits before using it as prospective evidence.

ChEMBL already integrates BindingDB (source 37) and a subset of PubChem
BioAssay (source 7). SARscope does not directly merge a second copy of those
measurements. The source-origin holdout partitions measurements before
aggregation and excludes compounds also observed in training origins. It is a
test of transfer to origin-unique compounds, not a guarantee of independent
biology: assays, chemistry series, and publications may still be related.
Direct PubChem qualitative screening now has a separate AID-level binary
classification workflow, with an offline source snapshot and scaffold-held-out
ROC/PR AUC. It does not coerce inactive calls to pIC50. The concise-response
import is limited to assays below 10,000 rows; large-screen export import and
assay-bias checks remain to do.

The existing target comparison is based on aggregated endpoint values, so its
ratios need assay-context qualification. An unmeasured off-target is unknown,
not inactive. A high potency difference across unrelated assays is not by itself
validated selectivity.

Applicability-domain flags are useful but are not prediction intervals. The
current 90% empirical residual band is estimated from training-side CV errors;
it has no distribution-free coverage guarantee. Report empirical coverage under
scaffold and time shifts; do not assume nominal coverage survives distribution
shift.

### Streamlit implementation constraints

Keep discovery, data preparation, chemistry, ML, and explanation as explicit
steps. Cache reusable data with bounded size and expiry, invalidate downstream
results when their inputs change, and show run settings alongside each result.
Page large tables and cap pair enumeration and explanation samples. Keep
larger analyses in the CLI; add persistent job infrastructure only when measured
usage shows it is necessary. No GPU is needed for these priorities.

## Success criteria

- A new user can find and confirm the intended target without knowing its ID.
- Every ranked series or proposed transformation can be traced to its underlying measurements.
- Missing data, contradictory measurements, weak coverage, and unsupported predictions are visible.
- A later-data test and an independent target family assess generalisation.
- A computational or medicinal chemist can export a justified shortlist and reproduce it.

## Primary references

- [ChEMBL search and data services](https://chembl.gitbook.io/chembl-interface-documentation/web-services/chembl-data-web-services)
- [ChEMBL assay confidence and pChEMBL definitions](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/chembl-data-questions)
- [ChEMBL source IDs and BindingDB/PubChem integration](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/document-and-data-source-questions)
- [PubChem BioAssay PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
- [ChEMBL assay context and comparability](https://chembl.gitbook.io/chembl-data-deposition-guide/file-structure/field-names-and-data-types-minimal-data-submission/assay.tsv)
- [Open Targets direct and indirect disease associations](https://platform-docs.opentargets.org/associations)
- [Open Targets target prioritisation](https://platform-docs.opentargets.org/web-interface/target-prioritisation)
- [Open Targets GraphQL API](https://platform-docs.opentargets.org/data-access/graphql-api)

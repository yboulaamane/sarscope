"""RunResults -> a folder a person can open, email, or attach as supplementary data.

    <out>/report.html          self-contained: figures inlined as base64 PNG
    <out>/provenance.json      versions, ChEMBL release, every parameter
    <out>/tables/*.csv         curation_log, curated_dataset, rejected,
                               table2_descriptor_profile, table3_pca_loadings,
                               table4_scaffold_diversity, scaffold_enrichment,
                               cliffs_<fp>, identical_pairs_<fp>,
                               cliff_generators_<fp>, consensus_cliffs,
                               table6_models
    <out>/figures/*.png        fig4_descriptors, fig5_pca, fig8_sas_<fp>,
                               fig9_generators_<fp>, fig10_domain

Table and figure names follow the reference paper's numbering so the two can
be read side by side. Figures use matplotlib's Agg backend (no display needed,
works in CI and on Colab).

The report opens with the curation log and the list of skipped steps, before
any result: a reader should know what the numbers are numbers *of* first.
"""

from __future__ import annotations

from pathlib import Path

from sarscope.pipeline import RunResults


def write_report(results: RunResults, out_dir: Path) -> Path:
    """Write everything above into ``out_dir`` (created if needed); return report.html.

    Raises FileExistsError for a non-empty directory that lacks a
    provenance.json, so a typo in --out cannot scatter files over an unrelated
    folder; a previous report's folder is overwritten. provenance.json is
    ``results.provenance``, or ``provenance.collect(results.params,
    {"kind": "unknown"})`` when that is empty (``analyse`` called directly).
    """
    raise NotImplementedError

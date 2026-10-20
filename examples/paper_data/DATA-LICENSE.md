# Data attribution and separate licences

The repository's MIT licence covers SARscope code, including the retrieval
script. It does **not** replace upstream dataset licences.

## ChEMBL assay cohorts and 2D benchmark

Source: EMBL-EBI ChEMBL, release **ChEMBL 37**, under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
This applies to the raw ChEMBL snapshots and attributed chemical-data derivatives
in `sar/` and `qsar2d/`.

The assay cohorts originate from Abdel-Maksoud et al. (2021),
[10.1021/acs.jmedchem.1c00230](https://doi.org/10.1021/acs.jmedchem.1c00230);
Zhang et al. (2000), [10.1021/jm990562x](https://doi.org/10.1021/jm990562x);
and Malamas et al. (2004), [10.1021/jm049719y](https://doi.org/10.1021/jm049719y).
The latter cohort is the **ER-alpha counter-assay**, not ER-beta selectivity.
Assay/document identifiers and original measurements remain in the snapshots.

Changes: assay filtering, structure standardization, median aggregation of
−log10(molar) IC50, scaffold/core selection and computed pair/statistical tables.
The full-cohort 2D CSVs are later reconstructions, with their verification scope
and export environments recorded in `qsar2d/export-provenance.json`. Original
saved predictions and frozen raw records are unchanged. No endorsement is implied.

## Original aligned 3D inputs

Thrombin/AChE: Sutherland et al. (2004),
[10.1021/jm0497141.s002](https://doi.org/10.1021/jm0497141.s002), publisher
metadata labelled [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
Thrombin was originally attributed to Böhm, Stürzebecher and Klebe (1999),
[10.1021/jm981062r](https://doi.org/10.1021/jm981062r).

Steroid/CBG: Haga et al. (2025),
[10.3390/ph18030440](https://doi.org/10.3390/ph18030440). The paper is CC BY 4.0;
the pinned source repository carries CC BY-SA 4.0 notices. Consult the original
notices rather than assuming all parts have the code licence.

Raw primary/third-party SD files are **retrieved**, not bundled in this folder,
consistent with the manuscript's existing redistribution policy. Public URLs,
file hashes, original activities and partitions are provided. Retrieval changes
no coordinates or activity values and downloads no external algorithm source.
The 3D membership tables are attributed data derivatives, not new measurements.

The existing CCR5 tutorial retains its own
[source and licence notices](../../src/sarscope/data/field_ccr5_example/README.md).

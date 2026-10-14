# Data attribution and licensing

ChEMBL source: EMBL-EBI, release **ChEMBL 37**. Data are provided under the
[Creative Commons Attribution-ShareAlike 3.0 Unported license](https://creativecommons.org/licenses/by-sa/3.0/),
as described in [ChEMBL's licensing documentation](https://chembl.gitbook.io/chembl-interface-documentation/frequently-asked-questions/general-questions).

The assay snapshots in `cases/`, associated numerical/chemical outputs in
`results/`, `benchmark-audit/` and `mmpdb-comparison/`, and any archived full
benchmark data are redistributed with the same CC BY-SA 3.0 license and
source attribution. These data are **not covered by SARscope's MIT code license**.

Changes made to source data: selected complete assay cohorts; recorded exclusions;
structure standardization and canonical tautomerization; concentration conversion
to -log10(molar); structure-level median aggregation; scaffold decomposition;
calculated position pairs, transformations and statistics. Source record IDs,
assay/document IDs, original records and snapshot hashes are retained.

Original data publications:

1. Abdel-Maksoud MS et al. (2021). Discovery of New Imidazo[2,1-b]thiazole
   Derivatives as Potent Pan-RAF Inhibitors with Promising In Vitro and In Vivo
   Anti-melanoma Activity. [doi:10.1021/acs.jmedchem.1c00230](https://doi.org/10.1021/acs.jmedchem.1c00230).
   ChEMBL document CHEMBL5046217; assay CHEMBL5046398; target CHEMBL5145.
2. Zhang X et al. (2000). trans-1-[(2-Phenylcyclopropyl)methyl]-4-arylpiperazines:
   mixed dopamine D(2)/D(4) receptor antagonists as potential antipsychotic agents.
   [doi:10.1021/jm990562x](https://doi.org/10.1021/jm990562x).
   Document CHEMBL1133592; assay CHEMBL671074; target CHEMBL217.
3. Malamas MS et al. (2004). Design and synthesis of aryl diphenolic azoles as
   potent and selective estrogen receptor-beta ligands.
   [doi:10.1021/jm049719y](https://doi.org/10.1021/jm049719y).
   Document CHEMBL1138773; **ER-alpha** assay CHEMBL829540; target CHEMBL206.

Publication metadata were checked against the ChEMBL document API. Case selection
does not imply the original authors endorse SARscope, nor that their published
SAR narratives have been independently recovered. No publisher's article text,
tables as typeset, or original graphics are included.

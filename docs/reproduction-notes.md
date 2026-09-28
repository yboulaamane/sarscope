# Reproduction notes

What was checked against Aouidate, *J. Mol. Liq.* 401 (2024) 124705, and how.
Every number here was measured, and the method is given so anyone can re-run it.
Data: ChEMBL 37 (release 2026-05-01), fetched September 2026.

## Target

The paper cites "ChEMBL version 31, target ID: 5651". **CHEMBL5651 is STK35**
(serine/threonine-protein kinase 35), with 3 IC50 records. BRAF is
**CHEMBL5145**, with 11,017 IC50 records in ChEMBL 37. The paper's counts only
make sense for BRAF, so 5651 is taken to be a typo.

```bash
sarscope target 5651
sarscope target CHEMBL5145
```

## Curation

The paper reports 5,432 activity entries curated to 3,952 molecules, with the
median taken over repeats. It does not state the relation, assay-type, variant,
duplicate or format filters. Measured on BRAF IC50 records from documents
published up to 2022 (10,741 records, 6,648 molecules), one filter at a time:

| Filter alone                | Records | Molecules |
|-----------------------------|--------:|----------:|
| none                        | 10,741  | 6,648     |
| relation "="                | 8,860   | 5,350     |
| pChEMBL value present       | 8,834   | 5,325     |
| single-protein format       | 7,234   | 4,800     |
| wild-type only (no V600E)   | 6,357   | 3,962     |
| not a potential duplicate   | 8,420   | 6,648     |

No combination of these filters reproduces both 5,432 records and 3,952
molecules. The document-year cut-off only roughly approximates the ChEMBL 31
contents, so exact agreement was not expected. The spread is the finding: the
unstated choices change the dataset by a factor of about two.

## Class counts and the enrichment factor

Section 2.1 gives 2,361 Group 1 and 1,591 Group 2 molecules. The class counts
in the same section (1,218 potent, 1,080 active, 948 intermediate, 706
inactive) sum to 2,298 and 1,654. The maximum scaffold EF reported (1.719,
for all-Group-1 scaffolds) equals N / A = 3952 / 2298 = 1.7198. So the class
counts are the ones actually used, and the EF definition is confirmed.

## Validation order

Section 2.5.3 oversamples, and section 3.4 then splits "the balanced BRAF
inhibitors dataset" 80:20. Duplicating rows before splitting places copies of
one molecule on both sides.

Prototype, 320 molecules with random labels (4 imbalanced classes, 200/60/40/20)
and 64 random bits, five seeds, the paper's hyperparameters:

| Model              | Leak-free test MCC | Oversample-then-split test MCC |
|--------------------|-------------------:|-------------------------------:|
| Extra Trees        | 0.00 on all seeds  | 0.98–0.99                      |
| Nearest neighbours | −0.08 to 0.07      | 0.52–0.65                      |

On noise, the correct answer is about zero. This does not mean the paper's
reported 0.733 is wrong, because real BRAF data carries real signal. It means
that figure contains an unknown amount of inflation, which `leakage_audit`
measures on the real data.

## Smaller discrepancies

- Table 4, Potent row: Nss = 445. The text gives 455 and 71.3%, and 455 / 638
  = 0.713.
- Section 2.5.2: "of the 220 features, there are 223 left". Probably 881
  PubChem bits → 223 → 104.
- Section 2.5.3 says the minority classes were oversampled up to the inactive
  count (706). The Fig. 3 caption says up to the potent count, giving 4,872 =
  4 × 1,218, which matches the reported total.
- Section 3.3's description of the SAS map swaps the two left-hand quadrants
  relative to its own axes. SARscope defines the regions by meaning (see
  `analysis/landscape.py`).

## Not reproducible as published

- **Cyclic skeletons.** Table 4's Ncsk (47) comes from DataWarrior, whose
  skeleton definition differs from RDKit's generic Murcko scaffold.
- **PubChem fingerprints.** The paper used PaDEL (Java). SARscope's optional
  PubChem fingerprint comes from scikit-fingerprints, and its agreement with
  PaDEL has not been verified.
- **The exact dataset.** The data availability statement reads "Data will be made
  available on request."

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

Measured on the real BRAF dataset (CHEMBL5145, ChEMBL 37, 2,917 curated
molecules, ECFP4, scaffold split, 5-fold CV), running both orders on identical
data:

| Model             | Leak-free test accuracy | Paper's order | Inflation |
|-------------------|------------------------:|--------------:|----------:|
| Random forest     | 0.723 | 0.877 | +0.154 |
| Extra trees       | 0.716 | 0.870 | +0.154 |
| Nearest neighbours| 0.668 | 0.766 | +0.099 |
| Gradient boosting | 0.697 | 0.778 | +0.081 |

The paper reports Extra Trees at train 0.920, CV 0.699, test 0.733. Our
leak-free Extra Trees run gives train 0.949, CV 0.683, test 0.716 - close to
the published figures, which is the useful result: the honest numbers for this
target sit near 0.72, and the published 0.733 is consistent with them. The
leaky order would have produced 0.870.

Note the ordering that gives the leak away. Under the leak-free protocol,
test accuracy (0.716) is just above CV (0.683), as expected. The paper reports
test 0.733 above CV 0.699 by a similar margin, so its reported numbers look
more like the honest protocol than the leaky one, whatever section 3.4 says.

A synthetic check isolates the mechanism: on 320 molecules with random labels
and random bits, the leaky order scores Extra Trees at MCC 0.98-0.99 across
five seeds while the leak-free order gives 0.00. That is a regression test in
`tests/test_science_model.py`.

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

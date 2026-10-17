# Real measured 3D-QSAR workflow example

This optional example belongs to **3D QSAR · own compounds**, not to the 2D
analogue-design workflow. It contains 50 real, literature-reported EGFR IC50
activities on one 4-anilinoquinazoline Murcko scaffold from ChEMBL assay
**CHEMBL680017**, target **CHEMBL203**. Potency values are not synthetic.

- `egfr_3d.csv`: `molecule_id,smiles,pactivity` for your own alignment.
- `egfr_3d_aligned.sdf`: the same measured molecules and activities, with
  generated, rigid-MCS-aligned 3D coordinates for a repeatable, faster start.
- `measurements.csv`: retained ChEMBL activity IDs, values, units and assay context.
- `source.json`: source hashes, filtering, structural selection and geometry assumptions.
- `alignment.csv`, `verification.json`: alignment QC and actual 3D results/checks.

## Walkthrough

Choose **3D QSAR · own compounds**, expand **Try a measured 3D-QSAR example**,
and check **Use measured 3D-QSAR example**. All 50 members of the one scaffold
are selected. Keep uploaded alignment and the suggested reference CHEMBL29197.
Prepare/review the overlay, check the review declaration, then explicitly
run 3D QSAR. No modelling runs merely by selecting the example.

After the fitted model's results, open **4 · Design analogues and predict with
this 3D model**. Prepare R-positions, choose a parent and editable sites,
generate a small batch, then align/predict. These predictions use the displayed
PLS estimator, its original training-only field mask/scales, charge method,
fixed grid and aligned reference. The 2D RF baseline is not used to score
analogues. No estimator is refitted on the test compounds.

With a pre-aligned model you must explicitly choose a candidate alignment
route. New conformers are generated hypotheses, not validated bound poses.
Per-field contributions explain the model's predicted delta relative to its
own parent prediction; they are not physical interaction energies.

## Honest limitations

This is a **real-data workflow demonstration, not a successful validation
benchmark**. The first fixed-default 3D run was weak (negative held-out R² and
training-CV q²); inspect the exact serialized-SDF verification record. The
previous 106-compound demo's high **2D** RF scores do not apply here. No seed,
field selection or test-dependent row selection was used to manufacture good
3D scores. SARscope's native end-to-end 3D validation remains incomplete.

With unmet checks, scoring is withheld by default. Explicit exploratory
consent can expose model predictions, but does not pass or change any checks.
Geometry, stereochemistry, fixed-grid bounds, missing-field reliability and
training-field PCA failures still withhold the affected potency. No proposed
analogue has been synthesised/tested by SARscope, and no gain is confirmed.

Source conversion: pIC50 = `9 - log10(IC50 in nM)`. Exact, unflagged records
surviving documented curation are retained; one alternative Murcko scaffold
is excluded by the structural criterion, not by potency. Missing mutation
annotation is unreported, not proof of wild type. ChEMBL assay annotation does
not independently establish all experimental conditions.

Coordinates were generated using SARscope's stereo-checked seed-42 conformer
workflow and rigid MCS alignment to CHEMBL29197; they are not crystal poses.
The SDF preserves that generated frame, with standard SDF coordinate precision.

## Attribution and separate data licence

Data from **ChEMBL, EMBL-EBI**, redistributed/adapted under
[CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/), separately
from SARscope's MIT-licensed code. This applies to the CSV/SDF, retained
measurements and accompanying data metadata in this directory. Changes:
quality filtering, one-scaffold selection, structure standardisation without
canonical tautomerisation, pIC50 conversion/aggregation, coordinate generation,
alignment and addition of a software walkthrough/evaluation record.
Retain attribution and the data licence. No endorsement is implied.

Original source: Bridges AJ et al. *Tyrosine kinase inhibitors. 8. An unusually
steep structure-activity relationship for analogues of
4-(3-bromoanilino)-6,7-dimethoxyquinazoline (PD 153035), a potent inhibitor of
the epidermal growth factor receptor.* J Med Chem. 1996.
[doi:10.1021/jm9503613](https://doi.org/10.1021/jm9503613).

Rebuilder: `examples/analogue_design/build_3d_example.py` in the repository.

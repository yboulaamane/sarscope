# Cliff-pair docking: Vina + ProLIF

This optional Streamlit step compares two members of a selected activity-cliff
pair against the same user-supplied receptor. AutoDock Vina and ProLIF are free,
open-source tools. No PoseView service, API key or commercial license is required.
There is no hardcoded receptor, BRAF variant or binding pocket.

## Before running

1. Curate activity data, run the activity-cliff analysis, and select a pair.
2. Open **Dock this cliff pair · optional Vina + ProLIF** and enable its controls.
3. Upload a prepared **rigid, protein-only receptor PDBQT** and the matching
   **prepared protein PDB**. Heavy-atom identifiers and coordinates must agree
   within 0.05 Å. Give the receptor a provenance label or PDB ID; this label does
   not fetch a structure or verify its target/variant.
4. Define the box center and dimensions explicitly. Alternatively upload exactly
   one bound 3D reference ligand SDF, in the receptor's coordinate frame, and
   click **Set box from bound reference** (8 Å total padding). A free conformer
   or a 2D depiction is not a bound reference.
5. Review the target/variant, input chemistry and pocket; confirm the checkbox
   and click **Dock this pair**. Changing settings alone never starts a job.

For example, after reviewing and repairing a protein structure outside the app,
Meeko can write both required files from the same preparation:

```bash
mk_prepare_receptor.py --read_pdb reviewed_protein.pdb \
  -o receptor -p --write_pdb receptor_prepared.pdb
```

Upload `receptor.pdbqt` and `receptor_prepared.pdb`, not the original PDB: preparation
can change coordinates. Resolve missing atoms, alternate locations, protonation,
termini, disulfides and residue templates deliberately. Do not automatically delete
unmatched residues just to force a preparation to succeed. See the
[Meeko receptor tutorial](https://meeko.readthedocs.io/en/develop/tutorial4b.html).

Protein-only is a scientific scope restriction, not advice to remove essential
cofactors. If metal ions, cofactors, waters or covalent chemistry are essential to
the binding mode, this version is unsuitable; use a validated external workflow.

## What happens

- Ligands retain the supplied curated SMILES state, formal charges and specified
  stereochemistry. RDKit ETKDGv3 generates a conformer and UFF minimises it;
  Meeko prepares the ligand PDBQT. There is no pH-dependent protomer/tautomer
  enumeration. Undefined stereochemistry, including E/Z, is rejected.
- Vina docks both ligands sequentially against the same receptor, box, seed and
  settings. Up to three raw poses per molecule are retained. Meeko reconstructs
  the top-ranked pose with its original bond orders; chemistry is checked again.
- ProLIF analyses only that top-ranked pose using its default geometric
  interaction definitions and template-based implicit hydrogen treatment.
  Protein templates are not a pKa prediction or H-bond-network optimisation.
- Each pose has a 2D interaction network and contact table. The comparison is
  by **protein residue + interaction type**, not ligand atom index: “retained”
  does not mean identical atoms or geometry. Reported distances are minimum
  detected distances for that residue/type; undefined distances remain blank.
  Exported atom indices are zero-based in ProLIF's standardized heavy-atom
  molecules, not original PDB atom serials or SDF labels.
- The 3D viewer overlays A (cyan carbons) and B (magenta carbons). Native network
  HTML is generated on the host, without an external interaction-analysis service.
  Browser viewers may fetch JavaScript libraries from public CDNs.

## Interpreting results

The panel shows each Vina score, ΔVina score = B − A (kcal/mol), and experimental
ΔpActivity = B − A. Lower docking scores are favourable; higher pActivity is
favourable. Their units and meanings differ: **do not convert docking scores to
pIC50 or treat the difference as a predicted experimental potency difference**.
The <0.05 kcal/mol near-tie label is only a display heuristic, not an uncertainty
estimate; even larger score differences may not be meaningful.

Score-rank agreement for one pair is not general validation. Disagreement is
reported explicitly. Gained/lost contacts are pose-dependent hypotheses, not
proof of the mechanism of an activity cliff. Before making a scientific claim,
validate the docking protocol (including redocking/pose recovery), inspect poses,
assay comparability, chemical states and relevant protein conformations.

## Bounds and reproducibility

Jobs use exactly two molecules, one CPU, and one active job per host. Limits:
3 MiB per receptor file, 15,000 receptor atoms, 6–25 Å per box dimension,
exhaustiveness 1–16, 1–3 poses, and a total timeout up to 300 seconds including
imports/preparation/interactions. Ligands must be single-component organic
C/N/O/F/P/S/Cl/Br/I molecules with 2–80 heavy atoms and at most 15 rotatable bonds;
macrocycles with rings of nine or more atoms are rejected. These are browser
resource/scope limits, not a comprehensive chemical-domain validation.

Timeouts or interrupted reruns kill the worker process group, including Vina.
Worker files are temporary. Results are session-local; changing the pair, receptor
or settings hides stale results. Resetting the workflow removes stored results.
No docking is silently repeated on unrelated widget changes.

**Download pair docking results** provides a separate ZIP with the receptor files,
prepared ligand PDBQTs, raw output PDBQTs, top-ranked SDFs, 2D HTML diagrams,
contacts, A→B contact changes, result JSON and protocol manifest. The manifest
records receptor input SHA-256 hashes, ligand SMILES and experimental pActivity,
seed, settings, engine/library versions, preparation method and warnings. Keep
this archive with the main SAR report; it is not automatically merged into it.

## Local setup and verification

```bash
pip install -e '.[app,docking]'
sudo apt-get install autodock-vina
streamlit run streamlit_app.py
```

For an executable elsewhere on Linux/macOS, set `SARSCOPE_VINA_BINARY` to its
path. Streamlit Cloud's `requirements.txt` and `packages.txt` include this step's
dependencies. Missing optional dependencies disable the panel, not the SAR app.

```bash
pytest -q tests/test_docking.py tests/test_app_docking.py
```

The real-engine smoke test uses a generated AA dipeptide and two small ligands:
it checks execution, pose restoration, interactions and the ZIP, **not predictive
performance**. It skips when docking dependencies/Vina are unavailable. No
independent target benchmark or scientific docking validation is run automatically.

References: [Vina preparation/scoring caveats](https://autodock-vina.readthedocs.io/en/latest/faq.html),
[ProLIF docking conversion](https://prolif.readthedocs.io/en/latest/notebooks/docking.html),
[ProLIF PDB and implicit-hydrogen workflow](https://prolif.readthedocs.io/en/latest/notebooks/pdb.html).

# Cliff-pair docking: Vina + ProLIF

This optional Streamlit step compares two members of a selected activity-cliff
pair against the same selected receptor. AutoDock Vina and ProLIF are free,
open-source tools. No PoseView service, API key or commercial license is required.
There is no hardcoded receptor, BRAF variant or binding pocket.

## Before running

1. Curate activity data, run the activity-cliff analysis, and select a pair.
2. Open **Dock this cliff pair · optional Vina + ProLIF** and enable its controls.
3. Keep **Fetch & prepare PDB** selected. Click **Find PDB structures** to find
   up to 20 experimental entries mapped to the selected target UniProt accession,
   sorted by resolution. Select one, or enter a known four-character PDB ID;
   click **Fetch selected structure**. No structure or pocket is hardcoded.
4. Review the title and chain mutation annotations, then select protein chains.
   A matching UniProt chain is required when target mapping is available, but
   UniProt identity does **not** confirm the assayed mutation, construct or state.
   Missing mutation annotations do not mean confirmed wild-type.
5. Choose a bound-reference ligand, or click **Predict candidate pockets · fpocket**
   when none is suitable. One qualifying reference is selected automatically;
   multiple references and predicted pockets require your selection. Preview the
   site and orange grid box; a manual box is also available.
6. Confirm that you reviewed the construct, chains and site, then click
   **Prepare receptor · Meeko**. Ligands are prepared from the cliff pair's SMILES:
   neither receptor nor ligand uploads are needed in this mode.
7. Review the editable grid and docking settings, confirm the docking checkbox,
   then click **Dock this pair**. Changing settings alone never starts a job.

If no suitable experimental structure exists, use a known PDB entry or externally
prepared receptor. This workflow does not generate a predicted protein structure.

## Automatic preparation and site selection

RCSB coordinates are fetched as mmCIF, with a 20 MiB download cap. Only coordinate
model 1 and your selected asymmetric-unit chains are retained; the app does not
infer the biological assembly. Chain IDs are remapped to single-character PDB
IDs and recorded. Alternate locations use the highest summed heavy-atom occupancy
per residue (alphabetical tie-break); waters and source hydrogens are removed.
Meeko's residue templates assign hydrogen/charge states, **not pH-dependent pKa
optimisation**. Unsupported protein residues and missing heavy atoms stop the
preparation; they are not silently deleted or rebuilt. The prepared protein PDB
and receptor PDBQT must retain identical heavy-atom sets and matching coordinates.

Bound-reference candidates are organic nonprotein components with at least six
heavy atoms and carbon, within 6 Å of the retained protein. These proximity rules
can include additives/cofactors; they do not establish the biologically relevant
site. For a selected reference, the box uses its heavy-atom coordinate bounds
plus **8 Å total padding** (4 Å on each side). Each dimension is at least 6 Å.
Boxes exceeding 25 Å are rejected, **never silently clipped**; choose another
site or a scientifically justified manual box.

The explicit fpocket step detects candidate cavities on the selected protein,
using its default settings. Up to 20 ranked pockets show their score, druggability
score, volume and box. Boxes enclose alpha-sphere centres plus the same padding;
the preview shows a subset of these points. Pocket rank and druggability are
**not validated binding-site assignments**. You choose the candidate; docking
does not start automatically. If no pocket is found, choose a manual box or
another structure. See [fpocket's guide](https://github.com/Discngine/fpocket/blob/master/doc/GETTINGSTARTED.md).

Known metal/cofactor components excluded from the fetched protein are recorded.
If one lies within the actual docking box plus 3 Å, protein-only docking is
blocked, including after manual grid edits. Covalent protein–nonprotein links
also stop automatic preparation. This guard is not exhaustive: essential waters,
unrecognised cofactors and special chemistry still require expert review.

## Advanced prepared uploads

Choose **Upload prepared files** for an externally reviewed **rigid, protein-only
receptor PDBQT** and its matching **prepared protein PDB**. Heavy-atom identifiers
and coordinates must agree within 0.05 Å. A receptor label/PDB ID in this mode is
provenance only: it does not fetch or verify a structure. Define the grid manually,
or upload exactly one bound 3D reference ligand SDF in the receptor's coordinate
frame and click **Set box from bound reference**. A free conformer or 2D depiction
is not a bound reference. Uploaded receptors cannot be checked against components
removed before upload; you are responsible for receptor/site suitability.

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

Docking jobs use exactly two molecules and one CPU. Preparation, pocket prediction
and docking share one active-job lock per host. Preparation and pocket prediction
each have a 120-second total deadline; docking has a user-selected deadline up to
300 seconds including ligand preparation and interactions. Limits:
3 MiB per receptor file, 15,000 receptor atoms, 6–25 Å per box dimension,
exhaustiveness 1–16 and 1–3 poses. Ligands must be single-component organic
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
seed, settings, engine/library versions, preparation method and warnings. For
fetched receptors it also records the PDB source/hash, chain selection/remapping,
alternate locations, excluded components, mutation annotations, automatic box
method and pocket-engine identity. A selected bound reference PDB is included.
Keep this archive with the main SAR report; it is not automatically merged into it.

## Local setup and verification

```bash
pip install -e '.[app,docking]'
sudo apt-get install autodock-vina
streamlit run streamlit_app.py
```

For an executable elsewhere on Linux/macOS, set `SARSCOPE_VINA_BINARY` to its
path. Streamlit Cloud's `requirements.txt` and `packages.txt` include this step's
dependencies. Missing optional dependencies disable the panel, not the SAR app.

On Linux x86_64, the **first explicit pocket-prediction request** downloads and
caches the approximately 1.8 MB conda-forge fpocket 4.2.2 `h7b35b64_0` package.
Both package and executable SHA-256 hashes are verified before execution; license
files are retained. `zstandard` is included in the docking extra to extract it.
No Java, compiler or landing-page download is needed. On another supported POSIX
host, install fpocket yourself and set `SARSCOPE_FPOCKET_BINARY`, or put it on
`PATH`; the executable hash is recorded. If download/setup fails, bound-reference
and manual boxes remain available. The cache can disappear when a cloud host
restarts. This is local geometric pocket detection, not a remote prediction service.

```bash
pytest -q tests/test_docking.py tests/test_receptor.py tests/test_app_docking.py tests/test_app_receptor.py
```

The real-engine smoke test uses a generated AA dipeptide and two small ligands:
it checks execution, pose restoration, interactions and the ZIP, **not predictive
performance**. Receptor tests use synthetic coordinates and mocked RCSB requests;
native fpocket tests use a tiny peptide and a synthetic cavity, not a binding-site
benchmark. Native tests skip when their engines/dependencies are unavailable. No
independent target benchmark or scientific docking validation is run automatically.

References: [Vina preparation/scoring caveats](https://autodock-vina.readthedocs.io/en/latest/faq.html),
[ProLIF docking conversion](https://prolif.readthedocs.io/en/latest/notebooks/docking.html),
[ProLIF PDB and implicit-hydrogen workflow](https://prolif.readthedocs.io/en/latest/notebooks/pdb.html).

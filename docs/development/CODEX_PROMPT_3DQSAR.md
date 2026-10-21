# Task: add field-based 3D-QSAR (CoMFA / CoMSIA) to SARscope

Historical development prompt, retained for transparency. Its descriptions of
the then-current code and proposed methods are not current user documentation,
the final scientific protocol, or a claim that every proposed item was implemented.

You are adding a new analysis capability to **SARscope**
(`github.com/yboulaamane/sarscope`, MIT). Work inside the existing repository
and follow its conventions. Do not create a separate application.

SARscope today does 2D QSAR: curation, descriptors, fingerprints, leak-free
regression, applicability domain, activity cliffs, SAR stress tests, and
opt-in pair docking. It has **no** molecular interaction field (MIF) capability
at all — zero mentions of CoMFA, CoMSIA, MIF or 3D-QSAR in the codebase. That
is the gap you are filling.

---

## 1. Hard constraints — read before writing code

### 1.1 Reuse, do not reimplement

The following already exist and are better than anything you would write fresh.
**Use them. Do not duplicate them.** Duplicating any of these is a failed task.

| Need | Use this | Why |
|---|---|---|
| Leak-free cross-validation, model selection, test-set discipline | `sarscope.analysis.regression` (`evaluate_regression`, `fit_deployment_model`) | Already does fold-local feature filtering with the test set touched once. This is SARscope's headline feature. |
| Run provenance: versions, SHA-256, parameters, timestamp | `sarscope.provenance` (`collect`, `file_sha256`, `package_versions`) | Add any new tracked package to `TRACKED_PACKAGES`. |
| Dataset curation, record accounting | `sarscope.curate` | Counts every dropped record under the step that removed it. |
| Physicochemical descriptors | `sarscope.analysis.descriptors` | Definitions chosen against measured references. |
| Fingerprints, in-CV feature filter | `sarscope.analysis.features` (`VarianceCorrelationFilter`) | |
| Applicability domain | `sarscope.analysis.domain` (`pca_bounding_box`) | |
| Conformer generation | `sarscope.conformers` (`embed_ligand_conformer`) | Validates stereochemistry from the actual 3D coordinates, not just the SMILES tags. |
| Receptor inspection, altloc choice, Meeko prep | `sarscope.receptor` (`inspect_structure`, `prepare_receptor`, `box_from_coordinates`) | Uses gemmi; selects highest summed-occupancy altloc. |
| Docking execution | `sarscope.docking` (`run_structure_job`) | Vina as a subprocess with process groups, timeouts, file locks and bounded inputs. **Do not add a second Vina pathway** — in particular do not introduce the `vina` Python bindings. |
| Standardisation, scaffolds | `sorbent` via `sarscope.chem` | |
| Report sections, tables, figures | `sarscope.report` | |
| Settings | frozen dataclasses in `sarscope.params` | |

### 1.2 Structure parsing: use gemmi, never hand-parse PDB text

SARscope already depends on gemmi. Use it for everything structural:

- `structure.assemblies` and `structure.transform_to_assembly(...)` for
  biological assemblies. **Do not write a REMARK 350 / BIOMT parser.**
- `structure.connections` for LINK records (covalent connectivity).
  **Do not infer covalent bonds from interatomic distance alone** — see
  gotcha 14.
- gemmi for altlocs, insertion codes, mmCIF, chains.

Column-slicing PDB text is forbidden in new code.

### 1.3 Licence hygiene — important

**Open3DQSAR and Open3DALIGN are GPLv3. SARscope is MIT.**

Implement every method from the **primary literature only** (citations in
§6). Do **not** read, port, translate or consult the source code of
Open3DQSAR, Open3DALIGN, or any GPL 3D-QSAR implementation. Doing so would
make SARscope's MIT licence untenable.

Open3DQSAR's *benchmark datasets* may be used as data (check their data
licence), but not its code.

---

## 2. What to build

### 2.1 `src/sarscope/analysis/fields.py` — MIF calculation

Pure NumPy + RDKit. No new dependencies. This is the core of the task.

**Grid.** A `GridSpec` dataclass: rectangular lattice spanning every aligned
conformer, padded by a margin. Defaults: **2.0 Å spacing, 4.0 Å margin** (the
CoMFA defaults; a finer grid multiplies columns without improving predictivity
and shifts contours toward noise). Expose the lattice as an `(n_points, 3)`
array plus shape and origin.

**Charge models.** A registry, because the charge model materially changes the
result (gotcha 9). Built in: `gasteiger` (default — what Cramer used in 1988,
so it is the right choice for comparison with published work) and `mmff94` via
`AllChem.MMFFGetMoleculeProperties(...).GetMMFFPartialCharge(i)`. Provide a
`register_charge_model(name, fn)` hook so AM1-BCC-quality models
(`openff-nagl`, `espaloma`) can be added locally without those becoming
dependencies. Store charges under one property name so the field code reads a
single source regardless of model. MMFF94 covers fewer atom types than
Gasteiger: fall back per molecule, **count the fallbacks and surface them** —
a silent mix of charge models makes the electrostatic field inconsistent
across the series.

**CoMFA fields** (Cramer, Patterson & Bunce, *JACS* 1988, **110**, 5959).
Probe: sp3 carbon, radius 1.70 Å, ε 0.1070 kcal/mol, charge +1.0.

- Steric, Lennard-Jones 12-6:
  `E = Σ_i ε_ij · [(R_ij/r)^12 − 2·(R_ij/r)^6]`
  with `ε_ij = sqrt(ε_i · ε_probe)` and `R_ij = r_i + r_probe` (van der Waals
  radii sum).
- Electrostatic, Coulomb:
  `E = 332.0636 · q_probe · Σ_i q_i / (D · r)`, dielectric `D = 1`.
- Clip `r` to a minimum of 0.3 Å to avoid the singularity at nuclei.
- Truncate steric energy at **+30 kcal/mol**.
- Where the steric cutoff is hit, the electrostatic term is **undefined**.
  Set it to `NaN`, not to zero and not to the column mean. Those columns carry
  no information and must be handled explicitly downstream (gotcha 6).

**CoMSIA fields** (Klebe, Abraham & Mietzner, *J Med Chem* 1994, **37**, 4130).
Gaussian-damped similarity indices, attenuation **α = 0.3**, probe carrying a
unit value for every property:

`A_k(j) = Σ_i w_probe,k · w_ik · exp(−α · r_ij²)`

Five fields: `steric` (atomic volume, taken as r³), `electrostatic` (partial
charge), `hydrophobic` (Crippen atomic logP contributions via
`Crippen._GetAtomContribs`), `hbond_donor` and `hbond_acceptor` (indicator
values from SMARTS; exclude amide and sulfonamide nitrogen from acceptors).
The Gaussian has no singularity, so no cutoffs are needed and every grid point
stays defined — that is CoMSIA's main practical advantage over CoMFA.

**Chunking.** Compute distances in blocks over grid points. Computing them all
at once allocates `n_points × n_atoms × 3` floats, which exhausts memory on a
fine grid (gotcha 12). Provide a memory estimator so a caller can check before
starting. Chunking must be numerically exact — assert it in a test.

**Public entry point.** Something like
`compute_fields(mols, grid, method, charge_model, **kw)` returning the
descriptor matrix, column labels of the form `"<field>_<point index>"`, and a
report naming the charge model each molecule actually used.

### 2.2 `src/sarscope/analysis/align3d.py` — alignment for field calculation

Alignment determines a field-based model more than any other choice and is the
least reproducible step in most published work. Offer three routes and state
plainly what each one assumes:

1. **Pre-aligned input** — an SDF already in a common frame (docked poses,
   superposed co-crystal ligands). Preferred, because the alignment hypothesis
   is then experimental. Must be a first-class path, not an afterthought.
2. **Rigid MCS onto a reference** — for a tight congeneric series.
3. **Core-constrained embedding** — pin the MCS core to a reference ligand's
   coordinates and sample only the substituents. For receptor-based work.

Requirements:

- Enumerate **all symmetry-equivalent MCS mappings** and keep the best-fitting
  one (gotcha 1).
- Use `bondCompare=rdFMCS.BondCompare.CompareOrderExact` and reject mappings
  pairing atoms of differing aromaticity or hybridisation (gotcha 2).
- Return a per-molecule report — alignment RMSD and MCS size — so a bad
  alignment is visible rather than silent, with warnings when RMSD exceeds a
  threshold or the common core covers too little of the molecule.
- Report the common core across the whole series and its coverage as a
  fraction of mean heavy-atom count. **Strip hydrogens before computing the
  MCS** if comparing against heavy-atom counts (gotcha 11).
- For core-constrained embedding, put the coordinate map on the
  `EmbedParameters` object via `SetCoordMap` (gotcha 3), restrain the core
  during force-field minimisation, rank conformers by restrained energy, and
  enforce a **per-molecule time budget** so one geometrically impossible
  molecule cannot stall a run.

### 2.3 `src/sarscope/analysis/contours.py` — contribution maps

The quantity mapped is the **PLS coefficient scaled by the column's standard
deviation**, never the raw coefficient. A grid point the series barely explores
can carry a large coefficient while explaining nothing, so a raw-coefficient
map places contours where there is no information.

- Contour levels at **percentiles** of contribution magnitude, not fixed
  numeric levels, because the absolute scale differs between CoMFA energies
  and CoMSIA indices and between datasets. The 80/20 favoured/disfavoured
  convention is the default.
- **Per-field percentage contributions** — the "steric 62%, electrostatic 38%"
  numbers a paper reports — computed as each field's share of total absolute
  contribution.
- State explicitly, in the code and in the output, what a positive and a
  negative contour mean **for each field**. Inverting that reverses the
  chemical conclusion, which is the most consequential error a 3D-QSAR paper
  can make.
- Export **OpenDX** grids (readable by PyMOL, Chimera, VMD, py3Dmol) and a
  PyMOL script setting the levels, using the conventional colours: steric
  green/yellow, electrostatic blue/red, hydrophobic orange/white, donor
  cyan/purple, acceptor magenta/salmon.

### 2.4 Column filtering and modelling

**Do not write a new PLS/CV layer.** Use `sarscope.analysis.regression`.
You will need to extend it (or add a thin adapter) for two things:

- **PLS as an estimator** alongside the existing regressors, since field-based
  QSAR is conventionally PLS and comparability with the literature requires it.
- **Component cap at n/5.** Uncapped, PLS extracts components that raise r²
  while q² stalls — the overfitting signature in much published CoMFA
  (gotcha 7). Select the component count by cross-validated q² with a
  stopping rule: the first count after which q² stops improving by more than
  5% of its own value.

Column filtering, in two parts:

- **Undefined (NaN) columns** are dropped **once over the whole set, before
  splitting.** The mask depends only on molecular geometry, never on activity,
  so it carries no information about y and cannot inflate q². Deciding it per
  fold leaves NaNs in the held-out rows (gotcha 6).
- **Variance filtering is refitted inside every fold**, like SARscope's
  existing feature filter. Two modes:
  - `absolute` — CoMFA's classic σ ≥ 2.0 kcal/mol threshold.
  - `percentile` — keep the most variable fraction. **Required for CoMSIA**,
    whose similarity indices are unitless and roughly a thousand times smaller
    than CoMFA energies, so the kcal/mol threshold is meaningless for them
    (gotcha 8).

### 2.5 Receptor-side additions (optional extra, gated like docking)

SARscope's docking is explicitly scoped to "rigid protein-only receptor; no
cofactors, waters, covalent ligands or macrocycles". These additions complement
that scope rather than widening it:

- **Redocking validation gate.** Redock the co-crystal ligand into the prepared
  receptor and report the symmetry-corrected heavy-atom RMSD **without
  superposition** (poses live in the receptor frame; superposing first hides
  the displacement being measured). If the native ligand is not recovered to
  within 2 Å, the pipeline must **stop and say why** — box, protonation,
  retained components, missing side chains. Report the top-ranked pose's RMSD
  *and* the best pose anywhere in the output with its rank: a large gap means
  sampling succeeded while scoring mis-ranked, which is itself the argument
  against selecting poses by score. SARscope has no pose validation at all
  today; this fits its validation ethos directly.
- **Covalent ligand detection.** Using `gemmi` `connections`, detect that a
  reference ligand is covalently bound to a protein nucleophile (Cys SG, Ser
  OG, Thr OG1, Lys NZ, Tyr OH, His NE2/ND1). Vina models every ligand as
  non-covalent, so the gate must **refuse** rather than report a meaningless
  RMSD. This explains to the user *why* their complex is out of scope.
- **Biological assembly awareness.** Via `gemmi` `assemblies` /
  `transform_to_assembly`, warn when the deposited coordinates are a fraction
  of the functional molecule. Docking into whatever chains are in the file is
  a real error — binding sites completed by a partner subunit are only correct
  in the assembly.

### 2.6 Wiring

- **`params.py`**: a frozen `FieldQsarParams` dataclass — method
  (`comfa`/`comsia`), grid spacing, margin, charge model, CoMSIA α, filter
  mode and threshold/fraction, alignment route, component cap. Every setting a
  caller can change must live here so `provenance.collect` records it.
- **`pipeline.py`**: add the stage following the existing pattern, producing a
  result dataclass consumed by `report.py`.
- **`report.py`**: a section with the statistics table, per-field percentage
  contributions, predicted-vs-experimental and residual plots, the alignment
  report, and the contour grids as downloadable files.
- **`streamlit_app.py`** / a `*_ui.py` module: a 3D-QSAR option following the
  existing UI conventions (see `sar_ui.py`, `receptor_ui.py`,
  `docking_ui.py`). Field calculation needs no extra dependencies, so it can
  be always-on; **pose generation must be gated behind the docking extra** and
  degrade with a clear message when unavailable, as docking already does.
- **`pyproject.toml`**: no new required dependencies. Add any optional ones as
  an extra. Add new packages to `provenance.TRACKED_PACKAGES`.

---

## 3. Gotchas — each of these cost real debugging time

Every one was found by running the code, not by reading about it. Treat them as
requirements, not advice.

1. **MCS symmetry gives ring-flipped alignments.** RDKit's
   `GetSubstructMatch` returns one arbitrary match, so a symmetric scaffold
   (a para-substituted phenyl, say) can be superposed backwards. Enumerate
   with `GetSubstructMatches(..., uniquify=False)`, deduplicate by the geometry
   imposed, and keep the best-fitting mapping. *Measured: mean alignment RMSD
   1.47 Å → 0.63 Å on an 18-compound series.*

2. **Default MCS bond comparison maps aromatic rings onto aliphatic ones.** A
   phenyl analogue gets mapped onto a reference's cyclohexyl, so constrained
   embedding is asked to place a planar ring at puckered coordinates. No
   conformer can satisfy it, and the search grinds on for minutes before
   failing with an opaque message. Use `CompareOrderExact` plus an explicit
   aromaticity/hybridisation check on every mapped pair. *Measured: one ligand
   went from 606 s and an uninformative failure to 2.7 s with a message naming
   the cause; its spurious core shrank from 32 to 27 atoms.*

3. **`EmbedMultipleConfs` rejects `coordMap` as a keyword** when the
   `EmbedParameters` overload is used. Put it on the parameters object:
   `params.SetCoordMap(coord_map)`, then
   `EmbedMultipleConfs(mol, n_confs, params)`.

4. **After Vina's local optimisation, its pose buffer is empty.** `poses()`
   and `energies()` return nothing. Write the pose out and read it back, then
   rescore. (Relevant only if you touch optimisation; prefer SARscope's
   existing docking path.)

5. **Vina's local optimisation destroys a core constraint.** On a 3-ligand
   test it moved the pinned core 6.06 Å and 6.71 Å from the reference on two
   of three molecules. If you offer local optimisation at all, you must guard
   it with a drift limit and keep the unoptimised pose when the limit is
   exceeded — reporting that you did so. Without the guard most of that series
   would have been silently misaligned.

6. **NaN masking must happen once over the whole set, before splitting.** Per
   fold, it leaves NaNs in the held-out rows and the fit fails. The mask is
   geometry-only so this is not activity leakage — but variance filtering
   *is*, and must stay inside the folds.

7. **Uncapped PLS components overfit invisibly.** On 18 compounds, unbounded
   selection gave r² 0.96 against q² 0.58; capped at n/5 the same data gave
   r² 0.64 and q² 0.38. The second pair is the truth.

8. **CoMSIA with CoMFA's σ threshold discards nearly everything.** Its indices
   are unitless and ~1000× smaller than kcal/mol. *Measured: q² 0.13 with the
   absolute 2.0 cutoff (170 columns kept) against q² 0.51 with a 30-percentile
   filter (809 kept) on the same data.*

9. **The charge model inverts the chemical conclusion.** Holding alignment,
   grid and every other setting fixed on a 26-compound series and changing
   only the charges: CoMFA/Gasteiger gave q² 0.362, r² 0.727, steric 51% /
   electrostatic 49%; CoMFA/MMFF94 gave q² 0.343, r² 0.702, **electrostatic
   70% / steric 30%**. The statistics barely move, so nothing looks wrong, but
   the contour maps and the medicinal-chemistry story reverse. MMFF94 charges
   were 2.3× larger in magnitude (absolute mean 0.201 vs 0.087). CoMSIA shifted
   by at most 1 percentage point, because four of its five fields do not
   involve charge. **The charge model must be a recorded parameter.**

10. **CoMFA and CoMSIA cannot be compared without matching retained column
    counts.** Their filters are not commensurable. *Measured: CoMSIA at 809
    columns scored q² 0.42 against CoMFA at 178 columns scoring q² 0.36 — but
    held to the same 132 columns, CoMFA scored 0.363 and CoMSIA 0.263,
    reversing the apparent winner.* If the report compares methods, it must
    report the column count alongside. In that same run the CoMFA model's
    scrambled maximum q² (0.392) exceeded its real q² (0.363) — a
    chance-correlation failure. Two models can differ in q² while both being
    untrustworthy, so weight the randomisation distribution over the q²
    ranking.

11. **MCS over hydrogen-bearing molecules compared against heavy-atom counts
    reports >100% coverage.** Strip hydrogens first.

12. **Field computation must be chunked.** The naive
    `grid.points[:, None, :] - xyz[None, :, :]` allocates
    `n_points × n_atoms × 3` floats and exhausts memory on a fine grid.

13. **Never mix deposited and newly added hydrogens.** Naming conventions
    differ, and a protonation tool cannot tell which polar hydrogens exist. In
    one case this placed a C-terminal OXT 1.77 Å from CA, which RDKit
    perceived as a bond, giving an overvalent carbon and failing PDBQT
    conversion with an error pointing nowhere near the cause. Strip hydrogens
    before re-protonating. (SARscope's Meeko path already replaces source
    hydrogens — preserve that behaviour.)

14. **Covalent connectivity cannot be inferred from distance.** In PDB 6LU7
    the N3 inhibitor is a peptidomimetic whose backbone runs through standard
    amino acids in its own chain (`02J-ALA-VAL-LEU-PJE-010`), so its component
    residues sit up to 9 Å apart while being one molecule. Distance-based
    grouping recovered 21 of 49 atoms; the LINK records recover all of it. Use
    `gemmi` `connections`.

15. **The redocking gate earns its place.** On 6W63, a widely used SARS-CoV-2
    Mpro structure, Vina cannot reproduce the X77 crystal pose: every pose in
    the output sits ~6.3 Å away, converged and confidently scored at
    −7.6 kcal/mol, with the centroid only 1.25 Å off — right pocket, wrong
    orientation. Verified thoroughly: self-RMSD 0.0, a 1 Å translation reads
    1.0, RDKit's independent `CalcRMS` agrees at 6.32, coordinate frames match
    to 0.000 Å, the extracted reference ligand reproduces the deposited
    coordinates exactly with all 37 bond lengths plausible, rebuilding the
    dimer changes nothing (the partner protomer contributes one atom within
    5 Å of this ligand), and box size and exhaustiveness are both irrelevant.
    This is a real negative result, not a bug to fix — and it is why pose
    selection by score is the wrong default.

16. **PDB-extracted ligands have no bond orders**, so Meeko cannot type them.
    Fetch the ideal SDF from the PDB Chemical Component Dictionary
    (`https://files.rcsb.org/ligands/download/{CCD}_ideal.sdf`) as a bond-order
    template via `AssignBondOrdersFromTemplate`. Multi-residue entities need a
    user-supplied SMILES; say so rather than guessing.

---

## 4. Validation requirements

### 4.1 The benchmark is mandatory

**The implementation is not complete until it reproduces a published q² on a
literature benchmark.** Everything else only demonstrates that the code does
what its author intended, not that what they intended is correct.

Use the classic steroid / corticosteroid-binding-globulin set (Cramer 1988 and
the many subsequent re-analyses). Source the structures and affinities from a
citable reference or a distributed benchmark dataset — **do not type them from
memory or from a model's recollection**, because half-remembered benchmark
values silently invalidate everything downstream while looking fine. Record the
source in the test and freeze it, in the style of
`tests/test_benchmark_freeze.py`.

Report the q² obtained alongside the published value and explain any gap.

### 4.2 Field correctness tests (required)

A field that ignores its inputs looks pleasingly stable while being
meaningless. Assert the dependence directly:

- Zeroing every partial charge must zero the electrostatic field **exactly**
  (for both CoMFA and CoMSIA).
- Negating every charge must invert the field: correlation with the original
  below −0.999.
- The steric field must be **unchanged** by any charge change.
- Spatial decay: assert the *shape*, not an absolute level, which depends on
  atom count and attenuation. The CoMSIA steric index should fall roughly two
  orders of magnitude per 2 Å — e.g. ~7.2 within 2 Å of the molecule to
  ~1.4×10⁻⁵ in the 6–8 Å shell. Assert monotonic shell-wise decay and a
  far/near ratio below 1e-4.
- Chunked field computation must be numerically identical to unchunked
  (`atol=1e-12`).

### 4.3 Validation behaviour tests

- Y-randomisation must **detect** a chance model: on scrambled activities,
  mean q² near or below zero and no scrambled run reaching the real q².
  A tool that cannot fail this test cannot be trusted to pass it.
- Golbraikh–Tropsha criteria must be reported per model and must **fail** when
  they should: a weak model on a small series should be reported as not
  predictive, not quietly passed.
- The redocking gate must refuse a covalent reference rather than report an
  RMSD.

### 4.4 Style of the output

Follow SARscope's existing discipline: the report says plainly what a result
does and does not establish. A model failing its checks must be reported as
failing, with the blocking failures named, in the generated text as well as
the UI. Do not let a user report a chance model as predictive.

---

## 5. Definition of done

- [ ] `analysis/fields.py`, `analysis/align3d.py`, `analysis/contours.py` added,
      with no duplication of §1.1
- [ ] No new required dependency; optional extras declared; new packages added
      to `TRACKED_PACKAGES`
- [ ] `FieldQsarParams` in `params.py`; every setting recorded by `provenance`
- [ ] Stage wired into `pipeline.py`, `report.py` and the Streamlit app
- [ ] Pose generation gated behind the docking extra, degrading with a clear
      message
- [ ] All tests in §4.2 and §4.3 present and passing
- [ ] **Literature benchmark reproduced, with the obtained and published q²
      both reported** (§4.1)
- [ ] Existing test suite still green; `make` targets and CI pass
- [ ] No GPL source consulted (§1.3)
- [ ] Docstrings in SARscope's house style: explain *why* a choice was made and
      cite the measurement or reference that justifies it, not just what the
      code does

---

## 6. References to cite in docstrings

- **CoMFA** — Cramer RD, Patterson DE, Bunce JD. *J Am Chem Soc* 1988,
  **110**, 5959–5967.
- **CoMSIA** — Klebe G, Abraham U, Mietzner T. *J Med Chem* 1994, **37**,
  4130–4146.
- **External validation criteria** — Golbraikh A, Tropsha A. *J Comput Aided
  Mol Des* 2002, **16**, 357–369.
- **Kennard–Stone selection** — Kennard RW, Stone LA. *Technometrics* 1969,
  **11**, 137–148.
- **AutoDock Vina 1.2** — Eberhardt J, Santos-Martins D, Tillack AF, Forli S.
  *J Chem Inf Model* 2021, **61**, 3891–3898.
- **Meeko / ligand preparation** — cite the current Forli-lab reference.
- **RDKit** — cite the version used, as `provenance` records it.

Also acknowledge the prior art the user may compare against, without using its
code: Open3DQSAR and Open3DALIGN (Tosco & Balle), Cloud 3D-QSAR, and the
`3d-qsar.com` web implementation.

---

## 7. A note on positioning

Field-based 3D-QSAR is not novel — it is 1988 and 1994 methodology, and free
implementations exist (Open3DQSAR, `3d-qsar.com`). What is absent from all of
them is the thing SARscope already does for 2D: validation you cannot skip,
curation that is counted, and provenance that lets someone rerun the work.
Build the 3D methods so that they inherit that, rather than sitting beside it.

The honest claim is "the first 3D-QSAR implementation that will not certify a
chance model, in a library you can `pip install`" — not "better than CoMFA".

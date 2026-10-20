# Generated-alignment validation artifacts

See [the validation report](../../docs/generated-alignment-validation.md).

The separate [external comparison](../../docs/external-3dqsar-comparison.md)
uses RDKit O3A, standalone Open3DALIGN and standalone Open3DQSAR. Compact
protocols, measured results and integrity records are in
`external-comparison-20261019/`. PLS solver agreement is not field equivalence.

- `generated-20261019/`: first successful measured-cohort run; 42 targeted tests,
  before adding the real nonresponding-worker termination test.
- `generated-20261019-py313/`: final Python 3.13/RDKit 2026.09.1 execution;
  43 targeted tests plus all 50 records in both generated routes.
- `generated-20261019-py312/`: final independent Python 3.12/RDKit 2024.9.6
  execution of the same source-hashed checks and membership.
- `full-suite-20261019.xml`: complete non-network regression, 653 passed,
  six skipped and two deselected; retained outside the route-run manifests.

The geometry fixtures are purpose-built controls, not measured binding poses.
An initial test used `C1C[C@]2(O)CC[C@@H]1C2` as a positive hydroxylated bridged
analogue and produced two test failures because it could not embed while
preserving the stated bridgehead geometry. The positive fixture was corrected
to `C1C[C@@]2(O)CC[C@@H]1C2` after deriving its graph by replacing an explicit
bridgehead H in the realizable parent with O, retaining neighbour ordering and
the parent geometry's stereochemical tag. In written SMILES that changes @/@@
because the implicit H becomes an explicit branch; it is not an instruction to
change a user-supplied molecule. The conflicting assignment is now an explicit
**negative** control expected to be refused in both routes. The original failed
JUnit record is retained as `fixture-development-failures.xml`. No production
stereochemistry rule or scientific threshold was disabled to obtain a pass.

The copied EGFR source CSV and generated molecular SDF derivatives retain the
source ChEMBL **CC BY-SA 3.0** terms and attribution recorded in each source.json:
ChEMBL/EMBL-EBI; Bridges et al., J. Med. Chem. 1996,
doi:10.1021/jm9503613. They are not relicensed as MIT. Code and newly authored
test fixtures follow SARscope's code licence. Numerical/geometry verification
does not add experimental measurements or validate a binding mode.

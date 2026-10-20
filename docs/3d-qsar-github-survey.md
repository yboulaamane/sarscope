# Relevant GitHub projects — inspected 19 October 2026

This is a focused metadata, README, public API and licence review, not an
execution benchmark of every project. No algorithm source from GPL projects
was read or ported for the native MMFF profile.

| Project | Declared licence observed | Relevant value | Decision for SARscope |
|---|---|---|---|
| [RDKit](https://github.com/rdkit/rdkit) | BSD-3-Clause | Public MMFF mixed-pair parameters, partial charges, molecular alignments and experimental MIF evaluators | Best immediate fit: already a dependency; public parameters underpin the new profile; MIF API is an independent test oracle |
| [Py-CoMSIA](https://github.com/clhaga/pycomsia) | CC-BY-SA-4.0 in repository LICENSE | Python CoMSIA fields, published example structures and PLS workflow | Useful existing scientific comparator; do not assume its code is MIT/BSD or copy it into our implementation |
| [eCoMFA](https://github.com/poclab-web/eCoMFA) | MIT | Electronic-field QSSR, cube descriptors, regularised regressions and contour/figure workflows | Interesting future offline comparison; focuses on asymmetric reduction selectivity, not a drop-in ligand-bioactivity backend |
| [3DQSARpy](https://github.com/viana1jess/3DQSARpy) | No declared licence detected by GitHub metadata | Published automated 3D-QSAR workflow; offline archives | Possible external comparator, not a code-reuse source without clarified permission and reproducible packaging |

The immediate implementation uses RDKit's public API plus independently
written published equations. It does not add every repository as a runtime
dependency. The other projects were not installed or run in this survey.

eCoMFA's README lists Gaussian/Psi4 calculations and a quantum-field workflow;
that makes it a distinct scientific direction rather than a small Streamlit
feature. A future uploaded-cube workflow could be evaluated without running
quantum calculations on a shared host. Its repository identifies the related
2025 paper, doi:10.1021/acs.jpca.5c03510. This is a potential extension, not
an implemented feature or a new validation result.

Sources:

- [RDKit licence](https://github.com/rdkit/rdkit/blob/master/license.txt) and
  [public MIF API](https://www.rdkit.org/docs/source/rdkit.Chem.rdMIF.html).
- [Py-CoMSIA README](https://github.com/clhaga/pycomsia) and
  [licence](https://github.com/clhaga/pycomsia/blob/main/LICENSE).
- [eCoMFA README/licence declaration](https://github.com/poclab-web/eCoMFA).
- [3DQSARpy repository](https://github.com/viana1jess/3DQSARpy) and GitHub's
  public repository metadata (`license: null` when inspected; not a legal
  determination about material hosted elsewhere).

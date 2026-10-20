# Original aligned 3D inputs

| Series | Train / held-out | Membership |
|---|---:|---|
| Thrombin | 59 / 29 | `THR_source_membership.csv` |
| AChE | 74 / 37 | `AChE_source_membership.csv` |
| Steroid/CBG | 21 valid / 10 | `STEROIDS_source_membership.csv` |

IDs should be read as strings to preserve leading zeros. Deposited log activities
are unchanged, **not automatically pIC50**. Steroid membership retains the original
`test` label; the other tables use `held_out` for the same partition role.
The steroid source also contains an
invalid training record; the original source-accounting disclosure is retained
in [the evidence](../../../benchmarks/field_qsar/steroid-alignment-reproduction-20261019/).

Download original SDFs explicitly, from the repository root:

```bash
python examples/paper_data/fetch_3d_inputs.py --out runs/paper-inputs
# Or just the thrombin data:
python examples/paper_data/fetch_3d_inputs.py --dataset THR --out runs/paper-thrombin
```

The standard-library script verifies source hashes, preserves original coordinates
and refuses to replace different files. It downloads only molecular data, not
external algorithm source. The separate [licences](../DATA-LICENSE.md) apply.
No downloads occur on import or app startup.

For the exact thrombin preset, use the workbench's opt-in measured thrombin
example and [worked-example protocol](../../../docs/end-to-end-3d-demo.md).
Other uploaded structures/splits are a new analysis, not a reproduction merely
because the compound names match. See [recorded evidence](../../../benchmarks/field_qsar/README.md)
for field settings, comparisons, permutations and sensitivity results.

# Worked-example testing record

The completed numerical experiment remains frozen in
`runs/end-to-end-demo-20261019/`; the compact report is
`../end-to-end-demo-20261019-v2/`. Its source snapshot retains the exact
pre-execution numerical and UI sources.

- `initial-tests.xml`: 82 passed, one failed. Both primary profiles had
  already reproduced their expected numerical metrics. The Gaussian test
  later failed because the split widget was absent after preparing the
  older CCR5 example during its final example-switch cleanup step. The
  original output is retained. The cause was not established by that output.
- An independent reproduction of the example-switch/preparation sequence
  completed without an app error and showed the expected reset split widget.
- The unchanged numerical workflows were repeated serially. Both full
  Streamlit primary-data paths—including actual 99-permutation fitting—passed,
  along with all selected geometry, worker, export and evidence tests:
  **87 passed** (`repeat-tests.xml`). An assertion was added to expose the
  actual preparation error if that final example-switch step fails again;
  it does not bypass preparation or alter its limits.
- The UI's post-freeze warning was made more explicit: the three offline
  MMFF pose-stress failures are stated directly instead of pointing vaguely
  at offline diagnostics. The caption also now meets the line-length rule.
  That is the only source-hash change from the frozen numerical run; field,
  alignment, regression, settings and export code are unchanged. No score,
  seed, mapping, stereochemistry, timeout or acceptance threshold was relaxed.
- Targeted Ruff E/F/I and `git diff --check` pass. These targeted checks are
  not represented as a new full-repository test-suite run.
- `related-tests.xml`: a further **64 passed**, covering field UI, MMFF
  equations, interpretation, sensitivity, reproduction and earlier steroid
  evidence. Together with the successful 87-test rerun this gives 151 passing
  targeted tests, not a claim about unrelated repository areas.

The demo is a positive **fixed-alignment** predictive example, not a claim
that every alignment or every app interaction succeeds for all chemistry.
The failed CCR5 CSV/constrained case and failed pose-sensitivity checks stay
in the scientific evidence.

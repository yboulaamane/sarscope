# Local 0.2.0rc1 candidate check record

Historical preparation record: commit `28bc9fb` was subsequently pushed to
public main on 20 October 2026. The execution outcomes and preparation-state
descriptions below are preserved as recorded, not rewritten as new runs.
Current publication status is in `docs/reproducibility-release-candidate.md`.

Prepared on Python 3.13.13 / Linux, with NumPy 2.5.4, pandas 3.0.6,
SciPy 1.18.1, scikit-learn 1.9.1, RDKit 2026.09.1, Streamlit 1.66.0
and Sorbent 0.1.1. This is a candidate engineering/reproducibility check,
not a new predictive benchmark, public release or package-wide certificate.

## Executions retained

| Record | Outcome |
|---|---|
| `initial-interrupted-tests.xml` | Interrupted intentionally to synchronize version metadata; 86 passed, one UI reset failure and one skip were retained. Not a complete run. |
| `offline-tests.xml` | First complete non-network run: 693 passed, one example-UI preparation failure, nine skipped and two network tests deselected. |
| `isolated-ui-tests.xml` | Both affected UI checks passed separately; two passes. |
| `offline-tests-repeat.xml` | Serial complete repeat: 695 passed, nine skipped, two network tests deselected. Version-consistency check added; UI assertions gained error text, not weaker acceptance criteria. |

Two small-class warnings in the separate 2D example were retained. The example
preparation/UI failures did not recur in the serial repeat; their underlying
cause was not established. The initial failures must not be silently replaced
by the successful repeat. No test was skipped or weakened to obtain that repeat.
Optional-dependency skips and excluded live-network calls are not tested claims.

Ruff lint and format checks, mypy and dependency consistency checks passed.
The first build attempt without isolation lacked setuptools in the shared
runtime; standard isolated building succeeded. Source/wheel build artifacts
are local; no upload was made. `MANIFEST.in` includes the app entry point,
comparison drivers, evidence, example builders and docs in source distributions.

The original numerical/alignment/parameter modules were checked byte-identical
to the pre-existing tested working tree. Comparison scripts received mechanical
formatting/import cleanup and two type annotations; no scientific calculation
or result was changed. Their historical source hashes remain in frozen records.
Two unrelated, unused assay-data draft modules, local BRAF input drafts and
the user's submission DOCX copies were deliberately excluded from the candidate;
all remain untouched in the original checkout.

The candidate uses a new branch from public main. Original checkout contents,
diverged history, historical manifests and test records were not rewritten.
It has not been pushed or tagged. See `docs/reproducibility-release-candidate.md`
for installation, provenance and reproduction instructions.

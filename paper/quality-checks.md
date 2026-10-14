# Local verification record — 12 October 2026

This records local execution, not GitHub CI, expert review, comparator usability
testing or a published release. The Git base is
`7a8e4c805d671d92bccf38673bd00bc2207ad7cd`; publication changes are uncommitted
at preparation time. The archive manifest records actual file hashes.

| Check | Observed result |
|---|---|
| `ruff check .` | Passed |
| `ruff format --check .` | Passed (110 Python files) |
| `mypy src/sarscope` | Passed (51 source files) |
| `git diff --check` | Passed |
| `pytest -q -m "not network"` | 390 passed, 3 skipped, 2 deselected, 5 warnings; 216.15 seconds |
| Final targeted publication/SAR UI/SAR-map/environment-lock tests | 16 passed |
| Offline case freeze/evaluation | 3 cases; all 273 positional pair records matched exhaustive enumeration |
| Actual mmpdb 3.1.4 comparison | Completed for all 3 curated assay cohorts; all 760 unique SARscope pairs shared with mmpdb, plus 56 mmpdb-only H-substitution pairs |
| Saved-prediction uncertainty audit | Completed; seed 42, 1,000 scaffold-cluster bootstrap replicates per target |
| Scientific graphics | Three SVG figures rendered to PNG and visually inspected; workflow text fitted and benchmark target labels shortened |
| Clean isolated environment | Installed direct pins and SARscope source into a new venv; all 26 case CSV outputs matched, excluding elapsed_seconds |
| Runtime lock | Full pip resolution with transitive versions and wheel SHA-256 hashes; Linux x86_64/Python 3.13; source installed separately |

The full suite produced UMAP warnings about unsupported Jaccard inverse transforms
and disconnected points on test inputs. These were warnings, not failed checks.
No live ChEMBL requests or new docking/model-training benchmark runs were part
of publication evaluation. Installing optional mmpdb was confined to a temporary
evaluation directory, not the app environment.

The full suite preceded final descriptive-caption and CSV-reader refinements;
the final targeted tests cover those paths and the environment-lock exporter.
Independent expert judgments,
literature-trend recovery, DataWarrior task timings and a release DOI are absent.

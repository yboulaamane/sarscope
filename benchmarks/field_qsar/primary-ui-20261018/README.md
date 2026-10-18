# Primary-source example: Streamlit verification

The expanded offline suite completed with **627 passed, 6 skipped and 2 network
tests deselected**. `offline-suite.xml` records the 633 executed/skip test cases.
The six skips concern optional UMAP, molfeat, SHAP and real Vina integrations.

`test_primary_ui_is_opt_in_and_actual_profile_reproduces_native_scores` executes
the real Streamlit preparation/run path and 99-permutation native fit. Its JUnit
properties retain actual field settings, source/partition metadata, source-code
SHA-256 hashes and recalculated q²/R²/RMSE. These tested module hashes were
independently checked against the pushed source files before committing.

Results: pooled nested q² **0.6070519813940478**, held-out predictive R²
**0.5071821301111985**, RMSE **0.771570968908504**. No saved model result replaces
the fit. Downloading is mocked with the separately obtained, hash-verified
primary archive, not synthetic molecules or activities. Raw chemical data are
not bundled here and retain their separate publisher licence.

The test also covers explicit loading/running, source-state retention, the full
59/29 supplied partition, subset refusal, failed alignment qualification and
ordinary-limit restoration when switching examples. AppTest does not render
WebGL or prove independent human alignment review. This is not full predictive
or external validation; see the [worked-example guide](../../../docs/primary-thrombin-worked-example.md).

Reproduce with the separately licensed primary archive available:

```bash
SARSCOPE_PRIMARY_ARCHIVE=/absolute/path/jm0497141_si_002.zip \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -m pytest -q -m 'not network' --junitxml=runs/new-offline-suite.xml
```

Without that archive the two primary-data integration tests explicitly skip;
the synthetic source/partition contract tests still run. No source download
occurs automatically during the suite.

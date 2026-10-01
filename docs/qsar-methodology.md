# QSAR representations and regression diagnostics

This independently implemented workflow is inspired by
[Patrick Walters' regression notebook](https://github.com/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/regression_model.ipynb),
[his model-comparison notebook](https://github.com/PatWalters/practical_cheminformatics_tutorials/blob/main/ml_models/comparing_regression_models.ipynb),
and [Esben Bjerrum's Scikit-Mol pipeline example](https://www.cheminformania.com/scikit-mol-easy-embedding-of-rdkit-into-scikit-learn/).
The patterns are diagnostic error plots, useful references and reproducible
featurization. Neither author has endorsed or validated SARscope.

## Representations

Choose ECFP4/MACCS, named RDKit 2D properties, or fingerprints plus descriptors
in the ML tab. Presets contain 12 physicochemical properties, 27 medicinal-chemistry
properties where available, and the installed RDKit 2D catalog excluding
Ipc/AvgIpc, which can be slow or unstable on large graphs. The available count
depends on RDKit's version. Edit the ordered list before fitting. QED and other
structural descriptors are not potency, efficacy or safety evidence.

Descriptor/hybrid models median-impute missing/nonfinite values, standardize,
remove constants and filter correlations using training rows only, refitting
inside every validation fold. Entirely missing training columns become zero
and are removed as constants. Fingerprint-only models retain their existing
variance/correlation filter. Nearest-analogue similarity and novelty diagnostics
always use binary fingerprints, never Tanimoto on real-valued descriptors.
The PCA bounding-box diagnostic instead uses the selected, training-preprocessed
model features; it is a distinct and more permissive criterion.

Prediction bundles save ordered names, settings, fitted preprocessing and the
estimator. The feature-schema CSV describes the deployment refit's retained
columns; fold selections can differ. Deployment is refitted on the modeling
dataset, while validation predictions were made on the untouched outer test.

## Optional Molfeat

`pip install 'sarscope[molfeat]'` enables Molfeat's RDKit 2D backend. The adapter
disables augmentation, IPC averaging and implicit standardization to preserve
the schema and already-curated structures. It downloads no pretrained weights.
[Molfeat](https://molfeat-docs.datamol.io/main/index.html) currently requires
PyTorch in its core, so it is excluded from the default cloud requirements.
Native RDKit supplies the named properties without Torch or a GPU. The real
optional-backend test is skipped when Molfeat is absent.

## Regression outputs

Select requested regressors by training CV RMSE, not test scores. The reference
predicts the training-fold mean on the same folds/test partition. Ridge is a
regularized linear reference alongside RF, Extra Trees, boosting, SVR, neighbors
and MLP. Metrics include train/CV/test R², RMSE, MAE and Spearman.

Held-out diagnostics include parity/residual plots, median absolute error,
typical fold error `10 ** median_absolute_error`, bias (measured minus predicted),
fractions within 3-fold/10-fold potency error and baseline improvement. Three-fold
means `abs(error) <= log10(3)`, not three log units. Dashed parity guides are
error boundaries, not confidence bands. Fixed structural-similarity bins show
counts and errors. Regression-only runs can report cliff-compound error.

Optional 500-repeat scaffold-block bootstraps resample whole held-out chemical
series; acyclic molecules form individual blocks. Model/baseline comparisons
use paired draws. Approximate 95% metric intervals are conditional on the fitted
model/test population and omit model-selection uncertainty. Few independent
blocks make them unstable; these are not prediction intervals or guarantees
under target/temporal shift. The existing training-CV residual band also has
no formal coverage guarantee: inspect observed test coverage.

Bootstraps are separately downloadable. Parity/residual figures, diagnostic
tables, novelty summaries and the feature schema enter the report ZIP.

## Limits

The browser applicability-domain panel overlays training and held-out points on
the training-fitted PCA ranges, with a 1D interval when only one component is
available. For regression it also shows nearest-training Tanimoto similarities
and the separate training-only cutoff (5th percentile of training compounds'
nearest non-self similarities), plus observed inside/outside regression errors.
All metrics and boundaries use the full partition; plots cap each partition at
2,000 points. Coordinates and per-compound flags are downloadable from the panel.
100% PCA coverage is not prediction accuracy: sparse regions, omitted dimensions,
activity cliffs and incompatible assays can still cause failures. These browser
diagnostics are not calibrated reliability probabilities. The coordinate CSV is
a separate browser download, not an additional report ZIP artifact.

Choose features/models before looking at test performance. Repeatedly adjusting
settings against test scores makes that test development data; reserve a new
independent evaluation before claiming improvement. More features cannot repair
incompatible endpoints, assay contexts or duplicate evidence.

Explain fits a separate descriptor model on the established partition. It
provides selected properties, held-out permutation importance and optional RF
TreeSHAP, not an explanation of the fingerprint winner or biological causality.
Browser explanations cap descriptors at 50 and TreeSHAP at 100 held-out rows.

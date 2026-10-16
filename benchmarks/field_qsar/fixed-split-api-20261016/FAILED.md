# Export-metadata failure, not a fitted-model failure

The reference's complete 1–12-component search and explicit 21/10 fit
completed. Its CSVs and processed arrays were exported. The wrapper then
failed to JSON-encode a NumPy integer in the estimator's metadata.

The corrected wrapper converts NumPy scalars to JSON scalars; no numerical
input, parameter, split, component count or fitted prediction was changed.
This directory remains incomplete and is not the final assessment.

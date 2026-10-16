# Incomplete diagnostic attempt

Both block-scaled field-source models completed and their scores/predictions
were retained. The comparison then failed because the diagnostic looked for
`cv_pooled_q2` on the generic regression score table. That derived quantity
is normally added by the field workflow, not `evaluate_regression` itself.

The corrected probe explicitly recomputes pooled PRESS/SST q² from all 21
retained LOO predictions. It does not change fitted predictions, charges,
fields, filtering or PLS. This directory has no completed assessment and
must not be treated as a finished parity run.

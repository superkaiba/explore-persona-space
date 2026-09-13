# Paired final comparison

`scripts/workspace_jr_compare.py` consumes a manifest of completed main fit
roots and their immutable upload receipts. Each cell declares `role`, `kind`
(`observed` or `affine_null`), `k`, `rotation`, `root`, `fit_relative`,
`upload_receipt`, and `terminal_relative`. The manifest also binds the frozen
configuration and selection SHA256 values. The complete grid contains 48
cells: two models × observed/null × four orientations × three sparsities.
An explicit `--allow-incomplete` produces an incomplete-grid diagnostic and
lists every missing cell; it does not satisfy experiment completion.

The consumer verifies source hashes, successful producer terminals, actual
per-example targets and predictions, test-input fingerprints, and reproduced
R²/SSE/SST. Within a model, observational cells must share their extraction,
readiness, dictionaries, and all-split context-input/full-target fingerprints.
Every null must trace to the included native k10 full-answer ridge fit,
including its coefficients, input proof, dictionary and split membership.

Within-model analyses use that model's explicitly reported common context
cohort. Cross-model analyses use the common cohort across both models and
report excluded IDs. Each target receives its own resampled variance
denominator. A single set of 2,000 context bootstrap draws is shared across
lenses, targets, predictors, controls and models within a reported scope.
These intervals are conditional on fitted predictors; they do not bootstrap
training, calibration or the distribution of all possible dictionaries.

The primary control contrast subtracts the mean gap from the three registered
rotations. Rotation minimum, maximum and sample standard deviation are
reported separately from the context-bootstrap interval. The native gap
minus the actual-dictionary affine-null gap is a descriptive diagnostic,
not a causal correction. Undefined targets or resamples retain null intervals;
missing rotations cannot be averaged as a smaller or zero-filled control set.

Optional `quality_matches` entries provide each role's quality artifact root
and upload receipt. The consumer verifies all four calibration summaries,
their common capture/numerical contracts, dictionary and readiness; it then
reproduces the calibration-only nearest-quality decisions. Only approximate
within-range matches enter supplementary comparisons. Outside-range and
zero-energy matches retain their exclusions and signed matching discrepancies.
The same-k comparisons remain primary.

Each completed cell has an exact-source checkpoint. `--resume` requires the
same manifest and actual source-proof bytes, including quality reports and
receipts. Mixed fresh/resumed cells normalize context-ID serialization before
checking exact pairing. Completed analyses remain immutable. These mechanics
implement the frozen analysis plan; they do not change sampling, tuning,
the 0.05 practical-gap threshold, or interpretation rules.

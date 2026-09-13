# Component statistics and rollout weighting

The post-fit `workspace_jr_analyze.py statistics` phase accepts every registered
rotation and sparsity. It validates the same persisted component files, canonical
inputs, fitted targets and predictions as native direction diagnostics, then
writes per-context decomposition statistics and component-specific rollout-noise
reports for train, validation and test. Native direction readouts remain in the
`diagnostics` phase because their J/R vocabulary directions must be unrotated.

Each token statistic includes its uniform-token mean (`mean_token`, also retained
as the legacy `mean`), each rollout's mean, and their equal-weight average
(`mean_equal_rollout`). The latter matches the weighting of the fitted targets
when answer lengths differ. Token counts and totals remain explicit. Invalid or
empty rollout lengths, nonfinite values and length mismatches fail. These are
summaries of the saved token decompositions; no decompositions or predictors are
refitted by this phase.

The `noise_*.json` reports retain within-context variability and the covariance
between each component and its remainder. A noise fraction above the frozen 0.1
threshold triggers the separately planned higher-K diagnostic; zero observed
variance stays undefined. Sampling variability is not a reasoning measure.

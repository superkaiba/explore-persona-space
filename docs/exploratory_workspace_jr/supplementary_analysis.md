# Supplementary analysis on the completed paired cohort

The primary comparison conditions on five nonempty final completed rollouts in
both models, as declared in `completion_scoring_20260913.md`. Diagnostics initially
save the full captured test set. `scripts/workspace_jr_supplement.py` rescales the
saved diagnostics and learning predictions to the exact ordered context IDs in
the completed 48-cell comparison. It does not refit, retune, replace excluded
contexts, or choose directions using test outcomes.

## Inputs and integrity checks

Run with `--manifest MANIFEST.json --out FRESH_DIRECTORY`. The manifest contains
`comparison: {root, upload_receipt}` and `models`, keyed by `primary` and
`comparison`. Each model contains `diagnostics: {root, upload_receipt}` and
`learning_curves: {root, upload_receipt}`. These are completed outputs of
`workspace_jr_analyze.py`, using the corresponding observed native k=10 fit.

Every consumed file must match its verified upload receipt. The reader binds
the diagnostic and learning sources to the native fitted artifacts used by the
main comparison, checks the actual analyzer implementation, and verifies the
completed-rollout ledger and all three comparison scopes. Direction eligibility,
bases, training variances and no-replacement control matches remain fixed on the
original training/calibration data. Learning curves must use the frozen training
prefixes, full validation set and full-training selected MLP recipes.

## Saved measurements

The output includes paired-cohort component similarity and reconstruction
identities; within-context rollout variability and component-specific noise
fractions; same-token J/R readout scores and variance-matched controls; and
learning curves from saved per-example predictions. Component/remainder
covariances and undefined scores remain visible. Near-zero component cosine
exclusions use the previously fixed training norm floor.

All intervals use 2,000 paired context bootstrap draws, the registered seed and
95% confidence level. Each target has its own variance denominator. Context
resampling is shared across targets, lenses, predictors and models; variation
across directions or random dictionaries is not relabeled as context uncertainty.

The report stores source hashes, its own implementation identity, the common
context IDs and all resulting estimates. The completion marker binds the report
and input manifest. The conditional higher-K decision is recomputed on this same
cohort; a triggered follow-up must retain the original fitted predictors.

The registered identity-plus-learned-bias and nearest-neighbor retrieval
references are also rescored on this cohort. Their predictors remain frozen;
retrieval uses only the retained targets as the candidate pool and records its
size and chance levels. R² receives the same paired context bootstrap. Retrieval
accuracy remains a descriptive point estimate, with no implied confidence
interval. This is the same reference recipe already used in the original fits.

## Figures

`scripts/workspace_jr_plot_supplement.py --input-root SUPPLEMENT_ROOT
--upload-receipt RECEIPT --out FRESH_FIGURE_DIRECTORY` produces six views:
decomposition agreement, sampling noise, readout-score distributions,
predictability versus training target variance, and one learning-curve figure
per model. Exports use the project plotting style and include PDF, color PNG,
grayscale PNG and metadata with exact plotted records, source hashes, the actual
renderer hash and output hashes. Undefined intervals are labeled; percentile
interval endpoints are drawn directly even when they exclude the point estimate.

These diagnostics describe relative predictability within the completed paired
population. They do not establish automaticity, reasoning, a causal capability
effect, or a disjoint partition of J-space and R-space.

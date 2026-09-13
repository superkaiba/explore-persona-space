# Decomposition statistics on the final scoring cohort

`scripts/workspace_jr_decomposition_summary.py` consolidates the requested
sparsity, reconstruction, variance, covariance and prediction-error reporting.
It consumes the completed 48-cell comparison and all 24 observed-cell statistics
artifacts. Native k=10 uses the existing diagnostics artifact; other observed
cells use their statistics artifacts. Affine-null cells use their saved
decomposition arrays and original rollout layouts. This is a reporting extension
of the frozen plan, with no new fitting, selection, matching or exclusion rule.

The summary requires the exact scoring cohort from the finished comparison:
the intersection of joint completion eligibility and all 48 fitted cell cohorts.
It recomputes that intersection and verifies every recorded exclusion before
writing any cell summaries. Each fit must match the comparison's immutable
source proof. Each
observed statistics artifact must match that fit's input manifest, predictions,
results and component-file ledger, the reviewed analyzer implementation, and its
successful uploaded terminal. Null statistics must be the actual prepared
sources consumed by the null fit. Missing cells, changed files, duplicate cells,
changed context order and inconsistent rollout weighting fail explicitly.

For token sparsity, average within each rollout, then average rollouts equally
within each context, then average contexts equally. The residual-energy fraction
is the ratio of the two equally weighted mean energies,
`E_context E_rollout E_token ||h-s(h)||² / E_context E_rollout E_token ||h||²`.
It is not an average of context-specific ratios. These saved sparse-algorithm
diagnostics use FP32 arithmetic. A secondary uniform-token summary makes length
weighting visible. The affine null repeats its single context vector over every
original token position and rollout; the algebraic reduction must agree with
explicit repetition, including under unequal rollout lengths.

Centered component variance is measured separately on the original context-level
pooled answer targets. It uses population normalization (divide by context
count), matching the existing component metric helper. The report includes full,
component and remainder variances, twice their centered covariance, and the
reconstruction error. The component/full variance ratio is descriptive: these
decompositions are not orthogonal projectors, and component and remainder
variances need the covariance term to recover full variance. Uncentered token
residual energy is not labeled centered token variance explained.

All five targets also retain component-specific SSE, SST, R² and bias statistics
for ridge and the saved MLP predictors. These are descriptive point estimates;
the paired confidence intervals remain in the main comparison. Zero input energy
and zero pooled full variance produce explicit flags and undefined ratios,
not zero estimates. Sparse-step counts and contexts with increasing-error steps
remain visible.

The input manifest has a `comparison` source (`root`, `upload_receipt`) and an
`observed_statistics` list. Each statistics entry includes `role`, `k`,
`rotation`, `root` and `upload_receipt`. Run in a clean checkout with the matching
native runtime:

```bash
python scripts/workspace_jr_decomposition_summary.py \
  --manifest /absolute/path/decomposition_summary_sources.json \
  --out /absolute/path/fresh_decomposition_summary
```

Use the worker runtime's Python executable; the shared VM uses `uv run --no-sync
python` and is not the production native-analysis environment. Each completed
cell is checkpointed before the next cell's arrays are analyzed. The final JSON,
96-row CSV, per-context token-statistic arrays and hash-bound completion markers
are saved together. The final completion marker binds all 48 cell markers,
which bind the cell summaries and per-context arrays. A failed partial summary
remains separate from a fresh retry.

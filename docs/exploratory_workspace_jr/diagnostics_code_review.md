# Independent diagnostics review

Codex reviewer `native_lens_review` independently inspected the learning-curve,
readout/noise and checkpoint changes on2026-09-12. No Claude was used and the
reviewer read no experimental outcomes or remote workers.

The review found missing full-fit configuration/recipe binding, legacy pilot
results lacking the new input fingerprints, insufficient snapshot rank
ownership checks, duplicate-context bootstrap eligibility, and floating-point
centering that could manufacture PCA rank/noise for bitwise-constant arrays.
All were corrected and closure was independently verified. Full fits now bind
exact input bytes/IDs, fit/statistics/seeds and selected recipe bytes. Prefix
curves retain the full-data validation-selected MLP recipe and all three seeds.
The immutable old pilot will require a separately labelled full-fit rerun for
curves; its original results will never be retrofitted with new provenance.

Readout random/PCA selection and variance matching use training targets only.
Scalar R² bootstrap samples contexts with all lenses/predictors paired and
re-centers each scalar direction in each draw. A direct numerical oracle agreed
across a chunk boundary. Zero-variance resamples remain undefined with counts.
Anchored centering prevents false rank/noise; constant K5 data produce exactly
zero noise and no higher-K trigger. Noise includes component/remainder
covariance and checks the variance reconstruction identity.

Four focused tests passed, Ruff passed, and both launcher shell syntax checks
passed. Scoped workflow lint passed for the four implementation paths (40
checks). This is not a claim that whole-repository lint passed. These helpers
still require artifact-bound runners and actual main data; no scientific
comparison is reported as completed by a code test.

The comparison native startup now requires an explicit model role and keeps a
separate output root. The reviewer found no launcher blocker; STOP duration,
disk auto-delete and instance ownership remain live provisioning checks.

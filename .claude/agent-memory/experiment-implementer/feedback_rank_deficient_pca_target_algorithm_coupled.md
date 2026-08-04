---
name: Rank-deficient PCA target makes a bootstrap-refit estimator algorithm-coupled
description: Before vectorizing any per-resample top-k PCA/SVD pipeline, probe the target's centered spectrum at the k boundary — rank < k means the registered estimator includes the SVD's arbitrary null-basis and CANNOT be batched exactly
type: feedback
---

Before vectorizing a per-resample top-k PCA → ridge → back-project pipeline
(the #722/#833 `make_refit_pair` refit-floor family), probe the FULL data
target's centered Gram spectrum at the k boundary FIRST (one n×n `eigvalsh`,
~ms). If centered rank < k (or the k/k+1 boundary is degenerate), the
registered estimator is ALGORITHM-COUPLED: the extra "top-k" directions are the
SVD's arbitrary null-basis, whose resample-to-resample variation is genuinely
part of the measured statistic — no alternative factorization (Gram eigh, etc.)
reproduces it, per-pair values differ macroscopically, and only a bit-faithful
serial fallback preserves semantics. **Why:** #833 r5 — V0 (base-era answer
profiles) is target-keyed only → centered rank ≈ n_targets−1 = 29 < TARGET_DIM
64; the m0/shift floors' ~35 junk dims made them un-batchable (all-fallback),
and the finding doubled as a science concern (floor possibly inflated by
junk-basis variance). **How to apply:** (1) rotation-invariance batching of a
top-k composite is exact ONLY on certified well-separated boundaries — gate on
`s_k/s_max` and the k/k+1 gap (used 1e-8 relative), plus a one-shot full-data
pre-check to skip wasted batched attempts wholesale; (2) also raise the
rank-deficiency itself as a concern — it usually means the registered
estimator is measuring something nobody intended. Second lesson from the same
round: batched small-matrix torch CPU ops (bmm/eigh over a (B,480,480) batch)
hold ~1 core while a serial fat-matrix gesdd threads ~3 — wall speedup under
contention was only ~1.7× (5× core-normalized) vs the ~50× FLOP estimate;
measure, and pair vectorization with process-level per-cell fan-out.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Rank-deficient PCA target = algorithm-coupled estimator](feedback_rank_deficient_pca_target_algorithm_coupled.md) — probe the centered spectrum at the k boundary BEFORE vectorizing a per-resample top-k SVD pipeline; rank<k means the SVD null-basis IS part of the statistic (serial-only) + raise as science concern; batched small-matrix CPU torch ≈1 core vs threaded serial gesdd (#833 r5)

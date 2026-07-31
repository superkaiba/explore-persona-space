---
name: numpy.linalg.svd non-convergence on bootstrap resamples
description: numpy gesdd SVD can fail to converge on rank-deficient bootstrap resamples in small clusters; use scipy gesvd fallback + per-pair skip-guard
type: feedback
---

`np.linalg.svd` (LAPACK `gesdd`, divide-and-conquer) can raise
`LinAlgError: SVD did not converge` on **rank-deficient inputs produced by
bootstrap resamples in small clusters** — a family-clustered resample-with-
replacement that draws the same small set of families repeatedly is
mean-centered to a near-singular matrix `gesdd` chokes on. Data-dependent and
deterministic on the same resampled indices, NOT a hardware/GPU issue, so it
re-crashes identically on a fresh pod (do NOT failover to RunPod — it's a
code fix).

**Why:** `gesdd` trades robustness for speed; the QR-based `gesvd` driver
converges on inputs `gesdd` cannot.

**How to apply (two-layer fix, both needed for any PCA/SVD inside a bootstrap
refit loop):**

1. In the SVD call site: `try np.linalg.svd(...) except np.linalg.LinAlgError:
   from scipy.linalg import svd; svd(..., lapack_driver='gesvd')`. Same SVD of
   the SAME matrix — no regularization, no resample skip — so the clean path
   stays bit-identical and only the rare degenerate resample takes the slow
   driver. (Prefer this over: tiny ridge on the centered matrix, which changes
   the basis numerically = a science change; or resample-with-new-seed, which
   perturbs the clustered draw the floor depends on.)
2. In the bootstrap loop ITSELF: wrap each pair/draw in
   `try/except np.linalg.LinAlgError` → **skip + count** the failed draw (don't
   crash the whole fit); raise loud only if EVERY draw fails (an empty result
   crashes the caller's `np.percentile`). Track the skip rate via an optional
   `skip_counter` dict out-param (keeps the ndarray return non-breaking) and
   surface >5% lost as a CONCERN, not silent loss. The fallback in (1) handles
   most cases; (2) is the belt-and-suspenders for the rare resample where even
   `gesvd` cannot converge.

Incident: #722 round 3 (2026-06-28) — `_pca_basis_v0`'s unguarded
`np.linalg.svd` crashed the whole GCP fit at sycophancy L7 on a degenerate
M⁺-refit bootstrap resample; the 3 em cells had fit clean.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [numpy SVD non-convergence on bootstrap resamples](feedback_numpy_svd_nonconvergence_bootstrap.md) — np.linalg.svd (gesdd) raises LinAlgError on rank-deficient bootstrap resamples in small clusters; fall back to scipy gesvd AND per-pair skip-guard the bootstrap loop (count skips, >5% = CONCERN). Deterministic on same data → code fix, NOT RunPod failover. #722 r3.

---
name: Batched np.linalg.solve dies on ONE singular slice — per-slice pinv fallback + cell flag
description: numpy raises a single LinAlgError for a whole (B,k,k) stacked solve when any one slice is singular; small ridge jitter is scale-absorbed in fl64; fix = batched-first, per-slice re-solve with pinv on the raise, flag the cell (#1739 r10)
type: feedback
---

Batched `np.linalg.solve` on a (B, k, k) x (B, k, 1) stack raises ONE
`LinAlgError: Singular matrix` for the ENTIRE batch when ANY single slice has
an exact zero pivot — one collinear/constant feature triple in one layer
killed #1739's sycophancy fits lane 25.2 h in (arm-10 stacked combiner,
`arms.py:604`). A small additive jitter (`+ 1e-8*np.eye(k)`) does NOT protect
when feature scale is huge: the jitter is absorbed in float64 (`nC² + 1e-8 ==
nC²`) and elimination still hits an exact zero pivot.

**Why:** the fix pattern that survived review: try the single batched solve
(healthy path bit-identical, keeps the vectorize-first shape); on the raise,
re-solve per slice — healthy slices via `np.linalg.solve` (same gesv),
singular slices via `np.linalg.pinv(ata) @ atb` (min-norm LS) — and return
the degenerate indices so the caller flags the persisted per-cell record
(`degenerate_ols: true` + fold→layer detail), never a silent placeholder.
Reject a cond/det pre-screen or a bigger jitter when completed cells will be
RESUMED: both change healthy-path numbers vs already-persisted records.

**How to apply:** any stacked/batched dense solve over per-layer/per-fold
Grams (`arms._solve_stacked_normal_eqs` is the worked example, commit
aff188af67df5f4e12f6398583d54e36834f3602). Deterministic singular test
fixture: Gram of a constant-column design (col1 = 2*col0, integer-exact) —
gesv raises reproducibly. Sibling entries: cuSOLVER eigh CPU fallback
(#1335), numpy SVD gesdd non-convergence (#722 r3).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Batched solve dies on ONE singular slice](feedback_batched_solve_singular_slice_pinv_fallback.md) — np.linalg.solve raises ONCE for a whole (B,k,k) stack; jitter scale-absorbed in fl64; batched-first + per-slice pinv + degenerate_ols flag (#1739 r10)

## Merged sibling index rows (#2032 curation, 2026-08-03)

This entry is the PRIMARY index pointer for its theme; the sibling index rows below were merged into one index row to fit the agent-memory index size cap (task #2032). Each merged row is preserved verbatim — follow its pointer for the sibling lesson's own entry file.

- [cuSOLVER eigh non-convergence CPU fallback](feedback_cusolver_eigh_nonconvergence_cpu_fallback.md) — cuda eigh raises LinAlgError on (#1335)
- [numpy SVD non-convergence on bootstrap resamples](feedback_numpy_svd_nonconvergence_bootstrap.md) — np.linalg.svd (gesdd) raises LinAlgError on (#722)

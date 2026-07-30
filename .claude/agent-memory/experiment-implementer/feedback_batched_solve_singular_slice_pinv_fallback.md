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

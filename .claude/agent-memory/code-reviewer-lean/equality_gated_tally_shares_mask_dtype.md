---
name: equality-gated-tally-shares-mask-dtype
description: A diagnostic COUNT equality-asserted against a set built by a dtype-sensitive mask must use the mask's exact dtype+scalar comparison; fp64-casting the tally diverges at exactly float32(threshold)
metadata:
  type: feedback
---

When a summary tally is EQUALITY-gated against a set built by a dtype-sensitive
mask (e.g. `assert n_quarantined_rows == n_ge_threshold` where quarantine comes
from `fp32_sims >= PY_FLOAT_THR`), the tally must run the IDENTICAL comparison:
same array dtype, same Python-float scalar → same ufunc dispatch, robust across
numpy versions. Counting on an fp64-cast copy diverges at exactly
`float32(thr)` — e.g. `float32(0.95) = 0.949999988079071` passes the fp32
`>= 0.95` mask but fails the fp64 `>= 0.95` → the consistency HALT false-fires
on a healthy run (#1336 R-delta Minor 1, verified with a 3-element numpy probe).

**Why:** the mask defines the SET; the tally is only a cross-check. Fix
direction matters for pinned quantities: conform the TALLY to the MASK (set and
any plan-pinned counts unperturbed), never the mask to the tally.

**How to apply:** on any diff touching a count that an equality/consistency
gate consumes, (1) grep for `.astype(np.float64)` (or implicit float() casts)
between the mask and the tally; (2) run the boundary probe yourself
(`np.float32(thr)` through both comparisons) rather than trusting the
implementer's claim; (3) confirm report-only diagnostics (histograms,
percentiles) may keep the fp64 cast — only gate-consumed counts must conform;
(4) check the conformed tally has no OTHER consumer asserting fp64 semantics.
Residual to note (usually pre-existing, non-blocking): row-max read from
sims[j,i] vs pair mask on sims[i,j] can differ in the last ulp across BLAS gemm
blocks — a distinct jitter channel the dtype fix does not close.

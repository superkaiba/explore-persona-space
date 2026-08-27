---
name: selection-symmetric-band-review-recipe
description: How to certify a selection-symmetric max-matched null fix (shared-selection identity, assert non-vacuity via value-path divergence, shape-keyed band orientation probe, committed FPR test) — #2569 r2 leg6
metadata:
  type: feedback
---

When a round fixes a per-comparison-band-vs-greedy-max defect with a "per-draw
same-selection band", certify four things, not just the docstring:

1. **Shared-selection identity** — grep BOTH call sites of the selection
   function (observed read + inside the null-draw loop); `null_aggregation_
   matches_observed` holds by construction only if the SAME function object
   computes both.
2. **Divergence-assert non-vacuity** — the runtime assert must compare two
   INDEPENDENT value paths (e.g. unnormalized np `|dot|` max vs a
   reference-fn-normalized per-match recompute). Live probe: feed non-unit
   columns and confirm the two paths diverge >1e-3; an assert recomputing one
   path twice proves nothing.
3. **Band shape/orientation keying** — probe the band at (r_a, r_b) vs
   (r_b, r_a): the p95s DIFFER (0.314 vs 0.244 at d=48, 3/7 in #2569), so the
   cache key must include ordered ranks and the observed matrix orientation
   (rows = which arm) must match the null's `F[:r_a, :]` slice. Also probe
   n_draws=0 → NaN band → all comparisons False (no fabricated verdict).
4. **Committed empirical FPR test** — demand a test that pushes ~300 H0 pairs
   through the PRODUCTION selection and measures per-pair rate ≈ nominal for
   the symmetric band AND >nominal for the pre-fix per-comparison shape (the
   blocker-discrimination pair).

**Why:** #2569 r2 (fb52ed2804) fixed observed-max-vs-single-comparison-band
inflation (0.096 realized at k=8 vs 0.05 nominal); all four checks passed live.
**How to apply:** any `.claude/rules/selection-symmetric-nulls.md` option-1
implementation; residual multiplicity axes (pairs, cells) still need explicit
disclosure notes — check the top-level any() aggregations too, they often lack
the note the pair level has.

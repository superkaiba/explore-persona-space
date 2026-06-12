---
name: eta-estimator-display-stats
description: Review heuristics for infra ETA/progress estimators (stage-duration stats -> displayed countdown) — overdue-floored chip, sum-of-medians vs median-of-sum, ratio kill-criteria near zero
metadata:
  type: feedback
---

Three failure modes recur in stage-duration → displayed-ETA designs (#587 plan v1):

1. **Overdue-floored chip is the real Must-Fix, not the 95%-parked bar.** The
   standard formula `eta_q = max(q_current − elapsed, 0) + Σ later q` degenerates
   to a CONSTANT small band the moment `elapsed > p75_current` — exactly the
   long-stuck tasks users check, and ≥25% of stage traversals reach that regime by
   construction. Plans that enumerate honesty degradations for blocked/stale but
   not overdue have an internal-consistency gap; fix is one `overdue` flag + UI
   degrade.
2. **Sum-of-stage-medians ≪ median-of-total for right-skewed pipelines.** In #587's
   own table: Σ stage medians 6.7h vs realized total median 16.1h (2.4×); Σ p75
   17.5h vs realized 42.1h (2.4×). Much of the gap is labeled-by-design exclusions
   (blocked detours, re-plan loops, human wait), so it's a Concern not REVISE — but
   the only way to adjudicate "typical range" honesty is a cheap historical
   backtest (predicted band at each stage entry vs realized clean-forward
   remaining; report coverage). Recommend wiring coverage into the kill criterion.
3. **Ratio-based kill criteria (p75/p25 > X) blow up near zero.** Late-pipeline
   remaining p25 can be ~0.05h, so the ratio drops chips that are genuinely
   informative ("~0–2h"). Anchor the check at a named stage (e.g. running-entry,
   historical basis) and/or floor the denominator.

Also recurring: stage medians ≈ 0 (same-second status transitions) crash
`elapsed/median` and zero out span weights — epsilon-floor medians; and
`gpu_hours_total=0` machine tokens (every infra plan) make a /gpu_count refined
band collapse to [0,0] — skip refinement on zero.

**Why:** UX estimators get a lower verdict bar (APPROVE-biased, honesty-label
framing), so the only REVISE-grade items are misleading-number-with-no-label in a
COMMON case; everything labeled (band-not-point, stale chip, assumed-1gpu) stays a
Concern.

**How to apply:** any `kind: infra` plan whose deliverable is a duration/progress
estimate read by humans (dashboards, titles, notifications). Check the four:
overdue regime, quantile-sum calibration backtest, kill-criterion computability
near zero, zero-denominator guards.

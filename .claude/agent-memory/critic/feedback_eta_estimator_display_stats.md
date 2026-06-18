---
name: eta-estimator-display-stats
description: ETA/progress-estimator infra plans — overdue-floored chip is the real Must-Fix; Σ-of-stage-medians vs median-of-total 2.4× gap needs a coverage backtest; ratio kill-criteria blow up near zero (#587)
metadata:
  type: feedback
---

Recurring failure modes in stage-duration → displayed-ETA designs (#587 v1):

1. **The overdue-floored chip is the real Must-Fix, not the 95%-parked bar.** `eta = max(q_current − elapsed, 0) + Σ later q` degenerates to a CONSTANT small band the moment `elapsed > p75_current` — exactly the long-stuck tasks users check, reached by ≥25% of traversals by construction. Fix: an `overdue` flag + UI degrade.
2. **Σ of stage medians ≪ median of total for right-skewed pipelines** (#587's own table: 6.7h vs 16.1h realized; p75 17.5h vs 42.1h — 2.4×). Much is labeled-by-design exclusions, so Concern not REVISE — but the only honest adjudication is a cheap historical backtest (predicted band at each stage entry vs realized remaining; report coverage); recommend wiring coverage into the kill criterion.
3. **Ratio kill-criteria (p75/p25 > X) blow up near zero** (late-pipeline p25 ~0.05h drops genuinely informative "~0–2h" chips). Anchor at a named stage and/or floor the denominator. Also: stage medians ≈ 0 (same-second transitions) crash `elapsed/median` — epsilon-floor; `gpu_hours_total=0` tokens collapse a /gpu_count band to [0,0] — skip refinement on zero.

**Why:** UX estimators get a lower verdict bar (APPROVE-biased, honesty-label framing) — REVISE-grade only for a misleading number with no label in a COMMON case; everything labeled (band-not-point, stale chip, assumed-1gpu) stays a Concern.

**How to apply:** any `kind: infra` plan whose deliverable is a human-read duration/progress estimate: check the four — overdue regime, quantile-sum calibration backtest, kill-criterion computability near zero, zero-denominator guards.

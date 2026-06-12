---
name: Re-judge-all designs residual refusal survivorship
description: Survivor-proxy correction rounds (re-judge ALL rollouts) still exclude judge-refused rows, a survivorship channel correlated with the construct; plus the aggregate-level anchor decomposition pattern (#591 e5)
type: feedback
---

Two patterns from #591 e5 (EM panel survivor-rate proxy fix, ~78k Sonnet re-judge):

1. **Refusal-exclusion is a residual survivorship channel in "all-rollouts" DVs.**
   A corrected DV defined as "mean over ALL judged rollouts" still drops
   judge-API refusals (`n_refused`) — and refusals concentrate on the most
   harmful completions (e1 measured: 8.3% empty-response in the villain self
   cell, the most-misaligned corpus, vs 1.7% elsewhere). So the corrected rate
   is biased DOWN exactly in high-EM cells. Bias bound: f·(1−r) ≈ 0.02 at
   f=0.083, r=0.76 — sub-τ but borderline cells can flip. Recoverable
   (Concern, not REVISE) iff per-cell `n_refused` is persisted: prescribe a
   worst-case imputation band (refused rows scored 0 vs 1) in analyzer notes.
   Both trained and base sides carry it, partially cancelling in the delta.

2. **Aggregate-level parity anchors decompose drift vs denominator correctly.**
   When the frozen parent persists aggregates only (no per-rollout verdicts),
   the right anchor design is: judge anchor cells in full, recompute the
   parent's EXACT conditional DV from fresh verdicts, compare to frozen.
   Fresh-conditional − frozen = judge drift; fresh-unconditional −
   fresh-conditional = denominator effect, from the SAME verdicts. Note the
   full-panel conditional recompute (sensitivity arm) gives a panel-wide
   drift read for free — the 5-cell anchor gate is just an early spend gate,
   so gate-edge ambiguities (e.g. exactly one cell >0.10 unspecified) are
   recoverable, not Must-Fix. Anchor SE on ~65-290-survivor conditional
   denominators ≈ 0.03-0.04; fresh coherence assignments RESAMPLE the
   survivor set, so add a survivor-count check (±25%) alongside the rate band.

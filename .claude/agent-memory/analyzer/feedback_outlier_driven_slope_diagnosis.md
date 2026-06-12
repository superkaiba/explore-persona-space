---
name: outlier-driven-slope-diagnosis
description: When N−1 of N held-out points cluster in a narrow predictor band and one sits far out, a (factor × continuous) interaction p-value is 1-point leverage; mandatory drop-anchor refit before headlining
metadata:
  type: feedback
---

When regressing a DV on (factor × continuous predictor) and the predictor's range is dominated by ONE held-out point (e.g. 7 of 8 personas at min_dist < 0.10, one at 0.21+), the interaction p-value is essentially asking whether that one anchor sits where the lines predict. It looks significant in the full panel and collapses in any drop-anchor refit — the per-level slopes fan out only because they share the anchor.

**Why:** task #405 — K×min_dist β = −0.81 (p = 0.011) full-panel collapsed to p = 0.51 with comedian dropped; Cook's d + DFBETAS top-4 were all comedian. The plan required significance in BOTH refits; not met → slope claim LOW (the K main effect stayed MODERATE via the dose-control).

**How to apply:**
1. Compute Cook's distance + |dfbetas| on the interaction coefficient; if the top-k offenders are one held-out unit, downgrade.
2. Report per-factor-level slopes with 95% CIs, full panel vs leave-one-out — ~10× SE inflation on drop-one is the visual tell.
3. Never headline the full-panel p; frame as "the slope-interaction is not supported by this design; only one anchor extends the predictor range."
4. The factor MAIN effect / mean-shift can still be robust — it doesn't depend on the predictor range.

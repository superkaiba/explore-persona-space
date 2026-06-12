---
name: outlier-driven-slope-diagnosis
description: For any K×min_dist (or analogous interaction) slope test, if 7/8 (or N-1/N) held-out points cluster in a narrow band of the predictor and 1 sits far, the interaction p-value across the full panel is doing 1-point leverage. Mandatory test before headline-claiming the interaction is real.
metadata:
  type: feedback
---

When regressing a DV on (factor × continuous_predictor) and the continuous predictor's range is dominated by ONE held-out point (e.g. 7 of 8 personas at min_dist < 0.10, one comedian at 0.21+), the slope-interaction term's p-value is essentially asking "does this one anchor sit where the lines predict it should." It will look significant in the full panel and fall apart in any sensitivity refit that drops the anchor.

**Why:** The fitted slopes for each level of the factor are anchored by the tight low-X cloud + the same far anchor. As the factor changes the cloud's height, the slopes fan out — but only because they share the anchor. Cook's distance + DFBETAS on the interaction term will show the anchor's observations dominating the top-k leverage list.

**How to apply:**
1. Before reporting a (factor × continuous) interaction as significant, compute Cook's distance and `|dfbetas|` on the interaction coefficient. If the top-k offending rows are all the same held-out unit, downgrade.
2. Per-K (or per-factor-level) slopes with 95% CIs on full panel vs leave-out — a 10× SE inflation on drop-one is the visual tell.
3. Frame the claim around what's left: "the slope-interaction is not supported by this design; only one anchor extends the predictor range." Do NOT report the full-panel p-value as the headline.
4. The K *main effect* / mean-shift can still be robust — it doesn't depend on the predictor range.

**Incident:** Task #405. Headline regression's K×min_dist β = −0.81 (p = 0.011) in full panel collapsed to β = −2.14 (p = 0.51) when comedian dropped. Cook's d + DFBETAS top-4 all comedian. Plan's success criterion explicitly required significance in BOTH refits — was not met. Correctly framed in the clean-result as LOW confidence on the slope hypothesis (the K main effect itself stayed MODERATE evidence via the dose-control).

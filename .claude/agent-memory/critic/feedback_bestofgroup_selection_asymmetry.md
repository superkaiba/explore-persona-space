---
name: Best-of-group selection asymmetry in predictor races
description: "Predictor-race headlines comparing best-of-Group-X vs best-of-Group-Y τ/ρ are biased when group sizes differ by 10-30× — winner's-curse inflation grows ~sqrt(2 ln K)·SE, so the big group wins under the null"
type: feedback
---

When a plan's headline is "best predictor in Group X beats best in Group Y by margin m" (e.g. #545 H2: best-of-B/C weighted Kendall τ > best-of-A by ≥0.15 under CV), check the GROUP SIZES. A geometry grid (extraction points × 8 layers × 9 metrics × 2 flavors) yields 100-300 Group-A variants vs ~10 for hand-built groups. Max-of-K selection on the SAME CV scores used for the decision inflates the larger group's champion by ≈ sqrt(2 ln K)·SE(τ): at SE≈0.06-0.10 and K=150 vs K=10, differential inflation ≈ 0.06-0.10 — comparable to the registered margin. Under a pure-noise null the big group "matches" the small group, manufacturing the surprise conclusion ("context geometry transfers after all").

**Why:** #545 plan v1 (2026-06-10): H2 criterion stated on raw leave-family-out CV scores with no nested selection and no group-size disclosure; Group A grid ~100-300 variants vs handfuls in B/C. Same family as the older best-of-K Type-I memory (feedback_spearman_threshold_n12.md) but at the group-champion level.

**How to apply:** Prescribe (any of): (a) nested selection — champion chosen on training folds only, scored on held-out fold; (b) confirmatory margin read on a quarantine split with CV-frozen champions; (c) max-stat permutation null per group so leaderboard inflation is visible; (d) always report K per group next to best-of-group scores. Usually analyzer-recoverable if per-predictor per-cell predictions are persisted → concern-for-analyzer with concrete prescription, not REVISE; escalate only if predictions are NOT stored or the quarantine structure is absent.

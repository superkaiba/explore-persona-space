---
name: Crossed-panel axis-power asymmetry
description: Run-axis n drops (80→16) while the persona axis stays — success-on-widest-axis + falsify-on-ALL-axes is conservative-OK; tiny strata make cluster CIs degenerate (#560)
type: feedback
---

Cross-recipe transfer designs reusing a crossed run × persona panel where one cluster axis shrinks hard (80 runs → 16 single-seed adapters) while the other stays (35 personas) — #560, APPROVE with concerns:

**Rule:** NOT a REVISE by itself when (a) per-axis CIs are all reported, (b) success = widest-axis CI excluding 0 (conservative), (c) falsification = ALL axes spanning zero (the well-powered axis must agree), and (d) the plan names the middle zone (persona-axis excludes 0, run axis spans) as not-success/not-falsified. The asymmetry biases toward "indeterminate", never false falsification. The middle zone is large by construction (run-cluster CI ~2.2× wider than parent) — demanding pre-registered resolution there would be a fake-precision gate. Most informative recipe-specificity signature: the well-powered axis's CI excluding the PARENT POINT ESTIMATE while containing 0.

**How to apply:** (1) check falsification includes the well-powered axis — if it can fire off the weak axis alone, THAT is a Must-Fix; (2) check tiny strata: a contrast with 3 persona clusters on one side has a degenerate cluster bootstrap (~10 distinct resamples) — that axis's CI is not a real uncertainty estimate; the run axis (15–16 clusters) is the meaningful one there. Analyzer concern when both axes are reported.

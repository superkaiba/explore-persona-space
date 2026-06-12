---
name: Crossed-panel axis-power asymmetry
description: Run-axis n drops (e.g. 80->16) while persona axis stays; success-on-widest-axis + falsify-on-all-axes creates a large non-verdict middle zone; tiny strata (3 persona clusters) make cluster CIs degenerate
type: feedback
---

Pattern (from #560 statistics review, 2026-06-10; APPROVE with concerns):
cross-recipe transfer designs reuse a crossed run x persona panel where one
cluster axis shrinks hard (80 runs -> 16 single-seed adapters) and the other
stays (35 personas).

**Rule:** This is NOT a REVISE by itself when (a) per-axis CIs are all
reported, (b) success = widest-axis CI excluding 0 (conservative), (c)
falsification = ALL axes spanning zero (requires the well-powered axis to
agree), and (d) the plan names the middle zone (persona-axis excludes 0, run
axis spans) as not-success/not-falsified. The asymmetry biases toward
"indeterminate", never toward false falsification — conservative, recoverable
by the analyzer.

**Why:** The middle zone is large by construction (run-cluster CI ~ sqrt(80/16)
≈ 2.2x wider than parent), so demanding a pre-registered resolution there would
be a fake-precision gate; the analyzer places the verdict from the reported
per-axis widths. Most informative recipe-specificity signature: the
well-powered axis's CI excluding the PARENT POINT ESTIMATE while containing 0.

**How to apply:** (1) Check the falsification criterion includes the
well-powered axis (all-axes requirement) — if falsification can fire off the
weak axis alone, THAT is a Must-Fix. (2) Check tiny strata: an exposure
contrast whose one side has 3 persona clusters has a degenerate persona-cluster
bootstrap (~10 distinct resamples) — the CI on that axis is not a real
uncertainty estimate; the run axis (15-16 clusters) is the meaningful clustered
CI there. Flag as analyzer concern, not REVISE, when both axes are reported.

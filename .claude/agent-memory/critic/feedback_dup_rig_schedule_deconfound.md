---
name: Duplication-rig ratio/count/schedule deconfounds
description: In round-robin duplicated-row rigs, "bigger build, 1 epoch" and "small build, k epochs" are near-identical data streams; the decisive contrast is small-build natural-T vs small-build stretched-T. Also: sparse parent trajectories can already pre-strain a count-accrual hypothesis.
type: feedback
---

When a rig builds "more rows" by duplicating a tiny distinct-row pool (e.g. #472: every budget cell = the same ~10 positive + ~40 negative distinct rows round-robined to fill the budget), a schedule-deconfound design has two structural facts (seen in #601 plan v1, APPROVEd):

1. **Double-size@1-epoch and quarter-size@4-epochs are near-replicates**, not two independent tests — both are ~the same row-visit stream over the same distinct rows, differing only in epoch-boundary shuffles and warmup arithmetic. The collinearity-breaking contrast is *small build natural-T vs same small build stretched-T* (same data, same ratio, 4x horizon). Treat the size-matched pair as an internal consistency check, and don't bounce a plan for "redundant arm" — it's cheap and validates the replicate.

2. **Check whether the parent's sparse trajectory already strains one hypothesis.** A cumulative-update-count (accrual) hypothesis predicts climbing trajectories; if the parent's FIRST checkpoint (~8-14% of updates consumed) already sits at terminal level (#472's 8:1 cells: 19.7-21.1 at step 10 of 113, flat thereafter), the accrual hypothesis enters pre-weakened. Not a REVISE when the new design's dense early grid (1-step checkpoints) directly resolves whether accrual completed before the parent's first read — but the analyzer should weigh the prior strike.

**Why:** Avoids two false critiques (redundant-arm, untestable-hypothesis) and one missed read (pre-strained hypothesis bookkeeping) on duplication-rig dose-response follow-ups.

**How to apply:** Any plan deconfounding ratio vs count vs schedule where the corpus is built by duplication: verify the distinct-row pool size first, identify which arm pair is the true single-variable contrast, and read the parent trajectory's first-checkpoint position against any accrual-shaped hypothesis.

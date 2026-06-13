---
name: Duplication-rig ratio/count/schedule deconfounds
description: In round-robin duplicated-row rigs, big-build@1ep and small-build@kep are near-replicates; the decisive contrast is same-build stretched-T; sparse parent first-checkpoint reads can pre-strain an accrual hypothesis (#601 v1)
type: feedback
---

When a rig builds "more rows" by duplicating a tiny distinct-row pool (#472: every budget cell = the same ~10 positive + ~40 negative distinct rows round-robined), schedule-deconfound designs have two structural facts (#601 v1, APPROVEd):

1. **Double-size@1-epoch and quarter-size@4-epochs are near-replicates**, not two independent tests — ~the same row-visit stream over the same distinct rows, differing only in epoch-boundary shuffles and warmup arithmetic. The collinearity-breaking contrast is *small build natural-T vs same small build stretched-T* (same data, same ratio, 4× horizon). Treat the size-matched pair as an internal consistency check; don't bounce for "redundant arm" (it's cheap and validates the replicate).
2. **Check whether the parent's sparse trajectory already strains one hypothesis.** A cumulative-update-count (accrual) hypothesis predicts climbing trajectories; #472's first checkpoint (~8–14% of updates) already sat at terminal level (19.7–21.1 at step 10 of 113, flat after) — the accrual hypothesis enters pre-weakened. Not a REVISE when the new design's dense early grid directly resolves whether accrual completed before the parent's first read; the analyzer should weigh the prior strike.

**How to apply:** any plan deconfounding ratio vs count vs schedule where the corpus is built by duplication: verify the distinct-row pool size first, identify which arm pair is the true single-variable contrast, read the parent trajectory's first-checkpoint position against accrual-shaped hypotheses.

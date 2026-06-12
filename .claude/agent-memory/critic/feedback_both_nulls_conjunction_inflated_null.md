---
name: Both-nulls conjunction with a documented-inflated null
description: SVD/direction-constancy plans that pre-register "clear BOTH row-shuffle AND sign-flip p95" as binding PASS, when the parent already documented row-shuffle as inflated-by-design for near-parallel columns — the conjunction can convert a null-construction artifact into a false kill narrative
type: feedback
---

When a plan's PASS criterion is "statistic > p95 of BOTH nulls" and one of those nulls is documented (by the parent issue itself) as preserving the effect's common component — e.g. the row-shuffle null permutes within feature rows, so for near-parallel columns (EM-style arms) the null distribution CONTAINS the shared direction and sits just under the statistic (#521 seed42: 0.524 vs row-shuffle p95 0.494, margin 0.03) — the conjunction is fragile in a misleading direction: a row-shuffle-only miss reads as "effect absent / source-dominated" when it actually means "this null is broken for this matrix shape."

**Why:** On #551 (LOO controls for #521), the kill-criterion prose interpreted any LOO null failure as "spectrum is source-dominated," but for the EM arm a row-shuffle miss would be a null artifact, not source dominance. The sign-flip null was the parent's own documented "cleaner floor" for that arm.

**How to apply:** Not a REVISE when both nulls + p95/p99 + the parent's caveat are all reported (analyzer can re-weight) — flag as the top Concern: analyzer should treat the non-inflated null as primary for the near-parallel arm and narrate a conjunction miss accordingly. Would become Must-Fix only if the plan reported a single pooled pass/fail without the per-null numbers.

Related check in the same family: derived-matrix (LOO/subset) nulls must be recomputed on the subset matrix (right in #551), and cos-to-U1 sign conventions can flip when U1 is recomputed on the subset — verify sign-alignment before cross-matrix comparisons.

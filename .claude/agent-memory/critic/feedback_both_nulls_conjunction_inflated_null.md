---
name: Both-nulls conjunction with a documented-inflated null
description: Clear-BOTH-nulls PASS criteria are fragile when one null is parent-documented as inflated for near-parallel columns (row-shuffle on EM arms); Concern not REVISE if per-null numbers ship (#551)
type: feedback
---

When a PASS criterion is "statistic > p95 of BOTH nulls" and one null is documented BY THE PARENT as preserving the effect's common component — e.g. row-shuffle permutes within feature rows, so for near-parallel columns (EM-style arms) the null CONTAINS the shared direction and sits just under the statistic (#521 seed42: 0.524 vs row-shuffle p95 0.494) — the conjunction is fragile in a misleading direction: a row-shuffle-only miss reads as "effect absent / source-dominated" when it actually means "this null is broken for this matrix shape".

**Why (#551, LOO controls for #521):** the kill-criterion prose interpreted any LOO null failure as "spectrum is source-dominated", but for the EM arm a row-shuffle miss would be a null artifact; the sign-flip null was the parent's own documented cleaner floor for that arm.

**How to apply:** Not a REVISE when both nulls + p95/p99 + the parent's caveat are reported — flag as top Concern: treat the non-inflated null as primary for the near-parallel arm and narrate a conjunction miss accordingly. Must-Fix only if the plan reports a single pooled pass/fail without per-null numbers. Same family: derived-matrix (LOO/subset) nulls must be recomputed on the subset matrix, and cos-to-U1 sign conventions can flip when U1 is recomputed on the subset — verify sign-alignment before cross-matrix comparisons.

---
name: Takeaway "CI excludes positive/negative" must match the bootstrap CI signs in the cited JSON
description: A Takeaway asserting a CI excludes positive can contradict the body's own Results prose AND the bootstrap JSON — point Codex at the exact ci_lo/ci_hi cells
type: feedback
---

When a clean-result body summarizes a multi-behavior partial-correlation /
bootstrap result in `## Takeaways` with a phrase like "CI excludes positive
for all three" (or "CI excludes the negative region"), that aggregate phrasing
is a recurring overclaim site: it can be TRUE for some cells and FALSE for
others, and the body's own per-result prose sometimes states the accurate
(per-cell) version while the Takeaway over-aggregates.

**Why:** the Takeaway is written as a one-line synthesis and "for all three" /
"all exceed" framings get applied before the author re-checks each CI. The
bootstrap CI is the ground truth — read `ci_lo` / `ci_hi` (or `ci_low`/`ci_high`)
for EVERY behavior/cell and check the SIGN of the relevant bound, not the point
estimate. A point estimate of -0.01 with a CI of [-0.26, +0.10] does NOT exclude
positive; an aggregate "excludes positive for all three" is then false for that
cell.

**How to apply:** in the composed Codex prompt, name the exact JSON path +
sub-keys (`by_behavior_layer.<beh>.<layer>.partial_clustered_bootstrap.ci_hi`)
and instruct Codex to read the SIGN of the bound per behavior, then cross-check
the Takeaway's aggregate claim AND the body's per-result prose — flag the
internal contradiction with verbatim numbers. This is also a natural mechanical-
verifier candidate (a check that a Takeaway "CI excludes X" claim matches the
cited JSON's bootstrap CI signs); surface it as plain English in the verdict
body for the orchestrator, never as a workflow-fix candidate from the Codex twin.

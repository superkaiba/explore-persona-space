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

**Mirror pattern — "wins are confined to X" / "only X clears both comparators"
scope claims (#841 r3, 2026-07-02).** The same over-aggregation bites the
POSITIVE-win direction: a caption/Takeaway that scopes a result ("system-mode
both-comparator wins are confined to sycophancy sources 6–8", "only the affine
map beats the source read") can silently OMIT a small-but-CI-separated cell in
another trait/layer. Don't let the Codex prompt accept the confinement at face
value — make the check an EXHAUSTIVE all-cell scan: for the relevant class,
scan EVERY cell across ALL groups (traits/layers/modes) for the both-clear
predicate (both `vs_*.excludes_zero==true` AND positive delta), then check the
body's "confined to"/"only" claim against the FULL win-set. Two extra beats
beyond the null-direction check above: (1) scan positive-win cells, not just
the sign of a CI bound; (2) when the scan surfaces an out-of-scope win with a
TINY delta (e.g. #841's hallucination src16 system, +0.04/+0.022 both
excl0=true, vs the featured +0.24/+0.46 sycophancy wins), have Codex ADJUDICATE
MATERIALITY with the delta values (a real omission the scope claim should
mention vs a negligible boundary case), not just flag existence — a +0.04
CI-separated win is a technical win but may be immaterial to a "wins" caption.
Compose the check neutrally ("evaluate, do NOT pre-decide"); as the
compose-only wrapper you may run the scan yourself to AIM the check + flag the
candidate cell to the orchestrator for the reconciler, but never issue the
verdict.

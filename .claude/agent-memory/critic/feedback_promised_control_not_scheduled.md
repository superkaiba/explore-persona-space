---
name: Promised-but-unscheduled control / vacuous-by-construction control
description: §5/§6 list a control the committed code never implements (or one algebraically guaranteed to match the main result); Concern not REVISE when the headline doesn't rest on it
type: feedback
---

A plan's Controls table (§5) or exploratory-plots list (§6) can name a
control the analysis script does not implement AND the plan never
schedules as in-scope work (§4 cleanup / §10 tests / §6.5 deliverables
all silent). Grep the actual script before trusting the table.

**Disposition (The Bar):** REVISE only if the headline cannot be made
without that control. If the headline rests on a deterministic / exact
estimator AND a structural unit test already guards the failure mode the
control would catch, the missing control is CONFIRMATORY → Concern, not
REVISE. Tell the analyzer to either build it or DROP the §5/§6 row so the
folded-in result doesn't claim a control that wasn't run.

**Why:** A control whose only job is to confirm a negative the estimator
already rules out (e.g. label-shuffle ridge when PRESS LOO is
machine-precision-exact and a "random input → R²≈0" unit test exists) is
not load-bearing for the headline. Promising it and not running it is a
plan-vs-code inconsistency, not a conclusion-changing flaw.

**Second pattern — vacuous-by-construction control.** A "z-scored INPUT"
robustness control on a ridge/linear map is ALGEBRAICALLY IDENTICAL to
the unscored result: a global per-dim affine rescale of X is fully
absorbed by ridge's own per-fold standardization, so z_ridge == ridge to
machine precision by construction. It tests nothing. Flag so the analyzer
doesn't cite it as input-dim-robustness evidence. (The nonlinear/MLP
z-variant CAN differ — only the linear one is vacuous.)

**How to apply:** Statistics & Measurement lens, item 4 (Controls). When
a plan's control table lists items, grep the cited script for them;
if absent + unscheduled, ask "does the headline rest on this?" — if no,
Concern + "build-or-drop". If a control is an affine-invariant of the
main estimator, it's vacuous → Concern.

Origin: #722 v5 (base map c_C→v0 skill-over-mean) — an earlier
review-stage concern that the implementer subsequently addressed in
commit 82e8400609: the §5 label-shuffled ridge control
(`SHUFFLE_RIDGE_LAYER = 18`) and the per-layer MLP-shuffle null
(`skill_shuffle_mlp`) ARE now implemented in
`scripts/issue722_skill_over_mean.py`. The z-scored-input RIDGE control
remains correctly DROPPED as algebraically vacuous (affine-invariant
under ridge per-fold standardization) — keep that vacuity lesson intact.
APPROVEd — headline was the deterministic ridge plateau, guarded by exact
PRESS + a zero-skill unit test.

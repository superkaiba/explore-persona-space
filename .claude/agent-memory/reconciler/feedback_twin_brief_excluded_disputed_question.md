---
name: twin-brief-excluded-disputed-question
description: A twin's PASS/APPROVE carries zero rebuttal weight on a question its composed brief excluded — treat as non-coverage, not disagreement; also check plan vs post-spawn BINDING body addenda
metadata:
  type: feedback
---

When one reviewer's composed brief marked the disputed question out-of-bounds
(e.g. "deferred arms are out of scope for this review"), its PASS-class verdict
is structural NON-COVERAGE of that question, not a verified-clean read that
contests the other side's finding. Adjudicate the finding on the artifact alone;
the blind twin's verdict neither corroborates nor rebuts.

**Why:** #2546 round 1 (alternatives lens): a BINDING user scope addendum landed
in `body.md` + an `epm:progress` marker ONE MINUTE after the planner spawn; plan
v3 still deferred two user-required arms, cited a nonexistent "body round-1
recommendation" as license, listed promoting the arms under §13 must-ask, and
under-sampled the required t-grid (3 interior points vs {0.1..0.9}). The Codex
twin APPROVEd — its own NOTE disclosed the brief excluded deferred arms. Grep
facts: plan v3 had ZERO hits for `addendum|BINDING`.

**How to apply:** (1) When verdicts split and the PASS side's brief (or its own
NOTE) shows the disputed question was excluded, weight it as non-coverage.
(2) On any plan reviewed after a task-body edit, grep the plan for the addendum
heading / "BINDING" and diff plan scope vs the body's CURRENT scope sections —
a "per the body's recommendation" clause is checked against the body as it
stands, not as the planner saw it. Sibling of
[[amendment-round-stale-literal-sweep]] (numeric literals) — this is the
scope/arms twin. (3) Distinct from
[[cross-lens-defect-refiled-under-every-critic-lens]]: a deferred arm that IS
the lens's discriminating control (necessity toggle vs borrowed-label
alternative) is IN-lens, so the out-of-lens-scope discard does not apply.

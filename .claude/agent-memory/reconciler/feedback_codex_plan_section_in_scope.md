---
name: Codex FAILs round-N for absent plan-section wiring outside round brief
description: Codex code-reviewer FAILs a round-N "targeted fix" when an OTHER plan section's wiring is absent, even if Claude flagged the gap explicitly as out-of-scope; verify round-brief scope + invocation reachability before believing FAIL
type: feedback
---

When the round-N brief is a targeted fix (e.g. round-6 = "LoRA rank
threading + eval cap bump"), Codex code-reviewer FAILs when a DIFFERENT
plan section (e.g. plan §5.5 smoke-gate fallback retry) is not yet
implemented in dispatcher code, citing the plan verbatim. Claude
PASSes because the round brief is COMPLETE on the in-scope items AND
explicitly flags the missing OTHER plan section in "Unaddressed Cases"
as known-deferred.

**Why:** Reviewers' binary verdict adjudicates whether the round-N fix
is correct, NOT whether the codebase satisfies every plan section.
A plan-compliance gap on an UN-invoked code path (smoke gate never
reached in production yet; the fallback retry is dead code today) is
a Real-but-non-blocking finding, not Real-blocking. Promote to a
binding concern via PASS+CONCERNS rather than bouncing to round 7.

**How to apply:** Before believing Codex's FAIL:
1. Read the round-N implementer brief (often pinned in the most recent
   `epm:implementer-brief` / `epm:status-changed` note). Verify the
   reported failing element is INSIDE the brief.
2. Trace reachability: is the absent plan-section actually invoked
   today, or is it dead code (e.g. smoke gate never reached in prior
   crashes)?
3. Check Claude's "Unaddressed Cases" / "Follow-up" section — if
   Claude already flagged the gap as known-deferred with the same
   evidence, this is SHARED territory and only the severity
   classification differs.
4. Consider operational safety of the deviation: is the current
   behavior a SAFER default than the plan literal (e.g. halt-and-
   surface vs auto-recipe-swap on bug-induced gate failure)? If so,
   the deferred concern should propose EITHER implementing the plan
   OR revising the plan to document the safer choice.

PASS-with-CONCERNS + standing recommendation to wire the fallback
(or revise the plan) is the natural shape. Do NOT bounce a complete
round-N fix because round-N+M scope is still open.

Origin: task #505 round-6 reconcile. Plan v1:128 specified auto-fallback
retry on smoke-gate FAIL; dispatcher returns code 2 instead. Round-6
brief was LoRA-rank threading; Codex FAILed on the §5.5 wiring gap,
Claude PASSed and flagged it as deferred. Smoke gate never reached in
production on this task — fallback path is dead code today. Adjudicated
CONCERNS (PASS-class) with the fallback wiring promoted to a binding
follow-up.

Companion to "Codex methodology-choice as code bug" — when the
implementer's choice (here: halt-on-gate-fail) IS a plan-listed
alternative or an operationally-safer default, plan-compliance becomes
a methodology question, not a code bug.

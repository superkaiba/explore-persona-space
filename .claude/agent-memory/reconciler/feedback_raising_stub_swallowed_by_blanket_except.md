---
name: Raising-stub "never called" tests are vacuous under a blanket exception guard
description: Plan-stage test-plan adjudication — when a plan specifies BOTH a blanket try/except Exception -> None probe wrapper AND a negative test whose only evidence is a raising stub inside that wrapper, the test passes under the exact bug it targets; Codex missed it, Claude caught it (#1051 r1)
type: feedback
---

Rule: when a plan (or diff) pairs a blanket `try/except Exception -> None`
guard around a probe/gate with a negative test whose ONLY evidence of
"the probe was never consulted" is a stub that RAISES if called, execute
the interaction yourself: the raise (incl. `AssertionError`, a subclass of
`Exception`) is swallowed by the guard, the fall-through verdict is exactly
what the test asserts, and the test is green under the precise mutation it
registers itself as ruling out. The discriminating forms are (a) a
return-True/positive stub whose bug path flips the asserted verdict, or
(b) a call-recording stub with an assert-empty-call-list.

**Why:** #1051 r1 (statistics lens, infra test plan): §4.2 specified
`issue_liveness_reason` "entirely wrapped in try/except Exception -> None";
§6 test 5 (`test_cleared_breadcrumb_not_probed`) stubbed
`pid_alive_with_identity` to raise AssertionError if called. Claude REVISE
(correct — verified against plan text; the sibling tests 13/14 already used
the discriminating return-True form, so the fix was one spec line); Codex
APPROVE and its S-A row affirmatively credited the vacuous test with
catching the clearing mutation. Production consequence was the plan's own
named risk R2 (wedged-leader HEALTHY latch on a finished phase).

**How to apply:** on any test-plan disagreement where one side flags a
"probe/branch never consulted" test, grep the plan/code for the exception
posture wrapping the stubbed call site before crediting the test. This is
Must-Fix caliber when the plan PRESCRIBES the stub shape (a plan-faithful
implementer ships the vacuous test and adherence review upholds it) — the
plan-verbatim-text family, fixable only at plan text. Sibling patterns:
feedback_live_replay_proposed_mechanical_checks.md (execute the proposed
check against the named offender), feedback_claude_underclasses_unverified_branch_test_gap.md.

---
name: Delta-scoped rounds beyond r3 — compose, don't hard-fail
description: Round cap is now 10 (workflow.yaml round_cap_per_reviewer; was 5 via #1017, was 3 before that) — accept rounds 1-10, refuse malformed (<=0, >10, non-integer); retains the delta-composition recipe for delta-scoped briefs (r4+ and follow-up re-gates).
type: feedback
---

When the orchestrator brief requests a `revision_round` beyond 3 as an
explicitly delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS vs
r3 Codex needs_targeted_fix -> reconciler binding REVISE -> r4 delta on the one
fix), COMPOSE the prompt rather than refusing. The same delta-scoping recipe
applies to a same-issue FOLLOW-UP re-gate round 1 whose brief carries a delta
scope note (#1739 `composition-grid-multiseed-plus-arm2-repair`, 2026-08-23:
round 1 on an already-critique-PASSed folded body — scope to the fold delta +
integration, pre-existing sections re-litigated only where the fold changes
their meaning, but still emit all fifteen lens lines).

**Why:** the per-reviewer round cap is 10 (workflow.yaml § ensemble_review
`round_cap_per_reviewer: 10`; the agent spec's rule 1 matches — the cap was 5
under #1017 and 3 before that; do not quote the stale 5). A reconciler-bound
REVISE naturally produces an r4+ re-verify; hard-failing a deliberate,
well-formed dispatch burns the round and forces single-Claude fallback exactly
when the cross-family re-check matters most (the Codex FAIL drove the REVISE).
Refusal is reserved for genuinely malformed rounds (<= 0, > 10, non-integer),
not deliberate delta rounds. Sibling precedent: codex-critic's
`feedback_delta_scoped_amendment_rounds.md`.

**How to apply:** for delta-scoped briefs, (1) state the assumption in the
return; (2) scope the composed prompt to the delta the brief names (verify THE
NAMED fixes/fold only; no re-litigating settled items; new findings only if
conclusion-relevant / spec-breaking; round-N quoting rule for applied/absent
claims; keep all fifteen lens lines — untouched lenses may carry a one-line
delta-scoped PASS); (3) keep the marker version = the round number
(`epm:clean-result-critique-codex v<n>`); (4) refuse only genuinely malformed
rounds (0, negative, >10, non-integer). Always re-check the current agent spec
+ workflow.yaml cap before quoting a number — this memory has gone stale on
the cap twice (3 -> 5 -> 10).

**Adjudicated-resolution clause (added #1739 r2, 2026-08-23):** when the brief
records that a prior blocker from THIS twin was resolved in a DIFFERENT form
than the twin asked for, because a project rule adjudicated against the asked
form (e.g. r1 asked for on-canvas "N/A" cells; the fix omitted the columns +
disclosed in the caption, since CLAUDE.md item 8-bis bans on-canvas scope
notes), the composed delta block must carry an explicit ADJUDICATION NOTE:
name the rule, mark the alternate form the BINDING resolution, instruct Codex
to verify only the adjudicated form and NOT re-raise the original ask.
Without it, the twin re-raises its own blocker verbatim and burns a
reconciler round on a settled call.

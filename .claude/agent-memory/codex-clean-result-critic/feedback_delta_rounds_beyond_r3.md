---
name: Delta-scoped rounds beyond r3 — compose, don't hard-fail
description: Round cap is now 10 (workflow.yaml round_cap_per_reviewer; spec accepts 1-10, malformed <=0/>10/non-integer refused); this memory retains the delta-composition recipe for r4+ delta-scoped briefs.
type: feedback
---

**Cap update (2026-08-24, #2388 r4):** the spec + workflow.yaml now accept
rounds 1-10 (`round_cap_per_reviewer: 10`) — the "1-5 as of #1017" claim
below is historical. Refusal is for <=0, >10, or non-integer only. The
delta-composition recipe below is unchanged.

When the orchestrator brief requests `revision_round` 4 or 5 as an explicitly
delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS vs r3 Codex
needs_targeted_fix -> reconciler binding REVISE -> r4 delta on the one fix),
COMPOSE the prompt rather than refusing.

**Why:** the ensemble policy caps the four iterating sites at 5 rounds, and a
reconciler-bound REVISE naturally produces an r4 re-verify; the agent spec's
rule 1 accepts rounds 1-5 (fixed by #1017; it historically bounded rounds to
1-3). Hard-failing a deliberate, well-formed dispatch burns the round and
forces single-Claude fallback exactly when the cross-family re-check matters
most (the Codex FAIL drove the REVISE). Refusal is reserved for genuinely
malformed rounds (<= 0, > 5, non-integer), not deliberate delta rounds.
Sibling precedent: codex-critic's `feedback_delta_scoped_amendment_rounds.md`.

**How to apply:** for r4/r5 briefs, (1) state the assumption in the return;
(2) scope the composed prompt to the delta the brief names (verify THIS fix
only; no re-litigating settled items; new findings only if conclusion-relevant
/ spec-breaking; round-N quoting rule for applied/absent claims); (3) keep the
marker version = the round number (`epm:clean-result-critique-codex v4`);
(4) the spec accepts rounds 1-5 as of #1017 — no spec-vs-cap mismatch to flag.
Still refuse genuinely malformed rounds (0, negative, >5, non-integer).

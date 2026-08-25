---
name: delta-scoped-rounds-and-round-cap
description: Round cap is 10 (workflow.yaml round_cap_per_reviewer; was 5 under #1017) — malformed = <=0, >10, non-integer; retains the delta-composition recipe for reconciler-bound delta-scoped briefs (r4+).
metadata:
  type: feedback
---

When the orchestrator brief requests a `revision_round` above 3 as an
explicitly delta-scoped re-review (e.g. #952 r4, 2026-07-04: r3 Claude PASS
vs r3 Codex needs_targeted_fix -> reconciler binding REVISE -> r4 delta on
the one fix), COMPOSE the prompt rather than refusing.

**Why:** the ensemble policy caps the four iterating sites at 10 rounds per
reviewer (workflow.yaml § ensemble_review `round_cap_per_reviewer: 10`;
verified against the live agent spec 2026-08-25 — the earlier #1017 cap of 5
is STALE), and a reconciler-bound REVISE naturally produces an r4+ re-verify.
Hard-failing a deliberate, well-formed dispatch burns the round and forces
single-Claude fallback exactly when the cross-family re-check matters most.
Refusal is reserved for genuinely malformed rounds (<= 0, > 10, non-integer).
Sibling precedent: codex-critic's `feedback_delta_scoped_amendment_rounds.md`.

**How to apply:** for delta-scoped briefs (typically r4+), (1) state the
assumption in the return; (2) scope the composed prompt to the delta the
brief names (verify THIS fix only; no re-litigating settled items; new
findings only if conclusion-relevant / spec-breaking; round-N quoting rule
for applied/absent claims); (3) keep the marker version = the round number
(`epm:clean-result-critique-codex v<n>`). A brief with no delta scope note
at r4+ gets the normal full-prior-history re-review. Still refuse genuinely
malformed rounds (0, negative, >10, non-integer). See also
[[compose-recipe-lens-ref-replacements]].

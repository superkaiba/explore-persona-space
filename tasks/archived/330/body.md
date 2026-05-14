---
title: 'Workflow: how to keep an issue alive as a todo when it doesn''t become a useful
  clean result'
kind: infra
tags: []
created_at: '2026-05-09T06:10:27.000Z'
has_clean_result: false
sagan_id: 401122df-ad28-42cb-bc5e-2146a7d469c1
sagan_number: 330
priority: normal
---
## Goal

Decide the canonical workflow path for issues that reach `status:awaiting-promotion` (or earlier) where:

- The experiment ran or partially ran,
- The result isn't load-bearing enough to promote as `clean-results:useful`,
- BUT we don't want to abandon the question — there's a clear next step we want tracked.

The current options are not crisp:

1. Promote as `clean-results:not-useful` and create a separate `status:proposed` follow-up issue. Works, but conflates "explored framing isn't useful for the paper" with "experiment didn't yield interpretable results — we'd like to retry differently."
2. Close the issue (auto-archives) and reopen / file a new one later. Loses lineage and the partial work.
3. Set `status:blocked`. Wrong semantics — nothing is blocking it, we just don't have bandwidth or it needs a redesign.
4. Leave it parked at `awaiting-promotion`. Pollutes the column the user uses to triage promotion decisions.

## Question

What is the canonical path? Candidates:

- **(a) Use `not-useful` as the todo bucket.** Document that `not-useful` doesn't mean "abandoned" — the live next-step is a fresh `status:proposed` follow-up that links back. Cheapest, no new mechanism.
- **(b) New label `status:parked-with-followup` or a new project column.** Explicitly distinguishes "experiment didn't produce a clean result, but we want to come back" from "promoted as not-useful." More mechanism, clearer semantics.
- **(c) A todo/backlog label** that lives orthogonally to the status state machine. The issue keeps its current status; the label flags "user wants to revisit."

## Trigger / context

Surfaced while promoting #284 (Gaperon evolutionary trigger round-0 diagnostic) — a clean MODERATE-confidence negative result that earns `useful`, but the conversation raised the question for the broader case of issues where clean-result promotion isn't the right exit.

## Acceptance

- Decision recorded in CLAUDE.md (or `.claude/skills/promote-clean-result/SKILL.md` if scoped to the promotion gate).
- If a new label / column is added, `scripts/gh_project.py` updated and the project board structure documented.
- The `/promote-clean-result` skill's "useful vs not-useful" decision tree updated to reference the new path.

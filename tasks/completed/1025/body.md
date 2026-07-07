---
title: 'daily-fix: codex-critic leak guard strips cross-issue #N refs'
kind: infra
tags:
- wf-fix
- wf-fix-fp:00cd006b115a
- daily-auto-filed
created_at: '2026-07-04T22:11:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-01 backfill route-2: codex-critic composer numeric-leak
  verification stripped the #720 task reference during the #795 critique, withholding
  redundancy evidence from Codex-Statistics; whitelist task-reference tokens'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-01 backfill problem sweep (route 2) from a prose follow-up surfaced in the #795 critique session (c8b5f744, 07:26Z).

## Goal

Let cross-issue task references (`#<N>` tokens) survive the codex-critic composer's numeric-leak verification, so Codex critics can weigh duplication/overlap evidence against prior tasks.

## Workflow gap

- **Bug observed:** during #795's plan critique (2026-07-01), the Codex composer's numeric-leak guard stripped the `#720` task reference from the composed prompt, so Codex-Statistics "could not weigh the #720 overlap sharply" — the very redundancy evidence that ultimately made #795 a Route A no-change deflection was withheld from one of the critics that had to find it.
- **Why it is a workflow gap:** `.claude/agents/codex-critic.md`'s composer numeric-grounding rule treats ANY numeric atom as a potential leak; a `#<taskid>` reference is provenance, not a predicted/authored result number, and stripping it degrades the redundancy/overlap lens the critics are mandated to apply.
- **Confidence (emitter):** medium.

## Proposed change (refine in planning)

In `.claude/agents/codex-critic.md`'s numeric-leak verification (and any shared verifier snippet it prescribes), whitelist task-reference tokens (`#\d+` when prefixed with `#`, plus `tasks/<status>/<N>` path forms and `issue-<N>` branch forms) from the numeric-atom accounting, provided they trace to `plan_body` / `lens_items` / prior-critique text OR the task registry. Keep every other numeric atom rule unchanged.

## Scope / surfaces

- Primary target: `.claude/agents/codex-critic.md`
- Grep the sibling composers for the same guard: `grep -rln 'numeric' .claude/agents/codex-*.md` — apply uniformly if the same accounting is duplicated.

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` default run passes; the #722-closing numeric-grounding rule's intent (no composer-authored result numbers) is preserved.

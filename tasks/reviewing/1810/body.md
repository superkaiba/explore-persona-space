---
title: 'daily-fix: pre-split multi-deliverable implementer dispatche'
kind: infra
tags:
- wf-fix
- wf-fix-fp:35bc563046b9
- daily-auto-filed
created_at: '2026-07-29T07:15:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1775''s implementer died
  at the context ceiling (''Prompt is too long'') after 139 tool calls / ~63 min on
  a 7-script build; recovery was a micro-scoped respawn — the same split applied AFTER
  the death that could have been applied at dispatch'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-H P4.

## Goal

Pre-split known multi-deliverable implementer builds at dispatch instead of recovering with the same split after a context-ceiling death.

## Workflow gap

- **Bug observed:** #1775's implementer was dispatched with a 7-script deliverable set; it died at the context ceiling after 139 tool calls / ~63 min (bulk of the work had landed durably; ~15 min triage + respawn overhead). The recovery — sequential micro-scoped rounds — is exactly the split available at dispatch time from the plan's own deliverable count.
- **Why it is a workflow gap:** the micro-scoped split exists only as a POST-death respawn recipe (SKILL.md ~2845); nothing keys it proactively off the plan's declared deliverable count.
- **Confidence (emitter):** medium (inferred; the death + recovery are transcript-verified)
- verified-at-filing: `grep -n 'micro-scop' .claude/skills/issue/SKILL.md` → respawn recipe at ~2845-2848 (recovery-side only) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

One dispatch-step clause: deliverable count > threshold ⇒ sequential micro-scoped rounds by default; cite the #1090/#1775 precedents.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (implementer dispatch step)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 35bc563046b9

- workflow_fix_target: .claude/skills/issue/SKILL.md


---
title: 'daily-fix: Step 10d tasks/* merge-conflict default + far-behind branch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:9e057d95ad05
- daily-auto-filed
created_at: '2026-07-01T06:53:21Z'
has_clean_result: false
origin_prompt: '/daily route-2 auto-file 2026-06-30: Step 10d worktree->main merge
  repeatedly conflicts on foreign concurrent-session `tasks/*` files (events.jsonl
  content + rename/delete conflicts from concurrent status moves) and re-imports stale
  task'
---
## Overview / Motivation

Auto-filed by the /daily three-route problem sweep (2026-06-30), route 2 (behavior/logic change → independent review pipeline).

## Goal

Make the scratch-worktree-detached-at-origin/main surgical merge the DEFAULT Step-10d path for workflow-fix/small-diff tasks when BEHIND exceeds a threshold (skip the doomed --rebase); add `.gitattributes merge=union` for `tasks/**/events.jsonl` (append-only) and `-X no-renames`/exclude `tasks/*` from the merge so concurrent task folders never conflict; exclude Step-5a-synced workflow-surface files from the Guard-3 out-of-scope check.

## Workflow gap

- **Bug observed:** Step 10d worktree->main merge repeatedly conflicts on foreign concurrent-session `tasks/*` files (events.jsonl content + rename/delete conflicts from concurrent status moves) and re-imports stale task-folder snapshots, forcing ad-hoc scratch-worktree-detached-at-origin/main + cherry-pick recovery reconstructed by hand every time (issues 739/748/749/752/753/755/762/764). Far-behind experiment branches (#697 3771 behind) additionally fail Guard-3 because Step-5a spec-freshness sync copied now-stale workflow-surface files, parking for a manual rebase.
- **Evidence:** Dominant friction — confirmed by /daily miners across batches 01/04/05/06/08/10.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface / code fix per the target; keep ruff + workflow_lint + the relevant tests green.
- The planner may deflect with a reasoned no-change report if the gap is already closed on main.

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- fingerprint: 9e057d95ad05
- source: /daily route-2 (2026-06-30)

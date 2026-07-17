---
title: 'daily-fix: pre-push lint reads ratchet from origin/main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b1ead0ea5359
- daily-auto-filed
created_at: '2026-07-17T06:57:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the pre-push lint gate
  trusts the long-lived worktree''s copy of scripts/workflow_lint.py, so ratchet-constant
  drift between main and the worktree fails the gate spuriously — #1366 burned ~10
  min applying main''s verbatim ratchet block to the worktree copy and re-running'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from transcript mining (#1366, ~00:55Z; RECURRENCE same day: #1411 ~23:09Z hit a stale byte-cap from the branch's overlaid lint copy predating main's #1409 cap raise).

## Goal

Remove spurious pre-push gate failures caused by worktree-vs-main lint-ratchet drift.

## Workflow gap

- **Bug observed:** the pre-push lint gate trusts the long-lived worktree's copy of scripts/workflow_lint.py, so ratchet-constant drift between main and the worktree fails the gate spuriously — #1366 burned ~10 min applying main's verbatim ratchet block to the worktree copy and re-running
- **Why it is a workflow gap:** Long-lived worktrees predictably drift behind main's ratchet bumps; a gate that trusts the stale copy fails on healthy pushes.
- **Confidence (emitter):** medium
- verified-at-filing: incident on #1366 transcript (ada8210a ~00:55Z: 'Applying main's verbatim ratchet block to the worktree's lint copy'); n/a for a grep — the gap is gate-recipe behavior in the SKILL's pre-push snippet, verified against the incident

## Proposed change (candidate diff sketch — refine in planning)

have the pre-push lint gate read ratchet constants from fetched origin/main's copy of workflow_lint.py (or auto-sync the ratchet block before the gate)

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: b1ead0ea5359


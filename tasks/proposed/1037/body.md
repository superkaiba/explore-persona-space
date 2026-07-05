---
title: 'daily-fix: issue-763 branch surgery — strip foreign files, re-merge'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-04T23:01:57Z'
has_clean_result: false
origin_prompt: /daily 2026-07-03 problem sweep — issue763-branch-surgery (fp da73b6886a9f)
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 (backfill run 2026-07-04) from the day's transcript problem sweep.

## Goal

In a scratch worktree detached at origin/main: strip the foreign paths from the issue-763 branch (rebase -i equivalent via filter or fresh cherry-pick of the task's own paths), re-run the Step 10d rebase-merge, confirm epm:merged and that body-linked figures resolve at main HEAD.

## Workflow gap

- **Bug observed:** #763 has a STANDING epm:merge-failed at Step 10d (guard-3 unsafe): the issue-763 worktree diff carries FOREIGN files (another task's tasks/ state + figures/issue_665), so its figures/eval-result commits are not on main while the task sits promotable at awaiting_promotion (2 cheap-band rounds complete).
- **Why it matters:** The task's clean-result links dangle until the merge lands; the guard correctly refused, the residual is the one-off cleanup.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `repo state: issue-763 worktree branch (ops, no workflow-surface change)`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py` default run passes; ruff on touched files passes.
- This task was auto-filed by the /daily three-route classifier (route 2 — behavior/logic change, independent review).

## Provenance

- workflow_fix_target: repo state: issue-763 worktree branch (ops, no workflow-surface change)
- fingerprint: da73b6886a9f
- source: /daily 2026-07-03 problem sweep (transcripts of 2026-07-03 UTC)

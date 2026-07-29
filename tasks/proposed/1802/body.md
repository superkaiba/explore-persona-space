---
title: 'daily-fix: new_worktree.sh sets upstream for issue branches'
kind: infra
tags:
- wf-fix
- wf-fix-fp:3b92e3a12b25
- daily-auto-filed
created_at: '2026-07-29T07:13:03Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): issue-<N> worktree branches
  get no upstream tracking, so a bare git pull --rebase inside a worktree fails with
  ''no tracking information'' — one live failure in #1689''s session on 2026-07-28'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-G P8 (miner-probed; re-verified).

## Goal

Give issue-branch worktrees upstream tracking so bare pulls work.

## Workflow gap

- **Bug observed:** In #1689's session a bare `git pull --rebase` inside the issue worktree failed: the issue-1689 branch has no upstream tracking configured. One wasted call, self-recovered with an explicit `origin issue-1689` — but the class recurs for every worktree consumer that composes a bare pull.
- **Why it is a workflow gap:** new_worktree.sh creates branches without ever calling --set-upstream/--track, so tracking depends on whether some later push happened to use -u.
- **Confidence (emitter):** high (probed)
- verified-at-filing: `grep -cn 'set-upstream\|--track' scripts/new_worktree.sh` → 0 (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

When the branch already exists on origin (the resume case), set upstream at worktree creation; else document/set it on first push. One-line change + a test if the script has a test file.

## Scope / surfaces

- Primary target: `scripts/new_worktree.sh`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 3b92e3a12b25

- workflow_fix_target: scripts/new_worktree.sh


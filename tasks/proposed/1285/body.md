---
title: 'daily-fix: remove stale duplicate tasks/reviewing/1196'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-12T06:52:51Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): tasks/reviewing/1196 coexists
  with registry-canonical tasks/completed/1196 on origin/main (the #644/#1253 duplicate-folder
  class), violating the one-folder-per-task invariant'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-11 from the problem sweep (surfaced by task #1274's session, which fixed the sweep's robustness to this duplicate but deliberately left the folder itself: "the stale `tasks/reviewing/1196` folder still exists on `main` — the #644/#1253 remediation applies"). NOT a workflow-surface gap — task-state hygiene on `main`.

## Goal

Remove the stale duplicate `tasks/reviewing/1196` folder from `main`, restoring the one-folder-per-task invariant (canonical: `tasks/completed/1196` per `tasks/REGISTRY.json`).

## Problem

- **Bug observed:** `tasks/reviewing/1196` coexists with the registry-canonical `tasks/completed/1196` on `origin/main` — the #644/#1253 duplicate-task-folder class (a squash/rebase merge re-imported the pre-move folder). #1274 (completed 2026-07-11) made the nightly routed-record sweep robust to duplicate status folders, but the stale folder itself remains and violates the status-=-parent-folder invariant every reader relies on.
- verified-at-filing: `git ls-tree --name-only origin/main tasks/reviewing/ | grep 1196` → present; `git ls-tree --name-only origin/main tasks/completed/ | grep -x tasks/completed/1196` → present; REGISTRY says `status: completed, path: tasks/completed/1196` (2026-07-12).

## Proposed change

Apply the documented #644/#1253 duplicate-folder remediation: in a scratch worktree detached at `origin/main`, verify the canonical folder's contents supersede the stale one (union-merge `events.jsonl` rows into the canonical copy if the stale folder holds rows the canonical lacks), `git rm -r tasks/reviewing/1196`, commit, push `HEAD:main`; then `task.py audit` to confirm registry/filesystem consistency.

## Scope / surfaces

- `tasks/reviewing/1196` (deletion of the stale duplicate only; canonical `tasks/completed/1196` untouched except a possible events union-merge).
- Never a repo-root branch switch or reset (use a scratch worktree per CLAUDE.md).

## Constraints / invariants

- Preserve every unique `events.jsonl` row (union into canonical before deleting the duplicate — no marker loss).
- `uv run python scripts/task.py audit` clean afterward.

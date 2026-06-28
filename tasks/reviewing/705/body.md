---
title: fix confirm_artifacts finalize gate on phase-scoped / multi-attempt launches
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:47Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: the confirm_artifacts finalize gate fails on phase-scoped / multi-attempt launches, forcing --skip-confirm-artifacts every time.

## Goal

Stop forcing `--skip-confirm-artifacts` on every phase-scoped / multi-attempt launch by fixing the finalize gate's path resolution.

## Problem (from /daily 2026-06-27)

The finalize gate fails on phase-scoped / multi-attempt launches (sessions 92450661 #661, 6c469aa1 #685): it resolves the repo-root/main tree + a stale attempt sentinel instead of the issue-branch worktree + the current attempt, so the gate always fails and `--skip-confirm-artifacts` is used every time. Routinely skipping the artifact-confirmation gate defeats its purpose.

## Proposed change

Fix the finalize path to resolve the issue-branch worktree + current attempt sentinel (not the repo-root/main tree + a stale attempt). Add a test covering the phase-scoped / multi-attempt case so the gate passes without `--skip-confirm-artifacts`.

## Scope / target files

- `scripts/dispatch_issue.py`
- `src/explore_persona_space/backends/` (the finalize/confirm-artifacts path)
- `tests/`

## Constraints

- Workflow-surface only (dispatch_issue.py + backends/ are in the in-scope list).
- Lint gate green; ruff clean on touched files.
- Don't relax the gate to "pass" — fix the resolution so it actually confirms artifacts on these launches.

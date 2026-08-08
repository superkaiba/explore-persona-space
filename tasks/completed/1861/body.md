---
title: 'daily-fix: repo-root guard recognizes cd-to-worktree via var'
kind: infra
tags:
- wf-fix
- wf-fix-fp:913b1da1aa4e
- daily-auto-filed
- trigger-dense
created_at: '2026-07-30T07:08:04Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-29 problem sweep (route 2): The guard blocked worktree-scoped
  `git checkout -- <paths>` composed as `WT=.../worktrees/...; cd "$WT"; git checkout
  --` in 2 sessions (2 firings) — the cd-worktree allow cannot see variable indirection'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-29 (problem sweep; emitting source: miner G-P5 (2 firings, 2 sessions; probed)).

## Goal

A worktree-scoped destructive git op composed via the common VAR+cd idiom should pass the guard's worktree allowance like the literal-cd form does.

## Workflow gap

- **Bug observed:** Two sessions composed `WT=...; cd "$WT"; git checkout -- <paths>` and were blocked; both had to recompose (git -C form). Fail-closed, nothing lost — but the idiom is the SKILL's own canonical shape elsewhere.
- **Why it is a workflow gap:** the detector matches any `git checkout ... -- ` without honoring a prior variable-indirect cd; the allow only sees literal `cd <path>`. 
- **Confidence (emitter):** medium
- verified-at-filing: miner probe: guard source L43 (cd-worktree allow comment) + detector ~L1665 matches regardless of prior indirect cd (2026-07-30).

## Proposed change (refine in planning)

Track simple VAR=literal assignments within the same compound when resolving the cd target, or amend the BLOCK text to name the `git -C "$WT"` recompose explicitly.

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh`
- Grep the workflow surface for the pattern before editing and update every hit.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh
- fingerprint: 913b1da1aa4e

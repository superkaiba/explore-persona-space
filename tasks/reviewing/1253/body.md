---
title: 'post-merge stale-task-folder recipe: checkout clause always '
kind: infra
tags:
- wf-fix
- wf-fix-fp:eaef71acc389
- daily-auto-filed
created_at: '2026-07-10T06:55:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-09 problem sweep (route 2): the documented recovery
  recipe''s git checkout fallback clause trips guard_repo_root_branch every time it
  runs from a cd-worktree compound (1198 x2, 1155)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-09 from the nightly transcript problem sweep (miner 03).

## Goal

Remove (or rewrite in `git -C` form) the checkout-fallback clause in the /issue SKILL.md post-merge stale-task-folder guard recipe, which as written ALWAYS trips the `guard_repo_root_branch.sh` PreToolUse hook when executed from a `cd <worktree>` compound.

## Workflow gap

- **Bug observed:** Sessions #1198 (×2) and #1155 (2026-07-09) executed the documented post-merge stale-task-folder recovery recipe; its `git checkout <pathspec>` fallback clause was hook-blocked each time (the guard matches command text, not cwd), and the sessions recovered via plain `git rm` instead — i.e. the documented recipe contains a clause that can never run.
- **Why it is a workflow gap:** A SKILL.md recipe that reliably trips a project hook costs a blocked turn + an improvised recovery every time it is followed; the recipe should prescribe the working form (`git -C <worktree> ...`, or the `git rm` path the sessions actually used).

## Proposed change (refine in planning)

In the Step 10d post-merge stale-task-folder guard recipe, replace the `git checkout <pathspec>` fallback with the `git -C "$WT" checkout -- <pathspec>` form (which passes the guard) or drop it in favor of the `git rm`/`git restore --source` path; verify against the guard's self-test that the prescribed form is not blocked.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only. SKILL-content-pinning test suites stay green; `scripts/workflow_lint.py --check-asks` passes.
- This session runs under EPM_WORKFLOW_FIX_SESSION=1 semantics (workflow_fix_target Provenance line) — it MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- fingerprint: eaef71acc389

- workflow_fix_target: .claude/skills/issue/SKILL.md

---
title: 'daily-fix: repo-root guard false-positive on worktree-scoped'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c8206fd3a275
- daily-auto-filed
created_at: '2026-07-05T07:03:36Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): The shared spec-freshness
  sync recipe (cd into the issue worktree, then a pathspec-scoped git checkout of
  main specs) was blocked by guard_repo_root_branch.sh in three sessions on 2026-07-04
  (#779 01:49, #813 06:46, #964 07:49 UTC) even though the command demonstrably targets
  a worktree - the guard string-matches without honoring the cd-into-worktree prefix;
  one parallel tool call was cancelled as'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

Either teach the guard to skip when the command targets a path under .claude/worktrees/ (a cd-into-worktree prefix or git -C <worktree>), and to ignore matches inside heredoc/document text, or rewrite the shared recipe to the git -C form the guard already accepts - make the recipe + guard consistent.

## Workflow gap

- **Bug observed:** The shared spec-freshness sync recipe (cd into the issue worktree, then a pathspec-scoped git checkout of main specs) was blocked by guard_repo_root_branch.sh in three sessions on 2026-07-04 (#779 01:49, #813 06:46, #964 07:49 UTC) even though the command demonstrably targets a worktree - the guard string-matches without honoring the cd-into-worktree prefix; one parallel tool call was cancelled as collateral. The 2026-07-05 /daily run hit a 4th instance: the guard blocked a python heredoc whose BODY TEXT merely mentioned the banned command.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/guard_repo_root_branch.sh, .claude/skills/issue/SKILL.md`
- Sessions: af3e8739 (#779), d36ef80b (#813), e5bd408a (#964), plus this /daily run. The recipe recurs every spec-freshness sync, so this false-positive keeps firing until fixed.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/guard_repo_root_branch.sh, .claude/skills/issue/SKILL.md
- source: /daily 2026-07-04 problem sweep (transcript-mined)

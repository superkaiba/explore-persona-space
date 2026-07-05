---
title: 'daily-fix: new_worktree pre-adds own issue eval_results cone'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2e6d7ee804d0
- daily-auto-filed
created_at: '2026-07-05T07:03:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-04 problem sweep (route 2): #906''s artifact commit
  failed on 2026-07-04 (15:14 UTC) because sparse worktrees exclude eval_results/
  by default and the session''s OWN output dir (eval_results/issue_906/) was not coned
  in - ''paths exist outside of your sparse-checkout definition''. Sibling of the
  #671 fixtures gap, but for the task''s own artifacts.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-04 (route 2: behavior/logic change -> independent review), from the nightly transcript problem sweep.

## Goal

new_worktree.sh pre-adds eval_results/issue_<N>/ (the worktree's own issue) to the sparse cone at creation, alongside the tests/sparse_cones.txt fixture cones.

## Workflow gap

- **Bug observed:** #906's artifact commit failed on 2026-07-04 (15:14 UTC) because sparse worktrees exclude eval_results/ by default and the session's OWN output dir (eval_results/issue_906/) was not coned in - 'paths exist outside of your sparse-checkout definition'. Sibling of the #671 fixtures gap, but for the task's own artifacts.
- **Why it is a workflow gap:** the failure originates in the workflow surface / helper named below, not in any one experiment.
- **Confidence (emitter):** medium

## Scope / surfaces

- Primary target: `scripts/new_worktree.sh`
- Session: 4ea4c2b6 (#906).

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` + `--check-references` stay green; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/new_worktree.sh
- source: /daily 2026-07-04 problem sweep (transcript-mined)

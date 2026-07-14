---
title: 'workflow-fix: workflow_lint no-flags run times out from a wo'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c0815954cce1
- daily-auto-filed
created_at: '2026-07-09T06:57:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The no-flags workflow_lint
  run exceeded a 510s timeout with cwd inside an issue worktree but passes in seconds
  from repo root — root/tree derivation appears to scan the worktree tree (including
  its data/ caches).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1019 by a recursion-guarded workflow-fix session.

## Goal

Fix workflow_lint root/tree derivation so a worktree-cwd invocation scans the same bounded file set as a repo-root invocation (resolve scanning to the invoking checkout's tracked surface; never walk worktree-local data/ download caches).

## Workflow gap

- **Bug observed:** The no-flags workflow_lint run exceeded a 510s timeout with cwd inside an issue worktree but passes in seconds from repo root — root/tree derivation appears to scan the worktree tree (including its data/ caches).
- **Why it is a workflow gap:** The lint is a pre-commit / pre-push gate that implementer sessions routinely run from issue worktrees; a 510s+ hang there defeats the gate.
- **Confidence (emitter):** medium
- **Sweep verification (2026-07-08):** No commit since 2026-06-30 on scripts/workflow_lint.py addresses worktree-cwd performance (git log reviewed 2026-07-08); _other_worktree_prefix (line 1647) only excludes OTHER worktrees — the current worktree is deliberately scanned, so the reported slow path is still plausible and unfixed. Reproduce before fixing (time the no-flags run from a live issue worktree).

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up; profile the no-flags run from a worktree cwd, then bound the scan — e.g. skip data/, .venv/, hf_dl caches inside the current worktree, or derive file lists from git ls-files)

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard, workflow-fix-on-bug.md).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #1019 at 2026-07-04T23:50:44Z

Verbatim parked note:

> source: prose-followup (round-2 implementer). target_file: scripts/workflow_lint.py. bug_observed: no-flags run exceeded a 510s timeout with cwd inside an issue worktree but passes in seconds from repo root — root/tree derivation may scan the worktree tree. routed: parked — running under workflow_fix_target Provenance (recursion guard, .claude/rules/workflow-fix-on-bug.md § Recursion guard); surface for the next human/orchestrator pass.

---
title: 'daily-fix: 10d recovery syncs lint targets in stale worktree'
kind: infra
tags:
- wf-fix
- wf-fix-fp:86cce36e890c
- daily-auto-filed
created_at: '2026-07-25T06:48:50Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): The 1622 Step 10d merge-recovery
  session hit a pre-commit workflow-yaml-lint FAIL because the 1240-behind worktree
  lacked scripts/plan_patch.py referenced by adversarial-planner SKILL.md - the cross-reference
  lint ran against stale worktree targets and recovery needed an ad-hoc sync of about
  25 workflow files from main'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 76a8649e, task #1622).

## Goal

A Step 10d merge-recovery in a stale worktree must not fail pre-commit reference lint on files the round never touched.

## Workflow gap

- **Bug observed:** the watcher-respawned #1622 merge-recovery hit `workflow-yaml-lint ... references 'scripts/plan_patch.py' which does not exist under .../issue-1622/scripts/` (exit 1) — the worktree was ~1240 commits behind and the referenced helper exists on origin/main. The session recovered by ad-hoc syncing ~25 workflow-surface files from main (re-lint PASS) before being killed by the reconcile pass (separate filing).
- **Why it is a workflow gap:** the Step 10d recovery recipe skips the blind sync when the branch carries feature edits, but the reference lint validates against the WORKTREE tree — so stale lint-target files (untouched by the round) produce false FAILs with no prescribed remedy.
- **Confidence (emitter):** medium (exact recipe placement for the targeted sync is a planner call).
- verified-at-filing: `ls scripts/plan_patch.py` → exists on main; incident evidence is the 76a8649e transcript hook-failure firing (1 firing) — session records. `grep -n 'plan_patch' .claude/skills/adversarial-planner/SKILL.md` → reference present.

## Proposed change (candidate diff sketch — refine in planning)

In SKILL.md Step 10d merge-recovery pre-gate: when the blind sync is skipped, add a targeted sync step for lint cross-reference TARGETS (scripts/ helpers + sibling .claude files not touched by the round) before any commit — or prescribe running the reference lint against the merge-base tree.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md` (Step 10d recovery recipe)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 86cce36e890c

- workflow_fix_target: .claude/skills/issue/SKILL.md

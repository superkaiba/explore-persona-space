---
title: 'workflow-fix: daily SKILL push recovery via sync_repo_root.p'
kind: infra
tags:
- wf-fix
- wf-fix-fp:07e9cc95c142
- daily-auto-filed
created_at: '2026-07-09T06:56:15Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): A hand-rolled repo-root
  `git push || { git pull --rebase=merges --autostash && git push; }` recovery recipe
  survives at .claude/skills/daily/SKILL.md:540, outside the #1047 fix (which covered
  only the /issue skill''s Step 10d sites).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1047 (park_form: recursion-guard).

## Goal

Route the daily-stub push-recovery recipe at .claude/skills/daily/SKILL.md:540 through scripts/sync_repo_root.py and re-sweep the workflow surface for any other surviving hand-rolled repo-root pull recipes.

## Workflow gap

- **Bug observed:** A hand-rolled repo-root `git push || { git pull --rebase=merges --autostash && git push; }` recovery recipe survives at .claude/skills/daily/SKILL.md:540, outside the #1047 fix (which covered only the /issue skill's Step 10d sites).
- **Why it is a workflow gap:** Hand-rolled pull-rebase recovery on the shared repo root is exactly the unserialised path sync_repo_root.py exists to replace (flock-serialised, autostash-recovering, merge-preserving).
- **Confidence (emitter):** medium (codex-critic alternatives lens, concern D)

## Proposed change (candidate diff sketch — refine in planning)

- git push origin HEAD:main || { git pull --rebase=merges --autostash && git push origin HEAD:main; }
+ git push origin HEAD:main || uv run python scripts/sync_repo_root.py

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md
- origin: parked candidate on task #1047 at 2026-07-05T13:48:41Z

source: prose-followup (codex-critic, alternatives lens, concern D). target_file: repo-wide sweep — any .claude/skills/**/SKILL.md, .claude/agents/*.md, .claude/rules/*.md carrying a hand-rolled repo-root 'git pull --rebase=merges --autostash' recipe outside .claude/skills/issue/SKILL.md. proposed_change: grep the workflow surface for surviving raw repo-root pull recipes and route each to sync_repo_root.py (the #1047 fix covers only the /issue skill's Step 10d sites). routed: parked — running under workflow_fix_target recursion guard; log + notify, not auto-filed. related_task: #1047

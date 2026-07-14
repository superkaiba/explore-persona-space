---
title: 'daily-fix: new_worktree.sh branches from origin/main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1ef02c7c2c54
- daily-auto-filed
created_at: '2026-07-09T07:01:39Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): 3 of 6 infra sessions today
  needed 10-25 min of bespoke Step 10d merge forensics because worktree branches are
  cut from LOCAL shared-root main, which accretes/rewrites unpushed task-state commits
  (78 baked-in commits in one case).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Cut worktree branches from origin/main so Step 10d merges stop inheriting unpushed local task-state churn.

## Workflow gap

- **Bug observed:** Transcripts d4b25d2d (#1142: branch cut from a since-rewritten local-main lineage, guard 3 flagged 38 foreign paths), 301cd696 (#1144: #1041 refusal + squash retry), af70601c (#1145: 78 unpushed repo-root task-state commits baked in, cherry-pick recovery) — 3 of 6 sessions, 10-25 min forensics each; guards caught every case.
- **Why it is a workflow gap:** The downstream half (#1147 invariant tests) landed today; the upstream cause — branching from local main — is untouched.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

In new_worktree.sh: `git fetch origin main` then `git worktree add -b <branch> <path> origin/main` (keep current behavior behind an explicit --base-local flag); update the usage text.

## Scope / surfaces

- Primary target: `scripts/new_worktree.sh`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/new_worktree.sh
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-B P4

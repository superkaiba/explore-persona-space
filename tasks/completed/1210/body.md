---
title: 'daily-fix: Step 10d re-snapshots Guard-1 strip on merge conf'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4235a8ed99e4
- daily-auto-filed
created_at: '2026-07-09T07:01:25Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): #1128 Step 10d merge failed
  twice under fleet marker churn — the Guard-1 strip snapshot of foreign tasks/ went
  stale between snapshot and merge; two worktree recovery merges, ~35 min extra.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-08 problem sweep (transcript-mined; emitting agent: /daily orchestrator).

## Goal

Make the Step 10d merge robust to fleet marker churn between the Guard-1 snapshot and the merge.

## Workflow gap

- **Bug observed:** Transcript 78a4234b (issue-1128) ~08:48-09:22Z: 'GraphQL: Pull Request has merge conflicts (mergePullRequest)' then 'MERGE FAILED (non-lint transport failure)'; completion note 09:24Z records two worktree recovery merges with foreign tasks/ ultimately pinned to a single captured main SHA.
- **Why it is a workflow gap:** The recovery recipe (pin to one captured main SHA) is only discovered as the SECOND fallback; under heavy concurrent marker commits the first-attempt snapshot is routinely stale.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Step 10d merge-guard block: capture MAIN_SHA once, strip foreign tasks/ against that SHA, and on a mergePullRequest-conflict error re-snapshot against a freshly captured SHA and retry once (documented, not silent).

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` + `--check-references` pass; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md
- origin: transcript-mined by /daily 2026-07-08 problem sweep
- evidence: mine-D P3 (78a4234b ~08:48-09:24Z)

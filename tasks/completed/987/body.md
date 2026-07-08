---
title: 'workflow-fix: dispatch lane infra from main on issue branches'
kind: infra
tags:
- daily-auto-filed
- wf-fix
- wf-fix-fp:42464ec7a886
created_at: '2026-07-04T07:12:35Z'
has_clean_result: false
origin_prompt: 'parked — running under workflow_fix_target recursion guard, see .claude/rules/workflow-fix-on-bug.md
  § Recursion guard. Candidate (surfaced by the #935 planner, plan §2/§4.7): long-lived
  issue branches dispatch with a STALE lane template — dispatch_issue.py invoked from
  a worktree imports the branch''s backends/gcp.py (Step 5a spec-freshness deliberately
  excludes src/), so pre-#935 worktrees keep launching poweroff-less GCE instances
  until rebased. Proposed target: make dispatch source lane infra (backends/) from
  main while cloning --repo-branch for the workload only — a dispatch-contract change
  with version-skew risk that needs its own planned task. target_file: scripts/dispatch_issue.py,
  src/explore_persona_space/backends/*.py. NOT auto-filed by this session (recursion
  guard); next non-workflow-fix orchestrator pass may route it.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-03 from a recursion-guard-parked workflow-fix candidate.

## Goal

make dispatch source lane infra (backends/) from main while cloning --repo-branch for the workload only — a dispatch-contract change with version-skew risk needing its own planned task

## Workflow gap

- **Bug observed:** long-lived issue branches dispatch with a STALE lane template — dispatch_issue.py invoked from a worktree imports the branch's backends/gcp.py (Step 5a spec-freshness excludes src/), so pre-#935 worktrees keep launching poweroff-less GCE instances until rebased
- **Why it is a workflow gap:** see candidate note
- **Confidence (emitter):** medium

## Proposed change (candidate sketch — refine in planning)

make dispatch source lane infra (backends/) from main while cloning --repo-branch for the workload only — a dispatch-contract change with version-skew risk needing its own planned task

## Scope / surfaces

- Primary target: `scripts/dispatch_issue.py, src/explore_persona_space/backends/*.py`
- Grep the workflow surface for the pattern before editing; list every hit in the plan.

## Constraints / invariants

- Workflow-surface only; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: scripts/dispatch_issue.py, src/explore_persona_space/backends/*.py
- fingerprint: 42464ec7a886

parked — running under workflow_fix_target recursion guard, see .claude/rules/workflow-fix-on-bug.md § Recursion guard. Candidate (surfaced by the #935 planner, plan §2/§4.7): long-lived issue branches dispatch with a STALE lane template — dispatch_issue.py invoked from a worktree imports the branch's backends/gcp.py (Step 5a spec-freshness deliberately excludes src/), so pre-#935 worktrees keep launching poweroff-less GCE instances until rebased. Proposed target: make dispatch source lane infra (backends/) from main while cloning --repo-branch for the workload only — a dispatch-contract change with version-skew risk that needs its own planned task. target_file: scripts/dispatch_issue.py, src/explore_persona_space/backends/*.py. NOT auto-filed by this session (recursion guard); next non-workflow-fix orchestrator pass may route it.

---
title: 'workflow-fix: enforce HF 10k-files-per-dir upload limit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cfb27ea16683
- daily-auto-filed
created_at: '2026-07-09T06:59:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The HF Hub 10000-files-per-directory
  commit limit (#658 r2) is documented in gotchas.md but has no mechanical enforcement
  — unlike its sibling HF-upload traps (--check-upload-as-file; the poller gpu-idle
  escalation) — so a workload emitting >10k sibling files into one HF directory silently
  re-hits a non-retriable BadRequestError after the full compute has already run.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #748 (recursion-guarded workflow-fix session).

## Goal

Add a workflow_lint check or hub-upload-time assert flagging a workload uploading >10000 files into a single repo directory (or a single upload_folder commit staging >10000 sibling files into one dir), mirroring the --check-upload-as-file backstop pattern.

## Workflow gap

- **Bug observed:** The HF Hub 10000-files-per-directory commit limit (#658 r2) is documented in gotchas.md but has no mechanical enforcement — unlike its sibling HF-upload traps (--check-upload-as-file; the poller gpu-idle escalation) — so a workload emitting >10k sibling files into one HF directory silently re-hits a non-retriable BadRequestError after the full compute has already run.
- **Why it is a workflow gap:** Documentation alone is the same 'documented but not enforced' pattern that let #664/#658 recur; only a human who reads gotchas.md first is protected.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

Option A (runtime, preferred): in the shared upload helper (orchestrate/hub.py), before upload_folder, count staged files per top-level dir and raise/shard when any dir >10000.
Option B (lint): flag upload_folder call sites lacking a documented per-dir sharding strategy. Planner picks.

## Scope / surfaces

- Primary target: `scripts/workflow_lint.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/workflow_lint.py
- origin: parked candidate on task #748 at 2026-06-30T04:18:26Z

Verbatim parked note:

parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target (.claude/rules/gotchas.md), see .claude/rules/workflow-fix-on-bug.md § Recursion guard. Surfaced by methodology critic on plan v1 (NOT a redundant re-raise of #748; distinct fingerprint).

Candidate source: methodology critic, round 1.
target_file: scripts/workflow_lint.py
bug_observed: the HF Hub 10000-files-per-directory commit limit (#658 r2) is being documented in gotchas.md but has no mechanical enforcement, unlike its two sibling HF-upload traps (line 101 has --check-upload-as-file, line 102 has the poller gpu-idle escalation); documentation alone is the same "documented but not enforced" pattern that let #664/#658 recur.
why_workflow_gap: a future workload emitting >10k sibling files into one HF directory will silently re-hit a non-retriable BadRequestError after the full compute has already run — the doc bullet only helps a human who reads gotchas.md first.
proposed_change: add a workflow_lint or hub-upload-time assert that flags a workload uploading >10000 files into a single repo directory (or a single upload_folder commit staging >10000 files into one dir), mirroring the --check-upload-as-file backstop pattern.
confidence: medium
related_task: #748 (this task documents; that one enforces)

Routing: PARKED under recursion guard. Distinct fingerprint from #748 (target_file: scripts/workflow_lint.py vs .claude/rules/gotchas.md; alternatives critic also weighed and ruled this NOT a blocker on the current doc fix). Recovered after this workflow-fix session completes: an interactive orchestrator (or a /daily pass) re-raising the candidate outside the recursion-guarded session will auto-file + spawn under the normal workflow-fix-on-bug default; alternatives critic noted "premature at N=1 recurrence; lint check is the right deferred follow-up, not a reason to bounce the doc bullet."

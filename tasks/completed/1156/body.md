---
title: 'workflow-fix: poller WARN: pid file older than run-launched '
kind: infra
tags:
- wf-fix
- wf-fix-fp:eabe5edc0545
- daily-auto-filed
created_at: '2026-07-09T06:57:28Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): A stale pid file surviving
  from a prior launch can mask a dead relaunch; the poller has no mechanical detection
  when the pid file''s mtime predates the newest epm:run-launched marker.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep from a candidate parked on task #1070 (park_form: recursion-guard).

## Goal

Add a WARN-only poller backstop: poll_pipeline.py warns when the pod-side pid file's mtime predates the newest epm:run-launched marker ts (never a verdict change).

## Workflow gap

- **Bug observed:** A stale pid file surviving from a prior launch can mask a dead relaunch; the poller has no mechanical detection when the pid file's mtime predates the newest epm:run-launched marker.
- **Why it is a workflow gap:** This is the mechanical layer against the residual 'agent skips the pid-file-rewrite prose contract' failure mode (pod-side-reporting rule, #813 family); prose-only contracts have already been skipped once.
- **Confidence (emitter):** medium (alternatives critic round 1 on #1070)

## Proposed change (candidate diff sketch — refine in planning)

In the poll loop, after resolving the pid file and the latest epm:run-launched ts: if pid_file_mtime + slack < run_launched_ts: log.warning('pid file predates newest run-launched marker — possible stale pid from a prior launch'). WARN only; no verdict change.

## Scope / surfaces

- Primary target: `scripts/poll_pipeline.py`
- Grep the workflow surface for the pattern before editing (`grep -rln '<pattern>' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/poll_pipeline.py
- origin: parked candidate on task #1070 at 2026-07-05T16:43:04Z

Prose follow-up surfaced by the alternatives critic (round 1) — PARKED, NOT ROUTED (workflow-fix recursion guard; routed: parked: EPM_WORKFLOW_FIX_SESSION): a poller-side detection backstop — poll_pipeline.py WARNs when the pod-side pid file's mtime predates the newest epm:run-launched marker ts (WARN only, never a verdict change). This is the mechanical layer against the residual 'agent skips the prose contract' failure mode; deliberately dispositioned out-of-scope in plan §4/§10 of #1070 (a hot shared-script behavior change; would not have rescued #813 where the marker pid was also dead). target_file: scripts/poll_pipeline.py. confidence: medium. related_task: #1070. A future human/orchestrator pass may file it.

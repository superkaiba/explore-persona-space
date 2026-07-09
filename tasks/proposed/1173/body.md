---
title: 'workflow-fix: daily route-2 bodies omit workflow_fix_target'
kind: infra
tags:
- wf-fix
- wf-fix-fp:8fd193af15b6
- daily-auto-filed
created_at: '2026-07-09T06:58:46Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): The /daily route-2 filing
  template does not mandate the ''- workflow_fix_target: <path>'' Provenance line,
  so some daily-filed wf-fix task bodies lack it (confirmed: tasks/completed/1134/body.md
  has ## Provenance but no workflow_fix_target line) and task_workflow.is_workflow_fix_session()
  returns False — the durable recursion-guard signal is missing on those sessions
  (they survive on the env-var leg'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1134 (recursion-guarded workflow-fix session).

## Goal

Make every daily-filed workflow-fix task carry the durable recursion-guard signal so a crash-recovery respawn stays guarded.

## Workflow gap

- **Bug observed:** The /daily route-2 filing template does not mandate the '- workflow_fix_target: <path>' Provenance line, so some daily-filed wf-fix task bodies lack it (confirmed: tasks/completed/1134/body.md has ## Provenance but no workflow_fix_target line) and task_workflow.is_workflow_fix_session() returns False — the durable recursion-guard signal is missing on those sessions (they survive on the env-var leg only, which a watcher respawn loses).
- **Why it is a workflow gap:** the fix targets the workflow surface (.claude/skills/daily/SKILL.md, scripts/daily_drive_filings.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/skills/daily/SKILL.md route-2 body template: add under ## Provenance:
+ - workflow_fix_target: <target_file>
scripts/daily_drive_filings.py: warn-or-inject when a wf-fix-tagged body's
## Provenance lacks a workflow_fix_target: line.
```

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md, scripts/daily_drive_filings.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/daily/SKILL.md, scripts/daily_drive_filings.py
- origin: parked candidate on task #1134 at 2026-07-08T07:32:02Z

Verbatim parked note:

> routed: parked — wf-fix task (tags wf-fix; recursion guard honored despite missing workflow_fix_target Provenance line). TWO candidates surfaced by round-1 critics, logged not auto-filed: (A) [methodology critic] target_file: scripts/verify_plan.py (+.claude/agents/planner.md) — add a check that a plan inserting protection prose into .claude/skills/**/SKILL.md either names a standing pin test or carries a one-line no-pin justification (3rd recurrence of the missing-durability-pin shape: #884, #1045, #1134); confidence medium. (B) [alternatives critic] target_file: scripts/autonomous_session_watch.py — non-gating observer pass (triage-observer shape, #967) flagging doubled-site rounds with disagreeing verdict markers and no epm:review-reconcile / no-show evidence — mechanically detects a #825 repeat and the no-show-misclassification incentive; confidence low-medium. ALSO note: /daily route-2 filing template omits the workflow_fix_target: Provenance line so task_workflow.is_workflow_fix_session()=False on daily-filed wf-fix tasks — same parked disposition, target .claude/skills/daily/SKILL.md.

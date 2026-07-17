---
title: 'daily-fix: widen closed-sibling advisory to infra tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d5d3390b4f22
- daily-auto-filed
created_at: '2026-07-17T06:56:06Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): the #1399 recently-closed-sibling
  advisory (recent_closed_workflow_fix_tasks) enumerates only workflow-fix:/daily-fix:-title-prefixed
  closed tasks (task_workflow.py L1060 startswith(WF_FIX_TITLE_PREFIXES)), so an ordinary
  closed infra task that landed the same fix (#1360) is invisible to the filer''s
  advisory — the mechanical half of the #1386-over-#1360 duplicate filing'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 Step C from a parked FORMAL candidate on task #1420 (Alternatives critic; fp 038e003c95f4).

## Goal

Make the filing-time recently-closed-sibling advisory see ordinary closed infra tasks, not only title-prefixed workflow-fix tasks.

## Workflow gap

- **Bug observed:** the #1399 recently-closed-sibling advisory (recent_closed_workflow_fix_tasks) enumerates only workflow-fix:/daily-fix:-title-prefixed closed tasks (task_workflow.py L1060 startswith(WF_FIX_TITLE_PREFIXES)), so an ordinary closed infra task that landed the same fix (#1360) is invisible to the filer's advisory — the mechanical half of the #1386-over-#1360 duplicate filing
- **Why it is a workflow gap:** The advisory exists to surface just-closed duplicates at filing time, but its title-prefix filter structurally excludes the non-prefixed infra-task population where functional fixes routinely land.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'WF_FIX_TITLE_PREFIXES' src/explore_persona_space/task_workflow.py` -> L1003 (def), L1060 (the startswith filter inside the enumerate loop), L1182; `grep -n 'def recent_closed_workflow_fix_tasks' src/explore_persona_space/task_workflow.py` -> L1192

## Proposed change (candidate diff sketch — refine in planning)

Add an opt-in second pass over recently-completed/archived kind:infra tasks (last ~7d) matched by target_file path token; keep the 10-most-recent cap and stderr-advisory-only semantics.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py`
- Secondary: scripts/file_infra_task.py (threads the advisory)
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: d5d3390b4f22


Verbatim formal candidate block lives on task #1420 events.jsonl (epm:workflow-fix-candidate, 2026-07-16T11:20:11Z, fp 038e003c95f4).

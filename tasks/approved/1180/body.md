---
title: 'workflow-fix: dedup predicate blind to daily-fix: titles'
kind: infra
tags:
- wf-fix
- wf-fix-fp:268c34e234e2
- daily-auto-filed
created_at: '2026-07-09T06:59:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): task_workflow.is_open_workflow_fix_task
  requires the ''workflow-fix:'' title prefix (task_workflow.py:1044) while /daily
  route-2 filings use ''daily-fix:'' titles — so an OPEN daily-filed wf-fix task is
  invisible to the orchestrator-channel (target_file, fingerprint) dedup and a same-fingerprint
  candidate raised in a normal session double-files. daily_drive_filings.py:270 explicitly
  documents the mism'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1102 (recursion-guarded workflow-fix session).

## Goal

Close the cross-channel dedup gap so a bug filed by /daily cannot be double-filed by the workflow-fix-on-bug orchestrator path (and vice versa).

## Workflow gap

- **Bug observed:** task_workflow.is_open_workflow_fix_task requires the 'workflow-fix:' title prefix (task_workflow.py:1044) while /daily route-2 filings use 'daily-fix:' titles — so an OPEN daily-filed wf-fix task is invisible to the orchestrator-channel (target_file, fingerprint) dedup and a same-fingerprint candidate raised in a normal session double-files. daily_drive_filings.py:270 explicitly documents the mismatch and works around it only for its OWN channel (find_open_fp_duplicate tag-scan).
- **Why it is a workflow gap:** the fix targets the workflow surface (src/explore_persona_space/task_workflow.py, scripts/daily_drive_filings.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
# task_workflow.py is_open_workflow_fix_task:
- if not str(entry.get("title", "")).startswith("workflow-fix:"):
+ if not str(entry.get("title", "")).startswith(("workflow-fix:", "daily-fix:")):
      continue
# and update the daily_drive_filings.py:270 docstring note accordingly.
```

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py, scripts/daily_drive_filings.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py, scripts/daily_drive_filings.py
- origin: parked candidate on task #1102 at 2026-07-07T08:19:50Z

Verbatim parked note:

> parked: EPM_WORKFLOW_FIX_SESSION / workflow_fix_target — see workflow-fix-on-bug.md § Recursion guard. Candidate (from consistency-checker, source: prose-followup): the /daily auto-filer titles workflow-fix tasks 'daily-fix:' but task_workflow.is_open_workflow_fix_task requires the 'workflow-fix:' title prefix, so #1102 is invisible to the (target_file, fingerprint) dedup while open — a same-fingerprint re-raise would double-file. target_file: scripts/daily_drive_filings.py (or the /daily skill's filing template) vs src/explore_persona_space/task_workflow.py prefix predicate. Logged only — this session runs under the workflow-fix recursion guard; route on a future non-guarded pass.

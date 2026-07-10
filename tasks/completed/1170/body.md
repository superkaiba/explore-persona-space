---
title: 'workflow-fix: observer for unreconciled doubled-site verdict'
kind: infra
tags:
- wf-fix
- wf-fix-fp:68ca3fb4c392
- daily-auto-filed
created_at: '2026-07-09T06:58:23Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): A doubled review site whose
  Claude and Codex twins post DISAGREEING verdict markers with no epm:review-reconcile
  marker and no no-show evidence goes undetected — the #825 misclassification shape
  (treating a disagreement as a no-show) has no mechanical detector.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1134 (recursion-guarded workflow-fix session).

## Goal

Mechanically detect a #825 repeat: doubled-site rounds where a PASS-vs-FAIL disagreement was never reconciled.

## Workflow gap

- **Bug observed:** A doubled review site whose Claude and Codex twins post DISAGREEING verdict markers with no epm:review-reconcile marker and no no-show evidence goes undetected — the #825 misclassification shape (treating a disagreement as a no-show) has no mechanical detector.
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/autonomous_session_watch.py); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
# new pass, modeled on _triage_observer (#967, asw :3853):
for issue in recent_review_rounds():
    v_claude, v_codex = round_verdict_markers(issue)
    if disagree(v_claude, v_codex) and not (has_reconcile_marker(issue) or no_show_evidence(issue)):
        sidecar_row(...); alert_once(...)
# kill switch: EPM_DISABLE_VERDICT_DISAGREE_OBSERVER=1
```

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- origin: parked candidate on task #1134 at 2026-07-08T07:32:02Z

Verbatim parked note:

> routed: parked — wf-fix task (tags wf-fix; recursion guard honored despite missing workflow_fix_target Provenance line). TWO candidates surfaced by round-1 critics, logged not auto-filed: (A) [methodology critic] target_file: scripts/verify_plan.py (+.claude/agents/planner.md) — add a check that a plan inserting protection prose into .claude/skills/**/SKILL.md either names a standing pin test or carries a one-line no-pin justification (3rd recurrence of the missing-durability-pin shape: #884, #1045, #1134); confidence medium. (B) [alternatives critic] target_file: scripts/autonomous_session_watch.py — non-gating observer pass (triage-observer shape, #967) flagging doubled-site rounds with disagreeing verdict markers and no epm:review-reconcile / no-show evidence — mechanically detects a #825 repeat and the no-show-misclassification incentive; confidence low-medium. ALSO note: /daily route-2 filing template omits the workflow_fix_target: Provenance line so task_workflow.is_workflow_fix_session()=False on daily-filed wf-fix tasks — same parked disposition, target .claude/skills/daily/SKILL.md.

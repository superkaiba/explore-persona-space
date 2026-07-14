---
title: 'workflow-fix: verify_plan check for SKILL.md-prose pin tests'
kind: infra
tags:
- wf-fix
- wf-fix-fp:08362b67c1e0
- daily-auto-filed
created_at: '2026-07-09T06:59:08Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-08 problem sweep (route 2): A plan that inserts protection
  prose into .claude/skills/**/SKILL.md without naming a standing pin test (or a one-line
  no-pin justification) passes verify_plan — the missing-durability-pin shape recurred
  3x (#884, #1045, #1134).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily Step-C parked-candidate sweep (2026-07-08) from a candidate parked on task #1134 (recursion-guarded workflow-fix session).

## Goal

Stop unpinned SKILL.md protection prose from shipping — the 3rd recurrence of the missing-durability-pin shape.

## Workflow gap

- **Bug observed:** A plan that inserts protection prose into .claude/skills/**/SKILL.md without naming a standing pin test (or a one-line no-pin justification) passes verify_plan — the missing-durability-pin shape recurred 3x (#884, #1045, #1134).
- **Why it is a workflow gap:** the fix targets the workflow surface (scripts/verify_plan.py, .claude/agents/planner.md); the originating session was recursion-guarded and could not route it.
- **Confidence (emitter):** see parked note below.

## Proposed change (candidate diff sketch — refine in planning)

```
def check_skill_md_prose_names_pin(plan_text):
    if touches_skill_md(plan_text) and inserts_protection_prose(plan_text):
        if not (names_pin_test(plan_text) or has_no_pin_justification(plan_text)):
            return WARN_OR_FAIL("SKILL.md protection prose without a standing pin test "
                                "or a no-pin justification (#884/#1045/#1134 shape)")
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py, .claude/agents/planner.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- The spawned session runs under `EPM_WORKFLOW_FIX_SESSION=1` / a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_plan.py, .claude/agents/planner.md
- origin: parked candidate on task #1134 at 2026-07-08T07:32:02Z

Verbatim parked note:

> routed: parked — wf-fix task (tags wf-fix; recursion guard honored despite missing workflow_fix_target Provenance line). TWO candidates surfaced by round-1 critics, logged not auto-filed: (A) [methodology critic] target_file: scripts/verify_plan.py (+.claude/agents/planner.md) — add a check that a plan inserting protection prose into .claude/skills/**/SKILL.md either names a standing pin test or carries a one-line no-pin justification (3rd recurrence of the missing-durability-pin shape: #884, #1045, #1134); confidence medium. (B) [alternatives critic] target_file: scripts/autonomous_session_watch.py — non-gating observer pass (triage-observer shape, #967) flagging doubled-site rounds with disagreeing verdict markers and no epm:review-reconcile / no-show evidence — mechanically detects a #825 repeat and the no-show-misclassification incentive; confidence low-medium. ALSO note: /daily route-2 filing template omits the workflow_fix_target: Provenance line so task_workflow.is_workflow_fix_session()=False on daily-filed wf-fix tasks — same parked disposition, target .claude/skills/daily/SKILL.md.

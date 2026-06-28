---
title: 'workflow-fix: crash-fix circuit-breaker → re-plan on spent fallback ladder'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6575b87f356d
created_at: '2026-06-28T19:44:53Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from #664 deep-dive: /issue crash-fix loop
  reached round 18 with no circuit-breaker forcing re-plan once Option A/B fallback
  ladder is spent (SKILL.md step + workflow.yaml pivot_criteria sub-clause)'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #664 (emitting agent: general-purpose deep-dive diagnostic,
PM-session routed). Task #664's autonomous `/issue --auto` crash-fix loop
reached **round 18** against a pre-registered Option-A→Option-B gate fallback
ladder; the binding "route to planner, not another gauge tweak" instruction
had to be hand-written into the r18 code-review marker because no mechanical
guardrail enforced it.

## Goal

Add a crash-fix circuit-breaker to the `/issue` recovery path that, when N
failures share the same (phase, failure_class, assert-tag) signature OR a
plan-enumerated Option A/B/... fallback set is exhausted for the same gate,
STOPS relaunching and routes to `task.py set-status <N> planning` +
re-invoke `/adversarial-planner` per `pivot_criteria.plan_contradiction_replan`
— and add the matching "enumerated-fallback-exhaustion" sub-clause to
`workflow.yaml § pivot_criteria` so the SKILL.md step and the orchestrator
prose point at one canonical predicate.

## Workflow gap

- **Bug observed:** #664's crash-fix loop reached round 18 against a
  pre-registered Option-A→Option-B gate fallback with no mechanical
  circuit-breaker forcing a re-plan once the gate's enumerated escape options
  were spent; the route-to-planner instruction had to be hand-written into the
  r18 code-review marker.
- **Why it is a workflow gap:** `pivot_criteria.plan_contradiction_replan` and
  the "~3 fundamentally different strategies then pivot" rule exist only as
  orchestrator-judgment PROSE. The `/issue` crash-fix recovery path has no step
  that counts same-failure-class rounds (or detects "the plan's enumerated
  fallback options for this gate are spent") and auto-routes to
  `set-status planning` + `/adversarial-planner`. So an autonomous session can
  burn rounds (and GPU-hours) past the point the project's own pivot rule says
  to re-plan.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```
+ ### Crash-fix circuit-breaker (recovery path)
+ Before any Nth relaunch, scan events.jsonl epm:failure markers:
+ - if >= K (default 4) failures share the same (phase, failure_class, assert-tag) signature, OR
+ - if a plan-enumerated fallback set (e.g. §11 "Option A -> Option B") has been tried to exhaustion and the SAME gate still trips,
+ then DO NOT relaunch. Run `task.py set-status <N> planning` and re-invoke
+ `/adversarial-planner` with pivot scope naming the contradiction
+ (workflow.yaml § pivot_criteria.plan_contradiction_replan). Autonomous sessions
+ post `Decision: re-plan` and continue; do not park for the user unless the
+ re-plan concludes a Goal-level research call only the user can make.
```

Plus a new explicit sub-clause under `workflow.yaml § pivot_criteria.plan_contradiction_replan`:
the "enumerated-fallback-exhaustion" trigger — "the plan pre-registered a finite
escape ladder (e.g. Option A → Option B) and the ladder is spent while the same
gate still trips" — so both the SKILL.md circuit-breaker and the orchestrator
prose reference one canonical named predicate.

## Scope / surfaces

- Primary targets: `.claude/skills/issue/SKILL.md`, `.claude/workflow.yaml`
- These two changes are COUPLED (the SKILL.md step references the workflow.yaml
  predicate) and MUST land together / stay consistent — that is why they are one
  task, not two.
- Grep before editing: `grep -niE 'pivot_criteria|plan_contradiction_replan|crash.?fix|circuit|fallback|fundamentally different strateg' .claude/skills/issue/SKILL.md .claude/workflow.yaml CLAUDE.md` and reconcile with the existing prose (do not duplicate the generic cap-3 pivot path; this is the narrower enumerated-ladder-exhaustion case).
- Keep CLAUDE.md's "Autonomous mode: a self-defeating PLAN routes to /adversarial-planner re-plan" bullet consistent if you touch it.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on any touched script passes;
  if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with SKILL.md and the rule files.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/skills/issue/SKILL.md, .claude/workflow.yaml
- fingerprint: 6575b87f356d

<!-- workflow-fix-candidate v1 -->
target_file: .claude/skills/issue/SKILL.md
bug_observed: Task #664's autonomous crash-fix loop reached round 18 against a pre-registered Option-A→Option-B gate fallback with no mechanical circuit-breaker forcing a /adversarial-planner re-plan once the gate's enumerated escape options are exhausted; the "route to planner not another tweak" instruction had to be written by hand into the r18 code-review marker.
why_workflow_gap: The pivot_criteria.plan_contradiction_replan rule and the "~3 fundamentally different strategies then pivot" rule exist only as orchestrator-judgment prose; the /issue crash-fix recovery path has no step that counts same-failure-class rounds (or detects "the plan's enumerated fallback options for this gate are spent") and auto-routes to set-status planning + re-invoke /adversarial-planner.
proposed_change: Add a crash-fix circuit-breaker step to the /issue recovery path — when N>=K (e.g. K=4) epm:failure markers share the same failure-class/phase signature OR a plan-enumerated Option A/B/... fallback set is exhausted for the same gate, STOP relaunching and route to task.py set-status <N> planning + /adversarial-planner with explicit pivot scope, per pivot_criteria.plan_contradiction_replan.
diff_sketch: |
  + ### Crash-fix circuit-breaker (recovery path)
  + Before any Nth relaunch, scan events.jsonl epm:failure markers:
  + - if >= K (default 4) failures share the same (phase, failure_class, assert-tag) signature, OR
  + - if a plan-enumerated fallback set (e.g. §11 "Option A -> Option B") has been tried to exhaustion and the SAME gate still trips,
  + then DO NOT relaunch. Run `task.py set-status <N> planning` and re-invoke
  + `/adversarial-planner` with pivot scope naming the contradiction
  + (workflow.yaml § pivot_criteria.plan_contradiction_replan). Autonomous sessions
  + post `Decision: re-plan` and continue; do not park for the user unless the
  + re-plan concludes a Goal-level research call only the user can make.
confidence: medium
related_task: #664
<!-- /workflow-fix-candidate -->

Coupled prose follow-up (folded into this task per the coupling note above):
workflow.yaml § pivot_criteria.plan_contradiction_replan should grow an explicit
"enumerated-fallback-exhaustion" sub-clause so the SKILL.md circuit-breaker and the
orchestrator prose point at one canonical predicate.

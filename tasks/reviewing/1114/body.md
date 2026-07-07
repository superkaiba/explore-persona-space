---
title: 'workflow-fix: verify_plan WARN when fence omits §7 conditional phase'
kind: infra
tags:
- wf-fix
- wf-fix-fp:eb9f08b11741
created_at: '2026-07-07T18:12:33Z'
has_clean_result: false
origin_prompt: 'Methodology critic round-1 prose follow-up on #1112: ''deliberate
  fence declared + §7 extension gate present => fence-reconcile must cost the conditional''
  — WARN-level verify_plan.py heuristic, c26 fence family (#599/#833/#1112 pattern)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1112 (emitting agent: critic, Methodology lens, round 1).

## Goal

Add a WARN-level `verify_plan.py` heuristic (c26 fence family): when a plan
declares a deliberate `--max-run-duration` fence AND its §7 (gates/extensions)
contains an extension/resume gate, WARN unless the fence-reconcile sentence
references that conditional phase's wall cost.

## Workflow gap

- **Bug observed:** #1112 v2's 48h fence was sized off base phases only,
  omitting the plan's own §7 G1 dose-extension (which runs on the same
  provision); at worst-case scaling the joint tail reached ~48-50h against a
  48h fence with zero margin. Only the Claude methodology critic caught it.
  Same pattern in #599 (a pre-registered extension probe hard-deleted by the
  fence) and #833.
- **Why it is a workflow gap:** `plan-compute-sizing.md` § fence clause
  already requires "base phases PLUS every conditional / extension phase that
  could run on the same provision", but `verify_plan.py` has no mechanical
  check for it — the c26 family checks cross-GPU basis scaling only, so the
  fence-omits-conditional class relies entirely on a critic noticing.
- **Confidence (emitter):** medium-high (mechanizable: yes, per the critic).

## Proposed change (candidate diff sketch — refine in planning)

```
+ In scripts/verify_plan.py, c26 fence family:
+   - detect a deliberate fence declaration (regex: --max-run-duration\s+\S+
+     or "fence" near a duration in §9)
+   - detect a §7 conditional/extension gate (regex: extension|resume|re-ladder|
+     retrain ... "same provision" / gate labels like G\d)
+   - WARN unless the fence-reconcile sentence (the §9 fence arithmetic line)
+     mentions the gate label or an extension cost term
```

## Scope / surfaces

- Primary target: `scripts/verify_plan.py`
- Grep the workflow surface for the pattern before editing
  (`grep -rln 'max-run-duration' scripts/verify_plan.py .claude/rules/plan-compute-sizing.md`)
  and keep the rule text + verifier in sync; list hits in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- WARN-level only (never FAIL — the heuristic is textual; false positives must
  not block a plan persist).
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  tests for the new check added to the verify_plan test file.
- This session runs under the workflow-fix recursion guard once spawned — it
  MUST NOT auto-route its own subagents' workflow-fix candidates.

## Provenance

- workflow_fix_target: scripts/verify_plan.py
- fingerprint: eb9f08b11741

Surfaced prose (verbatim, from the #1112 round-1 Methodology critic verdict):
"Follow-up (orchestrator may consider): the Must-Fix's check — 'deliberate
fence declared + §7 extension gate present ⇒ fence-reconcile must cost the
conditional' — is a concrete, recurring pattern (#599, #833, now #1112 v2)
suitable as a WARN-level `verify_plan.py` heuristic in the c26 fence family."

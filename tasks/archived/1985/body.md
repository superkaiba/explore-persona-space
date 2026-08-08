---
title: 'workflow-fix: pre_reg audit synonym classes for plan-verdict framing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:5d50c3933dd1
created_at: '2026-08-02T02:41:39Z'
has_clean_result: false
origin_prompt: 'clean-result-critic candidate block on #1902 (round 1, 2026-08-02):
  pre_reg scan synonym-escaped a third time by ''planned verdict'' phrasing'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate raised on task #1902 (emitting agent: clean-result-critic).

## Goal

Extend audit_clean_results_body_discipline.py's pre_reg pattern family with synonym classes for plan-verdict framing.

## Workflow gap

- **Bug observed:** #1902's body shipped "the planned verdict is Confirmed" / "The headline persistence verdict still confirms" — the banned pre-reg verdict-lattice construct with "planned" substituted for "registered" — and the audit's pre_reg scan (keyed on the token "registered") passed it clean.
- **Why it is a workflow gap:** the pre_reg pattern has now been synonym-escaped a third time (#1419's bare "registered <noun>", #1586's "registered layer", now "planned verdict"); each escape costs a clean-result-critic round to catch by hand.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c "registered" scripts/audit_clean_results_body_discipline.py` → 59 hits (the registered-keyed family exists in-target); `grep -nE "planned|pre-?specified" scripts/audit_clean_results_body_discipline.py` → 0 hits (the synonym class is absent — the gap is real) (2026-08-02)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # #1902 escape: 'planned/pre-specified/pre-committed' + lattice noun, and
+ # bare verdict-outcome announcements, evade the 'registered'-keyed scan.
+ r"\b(?:planned|pre-?specified|pre-?committed)\s+(?:verdict|lattice)\b",
+ r"\bverdict\s+(?:is|was|still)\s+(?:confirm|Confirmed|Falsified|Inconclusive)",
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 5d50c3933dd1

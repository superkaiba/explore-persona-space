---
title: 'workflow-fix: WARN-tier plan-conditions coverage check (silent condition drop)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:227f48f2d53f
created_at: '2026-07-29T10:16:07Z'
has_clean_result: false
origin_prompt: 'clean-result-critic r1 on #1774, prose follow-up: plan-§5 conditions
  vs body coverage is LM-only; cell_pre_own drop passed every mechanical check'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up surfaced by the
clean-result-critic on task #1774 (round 1, Lens 13 finding: a silent scope drop that
passed every mechanical check).

## Goal

Add a WARN-tier plan-conditions coverage check to verify_task_body.py that parses the
plan §5 conditions-table config-slug column and WARNs when a slug appears neither in
the body nor in an explicit skip/descope record.

## Workflow gap

- **Bug observed:** Task #1774's plan-committed pretrained-reads robustness condition
  (`cell_pre_own`, plan §5 conditions table + §4 substrate) was silently dropped —
  no artifact, no descope marker, aggregate `.skipped` empty, zero body mentions —
  and passed every mechanical check (verify_task_body.py 61 checks +
  audit_clean_results_body_discipline.py); only the LM critic's plan read caught it.
- **Why it is a workflow gap:** Lens 13's silent-drop check (plan §5 conditions vs
  body mentions) is currently LM-only; the planned-vs-actual mechanical surface
  (check 11b) keys on conditions the body NAMES, so a condition dropped from the body
  entirely is invisible to it. A slug-level WARN closes the gap cheaply.
- **Confidence (emitter):** medium (false-positive risk from paraphrased plain-English
  names — hence WARN tier + slug-level matching, per the emitting critic's own caveat)
- verified-at-filing: `grep -cn 'conditions table\|§5\|plan_conditions' scripts/verify_task_body.py` → 0 hits (no plan-§5-conditions coverage check exists in the target — absence claim, 0-hit in-target result IS the evidence); incident cross-check: `grep -c 'pre_own' tasks/reviewing/1774/body.md` → 0 while plan §5 names `cell_pre_own` (2026-07-29)

## Proposed change (candidate diff sketch — refine in planning)

```
+ # Check N (WARN): plan §5 conditions coverage — parse the plan conditions
+ # table's config-slug column (plans/plan.md, when resolvable); for each slug,
+ # WARN unless it appears in the body OR in an explicit skip/descope phrase
+ # ("not run", "descoped", "named deviation") within +-1 sentence of the slug
+ # or its plain-English row name. WARN tier only — plain-English paraphrase
+ # makes a FAIL tier too false-positive-prone.
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Consider whether the check belongs beside check 11b (planned-vs-actual) or as a new
  numbered check; the plan path resolves via the task folder (`plans/plan.md` symlink).
  Grep the workflow surface for prior art
  (`grep -rn 'planned-vs-actual\|check 11b' scripts/ .claude/rules/`) and keep
  clean-result-critic Lens 13's prose in sync if the mechanical tier lands.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes;
  WARN tier only (a FAIL tier is explicitly out of scope for this filing).
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own
  subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 227f48f2d53f

Verbatim surfaced prose (clean-result-critic r1 on #1774): "Lens 13's silent-drop
check (plan §5 conditions vs body mentions) is currently LM-only. A WARN-tier check in
`scripts/verify_task_body.py` (or `verify_plan.py`-adjacent tooling) could parse the
plan §5 conditions table's config-slug column and WARN when a slug (or its
plain-English name) appears neither in the body nor in an aggregate
`skipped`/deviation record — slug-level matching with WARN severity to bound the
paraphrase false-positive risk. Evidence: this round's `cell_pre_own` drop passed
every mechanical check and was only caught by the plan read. mechanizable: yes (with
the stated false-positive caveat)."

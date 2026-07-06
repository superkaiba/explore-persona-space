---
title: 'workflow-fix: audit pre-reg regex misses ''as registered'' phrasing'
kind: infra
tags:
- wf-fix
- wf-fix-fp:d4eabc008951
created_at: '2026-07-02T21:42:32Z'
has_clean_result: false
origin_prompt: 'clean-result-critic prose follow-up on #763: add ''as registered''
  to audit_clean_results_body_discipline.py''s pre-registration-mention regex'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a prose follow-up raised on task #763 (emitting agent: clean-result-critic, 9a-bis round 1).

## Goal

Add the "as registered" phrasing family to `scripts/audit_clean_results_body_discipline.py`'s pre-registration-mention pattern so the mechanical audit catches it instead of relying on the LM critic.

## Workflow gap

- **Bug observed:** a #763 clean-result body carried "As registered, SUCCESS was not met" twice; the audit script's pre-registration-mention regex did not flag it — the Claude clean-result-critic caught it manually and attached the rewrite as a procedural fix.
- **Why it is a workflow gap:** the audit is the mechanical pre-pass both critics incorporate; a phrasing family it misses recurs silently whenever the critics' attention is elsewhere.
- **Confidence (emitter):** low (the critic tagged this "a low-confidence workflow follow-up" — filed anyway per the 2026-06-11 standing directive; the spawned session's planner makes the deliberate call).

## Proposed change (candidate diff sketch — refine in planning)

```
scripts/audit_clean_results_body_discipline.py, pre-registration-mention pattern:
- r"pre-?registr|pre-?registered"
+ r"pre-?registr|pre-?registered|\bas registered\b"
(exact existing pattern to be read from the script; add the "as registered" alternation
plus a test row in the script's test file)
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep for sibling pattern definitions + the pinning test file before editing.

## Constraints / invariants

- Workflow-surface only; ruff passes; the audit's existing tests stay green; add a regression row for the new phrasing.

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: d4eabc008951

Surfaced prose (verbatim): "a low-confidence workflow follow-up suggests adding `as registered` to the audit's pre-reg regex."

---
title: 'workflow-fix: byte-equal synonym in bit_byte_identical audit regex'
kind: infra
tags:
- wf-fix
- wf-fix-fp:97443bba4ec6
created_at: '2026-07-16T12:53:32Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1005 r1 mechanizable item: extend the bit_byte_identical
  regex to catch the `byte-equal`/`byte equal` synonym family (byte[- ](identical|equal))'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1005 (emitting agent: clean-result-critic, round-1 critique
2026-07-16, mechanizable-yes fix-list item).

## Goal

extend the bit_byte_identical regex to catch the `byte-equal`/`byte equal` synonym family (byte[- ](identical|equal))

## Workflow gap

- **Bug observed:** issue 1005 body carried 'inherited byte-equal (sha-asserted)' which the byte-identical-only regex missed; the clean-result-critic caught it manually
- **Why it is a workflow gap:** the clean-result mechanical verifiers exist precisely to catch this class before an LM critic round has to; the miss cost a critic-round finding on #1005.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n byte scripts/audit_clean_results_body_discipline.py -> bit_byte_identical regex at lines 245-251 (2026-07-16); covers identical-forms only, no -equal synonym`

## Proposed change (candidate diff sketch — refine in planning)

```
- r'byte[- ]identical' family
+ r'(bit|byte)[- ](identical|equal)\b' family (keep existing bit forms)
```

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; keep SPEC.md consistent if the check semantics are documented there.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Grandfathered v3/v2/legacy bodies must not be newly hard-FAILed (WARN-grade where applicable).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: 97443bba4ec6

(surfaced prose: clean-result-critic #1005 round-1 minimal-necessary-fix list / procedural-fixes section, mechanizable-yes items)

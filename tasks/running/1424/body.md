---
title: 'workflow-fix: caption italic-lead-claim check (check 5 family)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:71428d13e30a
created_at: '2026-07-16T12:53:39Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1005 r1 mechanizable item: add the SPEC italic
  one-sentence lead-claim component to the figure-caption check (check 5 family):
  blockquote captions must open `> **Figure.** *lead claim.*`'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1005 (emitting agent: clean-result-critic, round-1 critique
2026-07-16, mechanizable-yes fix-list item).

## Goal

add the SPEC italic one-sentence lead-claim component to the figure-caption check (check 5 family): blockquote captions must open `> **Figure.** *lead claim.*`

## Workflow gap

- **Bug observed:** issue 1005 round-1 clean-result critique found all 7 captions missing the italic lead claim; check 5 is vacuously satisfied and tests no caption-form component
- **Why it is a workflow gap:** the clean-result mechanical verifiers exist precisely to catch this class before an LM critic round has to; the miss cost a critic-round finding on #1005.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -in caption scripts/verify_task_body.py -> check 5 documented 'vacuously satisfied' at line 85 (2026-07-16); grep -c italic -> 1 (no caption-form italic-lead test)`

## Proposed change (candidate diff sketch — refine in planning)

```
+ WARN-grade check: each result's blockquote caption matches r'^> \*\*Figure\.?\*\*\s+\*[^*]+\*' (italic lead claim present); WARN not FAIL (grandfathered bodies)
```

## Scope / surfaces

- Primary target: `scripts/verify_task_body.py`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes; keep SPEC.md consistent if the check semantics are documented there.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).
- Grandfathered v3/v2/legacy bodies must not be newly hard-FAILed (WARN-grade where applicable).

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: 71428d13e30a

(surfaced prose: clean-result-critic #1005 round-1 minimal-necessary-fix list / procedural-fixes section, mechanizable-yes items)

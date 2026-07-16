---
title: 'workflow-fix: Sample-slot disclosure-count check in verify_task_body'
kind: infra
tags:
- wf-fix
- wf-fix-fp:a0ac0e5068f9
created_at: '2026-07-16T12:53:15Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1005 r1 mechanizable item: add a Sample-slot
  disclosure-count check: parse the `Disclosure: <N> of <M>` integer and compare N
  to the number of example blocks actually shown in the Sample slot'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1005 (emitting agent: clean-result-critic, round-1 critique
2026-07-16, mechanizable-yes fix-list item).

## Goal

add a Sample-slot disclosure-count check: parse the `Disclosure: <N> of <M>` integer and compare N to the number of example blocks actually shown in the Sample slot

## Workflow gap

- **Bug observed:** issue 1005 body claimed 'Disclosure: 8 of 2,400' while only 6 example blocks followed; no mechanical check exists (grep Disclosure -> 0 hits)
- **Why it is a workflow gap:** the clean-result mechanical verifiers exist precisely to catch this class before an LM critic round has to; the miss cost a critic-round finding on #1005.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n Disclosure scripts/verify_task_body.py -> 0 hits (2026-07-16) — absence-of-check claim; 0-hit in-target result IS the evidence`

## Proposed change (candidate diff sketch — refine in planning)

```
+ in the Sample-slot pass: m = re.search(r'Disclosure:\s*(\d+)\s+of', slot_text)
+ if m and int(m.group(1)) != count_example_blocks(slot_text): FAIL (disclosure count mismatch)
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
- fingerprint: a0ac0e5068f9

(surfaced prose: clean-result-critic #1005 round-1 minimal-necessary-fix list / procedural-fixes section, mechanizable-yes items)

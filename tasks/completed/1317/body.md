---
title: 'daily-fix: code-reviewer Gate-scope line check (#1305)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:164678bd21d2
- daily-auto-filed
created_at: '2026-07-15T06:51:27Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): The #1305 gate-scope duty
  (implementer pre-report selector enumeration + pin-sweep) has no reviewer-side compliance
  check — a context-pressured implementer can skip or rubber-stamp the report''s Gate-scope
  line.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 Step C parked-candidate routing pass from a FORMAL candidate block parked under the recursion guard on task #1305 (emitting agent: Phase-2 alternatives critic, compliance-enforcement-point concern; park ts 2026-07-14T07:17:13Z).

## Goal

One bullet in code-reviewer.md instructing the reviewer to check the report's Gate-scope line for presence + diff-consistency, treating a NOT-RUN pin-hit as presumptively blocker-adjacent.

## Workflow gap

- **Bug observed:** The #1305 gate-scope duty (implementer pre-report selector enumeration + pin-sweep) has no reviewer-side compliance check — a context-pressured implementer can skip or rubber-stamp the report's Gate-scope line.
- **Why it is a workflow gap:** prose duties on the implementer are advisory; the code-reviewer spec never tells the reviewer to verify the Gate-scope check line exists and is consistent with the diff.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c "Gate-scope" .claude/agents/code-reviewer.md` -> 0 hits (absence-of-guard claim; the 0-hit in-target result IS the evidence) (2026-07-15). Retraction re-check on #1305 events after the park ts: none (task completed + merged with no reviewer-side check added).

## Proposed change (candidate diff sketch — refine in planning)

```
+ - **Gate-scope line check (#1305):** the implementation report's
+   `Gate-scope check` line must exist and be consistent with the diff
+   (every changed literal appears in the pin-sweep fragments; a
+   pin-sweep HIT marked NOT-RUN is presumptively blocker-adjacent).
```

## Scope / surfaces

- Primary target: `.claude/agents/code-reviewer.md`

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py --check-asks` passes.
- Recursion guard applies (workflow_fix_target Provenance line below).

## Provenance

- workflow_fix_target: .claude/agents/code-reviewer.md
- fingerprint: 164678bd21d2

Verbatim parked candidate: formal `<!-- workflow-fix-candidate v1 -->` block on #1305 events (2026-07-14T07:17:13Z), fingerprint 164678bd21d2, related_task #1305.

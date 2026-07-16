---
title: 'workflow-fix: check-30 trailing-parenthetical file-count shape'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1a0ac48a7ddb
created_at: '2026-07-16T12:53:25Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1005 r1 mechanizable item: extend check 30 (check_hf_file_count_claims)
  adjacency window to the `[label](tree-url) (N files)` trailing-parenthetical shape'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a workflow-fix candidate
raised on task #1005 (emitting agent: clean-result-critic, round-1 critique
2026-07-16, mechanizable-yes fix-list item).

## Goal

extend check 30 (check_hf_file_count_claims) adjacency window to the `[label](tree-url) (N files)` trailing-parenthetical shape

## Workflow gap

- **Bug observed:** issue 1005 footer's `[summary store](url) (51 files)` and `[fit results](url) (15 files)` understated the pinned Hub tree (52/17) but check 30 reported 'no file-count claims adjacent'
- **Why it is a workflow gap:** the clean-result mechanical verifiers exist precisely to catch this class before an LM critic round has to; the miss cost a critic-round finding on #1005.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'check_hf_file_count_claims' scripts/verify_task_body.py -> check 30 exists at lines 546/8359+ (2026-07-16); the trailing-parenthetical shape is absent from its adjacency parse`

## Proposed change (candidate diff sketch — refine in planning)

```
+ in check 30's claim scan: also match r'\[[^\]]+\]\([^)]*tree[^)]*\)\s*\((\d+)\s+files' and reconcile N against the linked prefix's Hub tree count
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
- fingerprint: 1a0ac48a7ddb

(surfaced prose: clean-result-critic #1005 round-1 minimal-necessary-fix list / procedural-fixes section, mechanizable-yes items)

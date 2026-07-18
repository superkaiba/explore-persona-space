---
title: 'workflow-fix: verify_task_body Context-row provenance vs epm:followup-scope
  cross-check'
kind: infra
tags:
- wf-fix
- wf-fix-fp:fb694bc00379
created_at: '2026-07-18T18:47:23Z'
has_clean_result: false
origin_prompt: 'clean-result-critic #1426 fold r1 Lens 5b mechanizable prose (Context
  row free-analysis vs marker proposer-9b-cheap)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from a clean-result-critic prose follow-up on task #1426 (fold round r1, Lens 5b, mechanizable: yes).

## Goal

Add a `verify_task_body.py` check that cross-checks the body `**Context:**` row's follow-up clause `cost_class`/`source` tokens against the latest `epm:followup-scope v1` note fields for the same followup_label; FAIL on contradiction.

## Workflow gap

- **Bug observed:** the #1426 fold's Context row said "proposer cost_class free-analysis" while the round's `epm:followup-scope` marker recorded `source: proposer-9b-cheap`, `est_gpu_hours: 14` (GPU cheap band) — a provenance contradiction only the adversarial critic caught.
- **Why it is a workflow gap:** the Context row is the run-provenance record the clean-result ships forward (SPEC.md § footer, check 17); the followup-scope marker is machine-readable ground truth in events.jsonl, so the contradiction is mechanically checkable but no check exists.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n 'followup-scope' scripts/verify_task_body.py` → 4 hits (lines 4261, 4389, 12161, 12236), all in OTHER checks' contexts (origin-prompt informational note, provenance-source enumeration, armed-label dispatchability helpers); none compares Context-row cost_class/source tokens to the marker fields (2026-07-18).

## Proposed change (candidate diff sketch — refine in planning)

```
+ new check: when the **Context:** row names a follow-up round (followup_label
+   or source/cost_class tokens), parse the latest matching epm:followup-scope
+   note via task_workflow.parse_followup_note_field; FAIL when the body's
+   source/cost_class contradicts the marker's (missing tokens = skip, not FAIL)
```

## Provenance

- workflow_fix_target: scripts/verify_task_body.py
- fingerprint: fb694bc00379

Surfaced prose (clean-result-critic #1426 fold r1): "mechanizable: yes — cross-check the body **Context:** follow-up clause's cost_class/source tokens against the latest epm:followup-scope v1 note fields; FAIL on contradiction."

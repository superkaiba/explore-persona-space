---
title: 'daily-fix: raise step9c pristine-oracle timeout default'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6944e2ce3aa3
- daily-auto-filed
created_at: '2026-07-08T06:58:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): #1098 (4c1c98c3, 13:18Z):
  the pristine single-file oracle run of test_workflow_lint.py needs ~13min but --pristine-timeout-s
  defaults to 600s; the first compare timed out and was re-run at 1200s (~10-20min
  wasted). Repeats for any gate selection including that file.'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

raise the pristine-timeout default or derive it per-file from the gate's own measured runtime (e.g. 2x the file's gate wall-time, floor 600s)

## Workflow gap

- **Bug observed:** #1098 (4c1c98c3, 13:18Z): the pristine single-file oracle run of test_workflow_lint.py needs ~13min but --pristine-timeout-s defaults to 600s; the first compare timed out and was re-run at 1200s (~10-20min wasted). Repeats for any gate selection including that file.
- **Why it is a workflow gap:** A fixed default below a known gate file's runtime guarantees a wasted timeout + rerun on every Step 9c selection including that file.

## Proposed change

Prefer the measured-runtime derivation (2x gate wall-time, floor 600s); keep the flag as an override. Default currently at scripts/step9c_baseline.py ~line 1353 (600.0).

## Scope / surfaces

- Primary target: `scripts/step9c_baseline.py`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: 4c1c98c3 (#1098) 13:18Z.

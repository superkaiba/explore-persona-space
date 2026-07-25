---
title: 'daily-fix: reconcile test_issue811_pre_user with extractor'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-25T06:51:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): tests/test_issue811_pre_user.py
  fails 5 of 21 on pristine main because scripts/issue667_extract.py no longer exposes
  PRE_USER_LAYER_ARMS or derive_alllayer_arms (test-vs-module API drift); every touched-scope
  Step 9c gate that pulls the file eats 5 red rows classified only by baseline-compare'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 858e5986, task #1656 — implementer prose follow-up parked under the recursion guard + out-of-scope note).

## Goal

`tests/test_issue811_pre_user.py` green on pristine main (currently 5/21 red from experiment-script API drift).

## Workflow gap

- **Bug observed:** #1656's Step 9c gate hit `AttributeError: module 'issue667_extract' has no attribute 'PRE_USER_LAYER_ARMS'` / `'derive_alllayer_arms'` — 5 failures in one gate run, classified pre-existing by baseline-compare (implementer reproduced 5/21 red on pristine main).
- **Why filed:** experiment-code test drift, not a workflow-surface gap (hence no wf-fix tags) — but every touched-scope gate pulling this file pays the classification cost until reconciled.
- **Confidence (emitter):** high.
- verified-at-filing: `grep -n 'PRE_USER_LAYER_ARMS\|derive_alllayer_arms' tests/test_issue811_pre_user.py scripts/issue667_extract.py` → test references at tests/...:15/:105/:118/:194/:198/:208; ZERO hits in scripts/issue667_extract.py (absence of the symbols confirmed, 2026-07-25).

## Proposed change (candidate diff sketch — refine in planning)

Either restore/re-export the two symbols in `scripts/issue667_extract.py` (if renamed, alias) or update the test to the current API; decide against the #811 line's intent (the test pins bit-exact re-derivation — check the owning issue's history before weakening).

## Scope / surfaces

- `tests/test_issue811_pre_user.py`, `scripts/issue667_extract.py`

## Constraints / invariants

- Do not weaken the bit-exactness pins without a recorded reason; ruff clean.

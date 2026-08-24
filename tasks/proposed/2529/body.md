---
title: verify_task_body per-unit-evidence WARN vocabulary misses per-character grain
kind: infra
tags: []
created_at: '2026-08-24T09:01:19Z'
has_clean_result: false
origin_prompt: 'workflow-fix candidate surfaced in prose by the #2479 interpretation-critic
  round-3 report'
workflow: v1
---
# verify_task_body per-unit-evidence WARN vocabulary misses "per-character" grains

## Goal

Extend `scripts/verify_task_body.py`'s per-unit-evidence WARN vocabulary so bodies whose natural unit is the character (or another established grain synonym) are not false-positive-WARNed. The check currently recognizes a fixed set of per-unit phrase forms and misses "per-character", so a body that fully satisfies the underlying-data rule at character grain still draws the WARN.

## Provenance

Surfaced in prose by the #2479 interpretation-critic round-3 report (2026-08-24): #2479's clean-result reports every aggregate with labeled per-character plots and per-character JSON fields, yet the verifier emitted the per-unit-evidence WARN because its vocabulary lacks the "per-character" form.

## Design sketch

Locate the per-unit-evidence WARN's phrase list/regex in `verify_task_body.py` and generalize it: add "per-character" (and audit the list for the project's other recurring grains — per-cell, per-seed, per-behavior, per-rung, per-conversation — adding any that appear in promoted v4 bodies but not in the vocabulary). Keep the check's intent unchanged: WARN only when NO per-unit evidence form is present.

## Acceptance criteria

1. A v4 body whose only per-unit evidence phrasing is "per-character" no longer draws the per-unit-evidence WARN.
2. A body with no per-unit evidence at all still WARNs (no weakening).
3. Tests in `tests/test_verify_task_body.py` cover both cases.
4. Run the verifier against #2479's current body as a live fixture check (WARN disappears; other verdict lines unchanged).

---
title: verify_task_body per-unit-evidence scanner misses overlay-points phrasings
  (false WARN on declared pairs)
kind: infra
tags: []
created_at: '2026-08-25T12:29:56Z'
has_clean_result: false
parent_id: 2378
workflow: v1
---
## Goal
Extend the per-unit-evidence scanner vocabulary in scripts/verify_task_body.py so declared per-unit views phrased as overlay points stop firing false WARNs.

## Context
#2378 CRC r4 (Claude, PASS): the new lenmatch result section declares its per-unit view as '5 global-family folds as open points' and the rendered figure shows per-fold open points on both legs, but the per-unit-evidence check's phrase list missed it — a false WARN on an overlay-style declared aggregate+per-unit pair. Add 'open points' / 'fold points' (and similar overlay-marker phrasings) to the check's per-unit vocabulary; add a fixture test reproducing the #2378 lenmatch section shape.

## Provenance
Surfaced as a workflow-surface prose follow-up in #2378 clean-result-critique v4 (Claude, Round 4 PASS).

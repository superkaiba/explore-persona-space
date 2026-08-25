---
title: verify_task_body figure opaque-code scan fires on sidecar provenance keys (scope
  to rendered surfaces)
kind: infra
tags: []
created_at: '2026-08-25T12:32:07Z'
has_clean_result: false
parent_id: 2378
workflow: v1
---
## Goal
Scope the figure opaque-code scan in scripts/verify_task_body.py to RENDERED text surfaces only (title/labels/legend/annotations/ticks), excluding sidecar provenance keys.

## Context
#2378 CRC r4 (Claude, PASS): the opaque-code WARN fired on 'chat_user_real' inside the meta.json sidecar's series_annotation provenance key — a location the check's own remedy text sanctions for slugs; the rendered PNG tick label is plain-English ('User (real)'). Restrict the scanned keys to rendered-text surfaces so slugs in provenance fields stop tripping it; add a fixture test with a slug-bearing provenance key + clean rendered text.

## Provenance
Surfaced as a workflow-surface prose follow-up in #2378 clean-result-critique v4 (Claude, Round 4 PASS).

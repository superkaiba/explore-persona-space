---
title: 'verify_task_body check 54: shared pinned-JSON loader lacks fail-soft on non-UTF8
  git blobs (pre-existing; surfaced by #2635 r2)'
kind: infra
tags: []
created_at: '2026-08-27T22:07:26Z'
has_clean_result: false
origin_prompt: '#2635 r2 implementer (d) flag: check 54''s own pre-existing non-UTF8
  exposure in the shared loader'
workflow: v1
---
## Goal

Harden check 54 (`check_artifact_content_claims`) in `scripts/verify_task_body.py` against non-UTF8 committed blobs (and other decode-time exceptions) in its shared pinned-JSON loader path, mirroring the fail-soft boundary #2635 round 2 added for check 62: a decode failure degrades the artifact to an unverifiable/WARN disposition with the reason recorded, never an uncaught exception out of the check (the verifier runs fleet-wide at every draft/promotion gate).

## Provenance

Surfaced by the #2635 round-2 implementer (2026-08-27, epm:results v2 section (d)): check 62's fail-soft fix wrapped its OWN load/view boundary, but the shared loader path as consumed by check 54 retains the pre-existing exposure — a non-UTF8 blob at a pinned sha can raise out of check 54. Pre-existing on trunk; deliberately out of #2635's round scope.

## Sketch

- Wrap check 54's loader consumption (or push the fail-soft into the shared `_git_json_text_at_sha` / `_working_copy_json_text` helpers with a per-caller disposition) so decode/OS/recursion exceptions degrade soft.
- Tests: non-UTF8 committed fixture through check 54 (no crash, WARN-class disposition); regression: check 62's #2635 tests stay green.

## Acceptance

- A body whose check-54-linked pinned artifact is a non-UTF8 blob produces a WARN-class verdict from check 54, never a traceback.
- Full tests/test_verify_task_body.py green.

---
title: 'workflow-fix: no-flags workflow_lint red on main — issue823_ladder_ext_gen.py
  trips --check-shared-tmp-name'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T23:42:37Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate emitted by the #2552 unit-1 experiment-implementer
  (2026-08-24): fleet-wide no-flags lint red, #1388 shape.'
workflow: v1
---
# workflow-fix: no-flags workflow_lint red on main — issue823_ladder_ext_gen.py:1330 trips --check-shared-tmp-name

## Goal

Restore the fleet-wide no-flags `workflow_lint.py` run (the Step 9c gate leg) to green on main.

## Symptom (verified 2026-08-24)

`scripts/issue823_ladder_ext_gen.py:1330` trips `--check-shared-tmp-name` (#2336) on the MAIN checkout and is on neither the batch-0 allowlist nor waived — the no-flags run FAILs (1 error) for every session's Step 9c gate (the #1388 fleet-gate shape). Verified by running the scoped check at the repo root 2026-08-24. Surfaced by the #2552 unit-1 implementer's payload lint pass (its own payload was zero-attributed).

## Provenance

workflow_fix_target: scripts/workflow_lint.py gate surface (offending file scripts/issue823_ladder_ext_gen.py:1330)
Fingerprint: shared-tmp-name-red-issue823-l1330.

## Fix sketch (one line, two options)

Either swap the line-1330 writer to `explore_persona_space.atomic_io.savez_atomic` (the #2336 canonical fix, preferred), or add `# SHARED_TMP_EXEMPT: <reason>` above line 1330.

## Acceptance criteria

1. No-flags `uv run python scripts/workflow_lint.py` on main: the `--check-shared-tmp-name` leg reports zero errors for scripts/issue823_ladder_ext_gen.py.
2. If the savez_atomic swap is taken: the touched write path round-trips (load the written npz once in a smoke).

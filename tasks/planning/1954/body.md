---
title: 'daily-fix: fix red main: test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py]
  — Convert the __main__ guard of scripts/is'
kind: infra
tags:
- urgent-main-red
created_at: '2026-08-01T01:03:49Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1689_fits_mirror.py\n\
  bug_observed: scripts/issue1689_fits_mirror.py (landed on main at 8c9140aa0a) ends\
  \ with sys.exit(main()) where tests/test_issue1689_os_exit_shutdown_bypass.py pins\
  \ the canonical flush-then-os._exit entrypoint form — 3 parametrized test nodes\
  \ for [issue1689_fits_mirror.py] are currently red on origin/main (confirmed by\
  \ the #1911 implementer's local run 2026-07-31; file byte-identical to origin/main).\
  \ The sibling user_slot entrypoints were already conformed by issue-1903 (34adf410e9,\
  \ \"fix red main\"); fits_mirror was missed.\nwhy_workflow_gap: Pre-existing red\
  \ in the pinned os._exit contract tests makes every session's Step 9c gate re-classify\
  \ the same failure (the #1681 fleet-wide per-hour cost class).\nproposed_change:\
  \ Convert the __main__ guard of scripts/issue1689_fits_mirror.py to the pinned flush-then-os._exit\
  \ form (rc type-coerced), matching the 34adf410e9 fix applied to the user_slot entrypoints.\n\
  diff_sketch: |\n  __main__ guard: capture rc = main(); flush stdout/stderr; os._exit(rc\
  \ if isinstance(rc, int) else 0) — the exact shape 34adf410e9 applied to the sibling\
  \ scripts.\nconfidence: high\nrelated_task: #1911\nurgency: main-red\nfailing_test:\
  \ tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py]\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1911. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py]` red on origin/main: Convert the __main__ guard of scripts/issue1689_fits_mirror.py to the pinned flush-then-os._exit form (rc type-coerced), matching the 34adf410e9 fix applied to the user_slot entrypoints.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** scripts/issue1689_fits_mirror.py (landed on main at 8c9140aa0a) ends with sys.exit(main()) where tests/test_issue1689_os_exit_shutdown_bypass.py pins the canonical flush-then-os._exit entrypoint form — 3 parametrized test nodes for [issue1689_fits_mirror.py] are currently red on origin/main (confirmed by the #1911 implementer's local run 2026-07-31; file byte-identical to origin/main). The sibling user_slot entrypoints were already conformed by issue-1903 (34adf410e9, "fix red main"); fits_mirror was missed.
- **Failing node (router-verified):** `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py]`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py] -q` -> rc=1 at main @ 9365e2f74b (2026-08-01T01:03:41Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1689_fits_mirror.py`
- Failing node: `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py]`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: 9b7296e983b8
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1689_fits_mirror.py
bug_observed: scripts/issue1689_fits_mirror.py (landed on main at 8c9140aa0a) ends with sys.exit(main()) where tests/test_issue1689_os_exit_shutdown_bypass.py pins the canonical flush-then-os._exit entrypoint form — 3 parametrized test nodes for [issue1689_fits_mirror.py] are currently red on origin/main (confirmed by the #1911 implementer's local run 2026-07-31; file byte-identical to origin/main). The sibling user_slot entrypoints were already conformed by issue-1903 (34adf410e9, "fix red main"); fits_mirror was missed.
why_workflow_gap: Pre-existing red in the pinned os._exit contract tests makes every session's Step 9c gate re-classify the same failure (the #1681 fleet-wide per-hour cost class).
proposed_change: Convert the __main__ guard of scripts/issue1689_fits_mirror.py to the pinned flush-then-os._exit form (rc type-coerced), matching the 34adf410e9 fix applied to the user_slot entrypoints.
diff_sketch: |
  __main__ guard: capture rc = main(); flush stdout/stderr; os._exit(rc if isinstance(rc, int) else 0) — the exact shape 34adf410e9 applied to the sibling scripts.
confidence: high
related_task: #1911
urgency: main-red
failing_test: tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_fits_mirror.py]
wf_fix: false
<!-- /workflow-fix-candidate -->

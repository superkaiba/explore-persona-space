---
title: 'daily-fix: fix red main: test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py]
  — make the four issue1689_user_slot entryp'
kind: infra
tags:
- urgent-main-red
created_at: '2026-07-31T00:13:12Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1689_user_slot_capture.py,\
  \ scripts/issue1689_user_slot_fits.py, scripts/issue1689_user_slot_render.py, scripts/issue1689_user_slot_gen_a1.py\n\
  bug_observed: tests/test_issue1689_os_exit_shutdown_bypass.py is red on pristine\
  \ origin/main — 12 failing params (test_entrypoint_calls_os_exit / test_entrypoint_flushes_before_os_exit\
  \ / test_entrypoint_os_exit_rc_type_coerced x issue1689_user_slot_{capture,fits,render,gen_a1}.py):\
  \ the committed issue1689 user-slot entrypoints do not satisfy the os_exit shutdown-bypass\
  \ contract the test pins\nwhy_workflow_gap: a live red on origin/main taxes every\
  \ intervening session's Step 9c gate with a re-classification round (the #1643/#1713\
  \ fleet-cost class); the fix itself is experiment-code (out of wf-fix scope) but\
  \ the red needs mechanical routing rather than per-session re-discovery\nproposed_change:\
  \ make the four issue1689_user_slot entrypoints satisfy the os_exit contract (call\
  \ os._exit with flush-before and rc type coercion per the test's pins), or update\
  \ the test's param list if the entrypoints were deliberately re-shaped — NOTE a\
  \ live #1689 session (uncommitted shared-root edits on exactly these scripts) may\
  \ be about to land this; the router's red-verification + the spawned session's clarifier\
  \ should check #1689's branch state first\ndiff_sketch: |\n  + per failing param:\
  \ add the flush + os._exit(int(rc)) tail the test pins\n  + (or) tests/test_issue1689_os_exit_shutdown_bypass.py:\
  \ drop/adjust params if the\n  + entrypoint shape legitimately changed on the #1689\
  \ branch\nconfidence: medium\nrelated_task: #1840\nurgency: main-red\nfailing_test:\
  \ tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py]\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1840. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py]` red on origin/main: make the four issue1689_user_slot entrypoints satisfy the os_exit contract (call os._exit with flush-before and rc type coercion per the test's pins), or update the test's param list if the entrypoints were deliberately re-shaped — NOTE a live #1689 session (uncommitted shared-root edits on exactly these scripts) may be about to land this; the router's red-verification + the spawned session's clarifier should check #1689's branch state first

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_issue1689_os_exit_shutdown_bypass.py is red on pristine origin/main — 12 failing params (test_entrypoint_calls_os_exit / test_entrypoint_flushes_before_os_exit / test_entrypoint_os_exit_rc_type_coerced x issue1689_user_slot_{capture,fits,render,gen_a1}.py): the committed issue1689 user-slot entrypoints do not satisfy the os_exit shutdown-bypass contract the test pins
- **Failing node (router-verified):** `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py]`
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py] -q` -> rc=1 at main @ 05ffc4f18f (2026-07-31T00:13:09Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1689_user_slot_capture.py, scripts/issue1689_user_slot_fits.py, scripts/issue1689_user_slot_render.py, scripts/issue1689_user_slot_gen_a1.py`
- Failing node: `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py]`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: e5f17affb1b8
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1689_user_slot_capture.py, scripts/issue1689_user_slot_fits.py, scripts/issue1689_user_slot_render.py, scripts/issue1689_user_slot_gen_a1.py
bug_observed: tests/test_issue1689_os_exit_shutdown_bypass.py is red on pristine origin/main — 12 failing params (test_entrypoint_calls_os_exit / test_entrypoint_flushes_before_os_exit / test_entrypoint_os_exit_rc_type_coerced x issue1689_user_slot_{capture,fits,render,gen_a1}.py): the committed issue1689 user-slot entrypoints do not satisfy the os_exit shutdown-bypass contract the test pins
why_workflow_gap: a live red on origin/main taxes every intervening session's Step 9c gate with a re-classification round (the #1643/#1713 fleet-cost class); the fix itself is experiment-code (out of wf-fix scope) but the red needs mechanical routing rather than per-session re-discovery
proposed_change: make the four issue1689_user_slot entrypoints satisfy the os_exit contract (call os._exit with flush-before and rc type coercion per the test's pins), or update the test's param list if the entrypoints were deliberately re-shaped — NOTE a live #1689 session (uncommitted shared-root edits on exactly these scripts) may be about to land this; the router's red-verification + the spawned session's clarifier should check #1689's branch state first
diff_sketch: |
  + per failing param: add the flush + os._exit(int(rc)) tail the test pins
  + (or) tests/test_issue1689_os_exit_shutdown_bypass.py: drop/adjust params if the
  + entrypoint shape legitimately changed on the #1689 branch
confidence: medium
related_task: #1840
urgency: main-red
failing_test: tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_user_slot_capture.py]
wf_fix: false
<!-- /workflow-fix-candidate -->

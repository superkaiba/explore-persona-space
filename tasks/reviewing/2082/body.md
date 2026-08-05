---
title: 'daily-fix: fix red main: test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py]
  — conform the script''s `__main__` guard to'
kind: infra
tags:
- urgent-main-red
created_at: '2026-08-05T06:43:10Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1689_real_u2_from_1738.py\n\
  bug_observed: tests/test_issue1689_os_exit_shutdown_bypass.py fails 3 nodes on origin/main\
  \ because commit 3a98d64680 landed scripts/issue1689_real_u2_from_1738.py with a\
  \ bare `main()` in its `__main__` guard (no flush + `os._exit` tail) while the invariant\
  \ test auto-enumerates every `issue1689_*` entrypoint.\nwhy_workflow_gap: the os._exit\
  \ shutdown-bypass invariant (gotchas.md #1689 entry) is test-enforced fleet-wide,\
  \ so this live red taxes every intervening session's Step 9c gate re-classification\
  \ until fixed.\nproposed_change: conform the script's `__main__` guard to the invariant\
  \ (flush stdout/stderr, then `os._exit(int(rc))`, matching the sibling issue1689\
  \ entrypoints).\ndiff_sketch: |\n  if __name__ == \"__main__\":\n  -    main()\n\
  \  +    _rc = main() or 0\n  +    sys.stdout.flush()\n  +    sys.stderr.flush()\n\
  \  +    os._exit(int(_rc))\nconfidence: high\nrelated_task: #2079\nurgency: main-red\n\
  failing_test: tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py]\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#2079. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py]` red on origin/main: conform the script's `__main__` guard to the invariant (flush stdout/stderr, then `os._exit(int(rc))`, matching the sibling issue1689 entrypoints).

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_issue1689_os_exit_shutdown_bypass.py fails 3 nodes on origin/main because commit 3a98d64680 landed scripts/issue1689_real_u2_from_1738.py with a bare `main()` in its `__main__` guard (no flush + `os._exit` tail) while the invariant test auto-enumerates every `issue1689_*` entrypoint.
- **Failing node (router-verified):** `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py]`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py] -q` -> rc=1 at main @ cfdf781789 (2026-08-05T06:43:07Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1689_real_u2_from_1738.py`
- Failing node: `tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py]`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: 1d2f84522b02
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1689_real_u2_from_1738.py
bug_observed: tests/test_issue1689_os_exit_shutdown_bypass.py fails 3 nodes on origin/main because commit 3a98d64680 landed scripts/issue1689_real_u2_from_1738.py with a bare `main()` in its `__main__` guard (no flush + `os._exit` tail) while the invariant test auto-enumerates every `issue1689_*` entrypoint.
why_workflow_gap: the os._exit shutdown-bypass invariant (gotchas.md #1689 entry) is test-enforced fleet-wide, so this live red taxes every intervening session's Step 9c gate re-classification until fixed.
proposed_change: conform the script's `__main__` guard to the invariant (flush stdout/stderr, then `os._exit(int(rc))`, matching the sibling issue1689 entrypoints).
diff_sketch: |
  if __name__ == "__main__":
  -    main()
  +    _rc = main() or 0
  +    sys.stdout.flush()
  +    sys.stderr.flush()
  +    os._exit(int(_rc))
confidence: high
related_task: #2079
urgency: main-red
failing_test: tests/test_issue1689_os_exit_shutdown_bypass.py::test_entrypoint_calls_os_exit[issue1689_real_u2_from_1738.py]
wf_fix: false
<!-- /workflow-fix-candidate -->

---
title: 'workflow-fix: main-red — post_marker_echo test stubs lack set_status''s new
  if_plan_v kwarg (4 tests)'
kind: infra
tags:
- wf-fix
- main-red
created_at: '2026-08-17T13:35:00Z'
has_clean_result: false
origin_prompt: 'surfaced by #2155 step9c gate re-run 2026-08-17; pristine-oracle confirmed
  at e429f8cff9'
workflow: v1
---
# workflow-fix: main-red — test_task_workflow_post_marker_echo stubs lack the new set_status if_plan_v kwarg (4 tests red on main)

## Provenance

workflow_fix_target: tests/test_task_workflow_post_marker_echo.py
urgency: main-red
failing_test: tests/test_task_workflow_post_marker_echo.py::test_set_status_normal_echo_prints_path
wf_fix: true
Surfaced by task #2155's Step 9c gate re-run (2026-08-17); pristine-oracle CONFIRMED at base e429f8cff9 (detached scratch worktree, no #2155 payload): 4 failed / 57 passed in the file.

## Goal

Restore the four red tests to green on main. The mission-control rung-0 commits (`c0d76f99c9` + `8a90a881f2`, 2026-08-17) added an `if_plan_v` keyword argument to `set_status` calls (crash site: `scripts/task.py:680`), but the test file's monkeypatched stubs/lambdas do not accept it: `TypeError: ... got an unexpected keyword argument 'if_plan_v'` in all four:
- test_set_status_broken_pipe_on_echo_is_nonfatal
- test_set_status_normal_echo_prints_path
- test_set_status_followup_hold_refusal_exits_cleanly
- test_set_status_followups_running_missing_tag_warns

## Fix shape

Update the four stubs to accept the new kwarg (`**kwargs` or explicit `if_plan_v=None`), preserving each test's assertion substance; verify against the CURRENT set_status signature rather than pinning a new one. Coordinate with the live mission-control rung-0 session if it owns a fix in flight (one implementer per file set) — this is a lockstep miss from its own commits, so check its worktree/branch first.

## Acceptance criteria

1. `uv run pytest tests/test_task_workflow_post_marker_echo.py -q` green on main.
2. No weakening of the four tests' assertions.

Estimated GPU-hours (total): 0

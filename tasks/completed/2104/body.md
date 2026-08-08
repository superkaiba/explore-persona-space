---
title: 'workflow-fix: reconcile tests/test_workload_cmd_env_lint.py''s mock-backend'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b5bbd691dd5b
- urgent-main-red
created_at: '2026-08-06T00:03:21Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: tests/test_workload_cmd_env_lint.py,\
  \ src/explore_persona_space/backends/router.py\nbug_observed: 7 deterministic tests\
  \ in tests/test_workload_cmd_env_lint.py fail on the main checkout (and at fetched\
  \ origin/main base 8f95ad4158) — each _MockBackend nibi expects 1 launch and gets\
  \ 0 (e.g. test_kill_switch_env_skips_inline_lint, tests/test_workload_cmd_env_lint.py:865).\n\
  why_workflow_gap: a workflow-invariant test file is red on origin/main, so every\
  \ intervening session's Step 9c gate must re-classify the red (the #1643/#1681 fleet-cost\
  \ class); unverified hypothesis — verify at plan time: the #2054 runpod-first DEFAULT_AUTO_LANE_ORDER\
  \ reorder (and/or the #1997/#1775 pod-lifecycle change, both recent on this surface)\
  \ landed without updating these tests' auto-lane launch expectations.\nproposed_change:\
  \ reconcile tests/test_workload_cmd_env_lint.py's mock-backend launch expectations\
  \ with the post-#2054 runpod-first auto lane order (or fix the router if the launches\
  \ are genuinely mis-routed).\ndiff_sketch: |\n  In the 7 failing tests, either register\
  \ the mock runpod lane as the\n  expected launch target (post-#2054 order: runpod,\
  \ fellows, nibi, fir,\n  mila) or pin EPM_AUTO_LANE_ORDER in the fixtures so nibi\
  \ is first,\n  e.g.:\n  + monkeypatch.setenv(\"EPM_AUTO_LANE_ORDER\", \"nibi,fir,mila\"\
  )\nconfidence: medium\nrelated_task: #1996\nurgency: main-red\nfailing_test: tests/test_workload_cmd_env_lint.py::test_kill_switch_env_skips_inline_lint\n\
  wf_fix: true\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1996. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_workload_cmd_env_lint.py::test_kill_switch_env_skips_inline_lint` red on origin/main: reconcile tests/test_workload_cmd_env_lint.py's mock-backend launch expectations with the post-#2054 runpod-first auto lane order (or fix the router if the launches are genuinely mis-routed).

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** 7 deterministic tests in tests/test_workload_cmd_env_lint.py fail on the main checkout (and at fetched origin/main base 8f95ad4158) — each _MockBackend nibi expects 1 launch and gets 0 (e.g. test_kill_switch_env_skips_inline_lint, tests/test_workload_cmd_env_lint.py:865).
- **Failing node (router-verified):** `tests/test_workload_cmd_env_lint.py::test_kill_switch_env_skips_inline_lint`
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_workload_cmd_env_lint.py::test_kill_switch_env_skips_inline_lint -q` -> rc=1 at main @ 6fe11b459a (2026-08-06T00:03:09Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `tests/test_workload_cmd_env_lint.py, src/explore_persona_space/backends/router.py`
- Failing node: `tests/test_workload_cmd_env_lint.py::test_kill_switch_env_skips_inline_lint`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: tests/test_workload_cmd_env_lint.py, src/explore_persona_space/backends/router.py
- fingerprint: b5bbd691dd5b
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_workload_cmd_env_lint.py, src/explore_persona_space/backends/router.py
bug_observed: 7 deterministic tests in tests/test_workload_cmd_env_lint.py fail on the main checkout (and at fetched origin/main base 8f95ad4158) — each _MockBackend nibi expects 1 launch and gets 0 (e.g. test_kill_switch_env_skips_inline_lint, tests/test_workload_cmd_env_lint.py:865).
why_workflow_gap: a workflow-invariant test file is red on origin/main, so every intervening session's Step 9c gate must re-classify the red (the #1643/#1681 fleet-cost class); unverified hypothesis — verify at plan time: the #2054 runpod-first DEFAULT_AUTO_LANE_ORDER reorder (and/or the #1997/#1775 pod-lifecycle change, both recent on this surface) landed without updating these tests' auto-lane launch expectations.
proposed_change: reconcile tests/test_workload_cmd_env_lint.py's mock-backend launch expectations with the post-#2054 runpod-first auto lane order (or fix the router if the launches are genuinely mis-routed).
diff_sketch: |
  In the 7 failing tests, either register the mock runpod lane as the
  expected launch target (post-#2054 order: runpod, fellows, nibi, fir,
  mila) or pin EPM_AUTO_LANE_ORDER in the fixtures so nibi is first,
  e.g.:
  + monkeypatch.setenv("EPM_AUTO_LANE_ORDER", "nibi,fir,mila")
confidence: medium
related_task: #1996
urgency: main-red
failing_test: tests/test_workload_cmd_env_lint.py::test_kill_switch_env_skips_inline_lint
wf_fix: true
<!-- /workflow-fix-candidate -->

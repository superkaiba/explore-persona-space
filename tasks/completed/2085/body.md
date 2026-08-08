---
title: 'workflow-fix: update the two test files'' mocked auto-chain fixtures to the'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e48a62527498
- urgent-main-red
created_at: '2026-08-05T13:13:19Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: tests/test_workload_cmd_env_lint.py,\
  \ tests/test_workload_cmd_persist_lint.py\nbug_observed: 10 nodes red on PRISTINE\
  \ main (verified twice by #1997's Step 9c compare pristine-scratch oracle at f2bdd400d430\
  \ and a repo-root baseline run: 10 failed / 63 passed): each failing node asserts\
  \ the auto chain lands on nibi (`assert len(nibi.launches) == 1` → 0) in the workload-cmd\
  \ env/persist lint-arm tests.\nwhy_workflow_gap: workflow-invariant-adjacent tests\
  \ red on trunk force every intervening session's Step 9c gate to re-classify the\
  \ same pre-existing red (this session burned a 2-file baseline run + a 4-file pristine-oracle\
  \ compare cycle on it); the Step 10d gate's pass verdicts now rest on workflow-surface\
  \ pre-existing red.\nproposed_change: update the two test files' mocked auto-chain\
  \ fixtures to the current lane order — unverified hypothesis, verify at plan time:\
  \ the #2054 runpod-first promotion and/or #2028 GCP-removal (commit 538e0d7a9b landed\
  \ on this surface within the window) changed which mock backend the auto chain reaches,\
  \ so nibi is never consulted in the mocked scenario; fixtures should either make\
  \ the earlier rungs capacity-miss in the mock or pin an explicit backend.\ndiff_sketch:\
  \ |\n  In tests/test_workload_cmd_env_lint.py + tests/test_workload_cmd_persist_lint.py:\n\
  \  - mock chain setup: make runpod (and fellows) rungs capacity-miss so auto\n \
  \   still reaches nibi, OR assert on the FIRST consulted backend's launches\n  \
  \  instead of nibi's;\n  + re-derive expected lane from router.DEFAULT_AUTO_LANE_ORDER\
  \ rather than\n    hardcoding nibi as the auto landing rung.\nconfidence: medium\n\
  related_task: #1997\nurgency: main-red\nfailing_test: tests/test_workload_cmd_env_lint.py::test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds\n\
  wf_fix: true\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1997. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_workload_cmd_env_lint.py::test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds` red on origin/main: update the two test files' mocked auto-chain fixtures to the current lane order — unverified hypothesis, verify at plan time: the #2054 runpod-first promotion and/or #2028 GCP-removal (commit 538e0d7a9b landed on this surface within the window) changed which mock backend the auto chain reaches, so nibi is never consulted in the mocked scenario; fixtures should either make the earlier rungs capacity-miss in the mock or pin an explicit backend.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** 10 nodes red on PRISTINE main (verified twice by #1997's Step 9c compare pristine-scratch oracle at f2bdd400d430 and a repo-root baseline run: 10 failed / 63 passed): each failing node asserts the auto chain lands on nibi (`assert len(nibi.launches) == 1` → 0) in the workload-cmd env/persist lint-arm tests.
- **Failing node (router-verified):** `tests/test_workload_cmd_env_lint.py::test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds`
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_workload_cmd_env_lint.py::test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds -q` -> rc=1 at main @ f4250f8042 (2026-08-05T13:13:08Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `tests/test_workload_cmd_env_lint.py, tests/test_workload_cmd_persist_lint.py`
- Failing node: `tests/test_workload_cmd_env_lint.py::test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- This session carries a `workflow_fix_target:` Provenance line — it MUST
  NOT auto-route its own subagents' workflow-fix candidates (recursion
  guard, `.claude/rules/workflow-fix-on-bug.md` § Recursion guard).

## Provenance

- workflow_fix_target: tests/test_workload_cmd_env_lint.py, tests/test_workload_cmd_persist_lint.py
- fingerprint: e48a62527498
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: tests/test_workload_cmd_env_lint.py, tests/test_workload_cmd_persist_lint.py
bug_observed: 10 nodes red on PRISTINE main (verified twice by #1997's Step 9c compare pristine-scratch oracle at f2bdd400d430 and a repo-root baseline run: 10 failed / 63 passed): each failing node asserts the auto chain lands on nibi (`assert len(nibi.launches) == 1` → 0) in the workload-cmd env/persist lint-arm tests.
why_workflow_gap: workflow-invariant-adjacent tests red on trunk force every intervening session's Step 9c gate to re-classify the same pre-existing red (this session burned a 2-file baseline run + a 4-file pristine-oracle compare cycle on it); the Step 10d gate's pass verdicts now rest on workflow-surface pre-existing red.
proposed_change: update the two test files' mocked auto-chain fixtures to the current lane order — unverified hypothesis, verify at plan time: the #2054 runpod-first promotion and/or #2028 GCP-removal (commit 538e0d7a9b landed on this surface within the window) changed which mock backend the auto chain reaches, so nibi is never consulted in the mocked scenario; fixtures should either make the earlier rungs capacity-miss in the mock or pin an explicit backend.
diff_sketch: |
  In tests/test_workload_cmd_env_lint.py + tests/test_workload_cmd_persist_lint.py:
  - mock chain setup: make runpod (and fellows) rungs capacity-miss so auto
    still reaches nibi, OR assert on the FIRST consulted backend's launches
    instead of nibi's;
  + re-derive expected lane from router.DEFAULT_AUTO_LANE_ORDER rather than
    hardcoding nibi as the auto landing rung.
confidence: medium
related_task: #1997
urgency: main-red
failing_test: tests/test_workload_cmd_env_lint.py::test_launch_auto_flagged_cmd_warns_flags_marker_and_proceeds
wf_fix: true
<!-- /workflow-fix-candidate -->

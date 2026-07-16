---
title: 'workflow-fix: fix two red workflow tests on main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:12839736f95a
- daily-auto-filed
created_at: '2026-07-16T07:19:19Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): two workflow-invariant
  tests pre-existing red on main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1326 (emitting agent: implementer r1; both items discriminator-confirmed NOT introduced by #1326's diff).

## Goal

Fix two PRE-EXISTING red workflow-invariant tests on main: (1) `tests/test_living_docs.py::test_check_degrades_on_drifted_registry_path` — stale vs `find_task_path`'s registry-drift self-heal (task_workflow.py ~:3521); update the test to the self-heal contract. (2) `tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers` — #1092's 2026-07-11 followup-scope marker carries an unparseable label; fix the corpus-replay expectation or retro-repair the marker parse.

## Workflow gap

- **Bug observed:** two workflow-invariant tests fail on pristine main, poisoning every Step 9c test-verdict baseline that selects them.
- **Why it is a workflow gap:** red-on-main workflow tests force every session into per-run pre-existing-failure discrimination and can mask genuine regressions.
- **Confidence (emitter):** medium
- verified-at-filing: `uv run pytest tests/test_living_docs.py::test_check_degrades_on_drifted_registry_path tests/test_workflow_followup_labels.py::test_corpus_replay_all_historical_markers` → BOTH FAIL on current main (AssertionError at test_living_docs.py:532 — REGISTRY stale for #207 approved→completed; AssertionError at test_workflow_followup_labels.py:1124), re-run 2026-07-16 UTC at filing time (test-run evidence; grep n/a for a behavioral red-test claim)

## Proposed change (candidate diff sketch — refine in planning)

(1) update test_check_degrades_on_drifted_registry_path to the find_task_path self-heal contract; (2) fix the corpus-replay expectation for #1092's unparseable followup_label or retro-repair the marker parse.

## Scope / surfaces

- Primary target: `tests/test_living_docs.py, tests/test_workflow_followup_labels.py`
- Note: the living-docs failure surfaces a REGISTRY drift for task #207 (`tasks/approved/207` vs on-disk `tasks/completed/207`) — the planner should check whether `task.py audit --repair --apply` is part of the fix or a separate concern.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/` bodies (a `task.py audit --repair` registry re-sync via the canonical API is allowed if the plan calls for it).
- Never weaken a test to make it pass — encode the CURRENT intended contract.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 12839736f95a

- workflow_fix_target: tests/test_living_docs.py

parked prose follow-up (verbatim, from #1326 events.jsonl 2026-07-15T13:41:29Z): "(1) target_file: tests/test_living_docs.py — test_check_degrades_on_drifted_registry_path PRE-EXISTING red on main; stale vs find_task_path's :3521 registry-drift self-heal; proposed: update the test to the self-heal contract. (2) target_file: tests/test_workflow_followup_labels.py — test_corpus_replay_all_historical_markers PRE-EXISTING red on main; #1092's 2026-07-11 followup-scope marker carries an unparseable label; proposed: fix the corpus-replay expectation or retro-repair the marker parse. Both discriminator-confirmed NOT introduced by #1326's diff."

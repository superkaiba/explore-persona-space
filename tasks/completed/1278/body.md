---
title: 'workflow-fix: consolidate _stub_fleet_mutating_passes'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2a75a0425585
- daily-auto-filed
created_at: '2026-07-12T06:52:17Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): the _stub_fleet_mutating_passes
  watcher-test helper is duplicated across tests/test_autonomous_session_watch.py
  and tests/test_stalled_detector_and_gc.py with diverged docstrings — silent-drift
  risk'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-11 parked-candidate routing pass (Step C) from a workflow-fix candidate parked on task #1265 (emitting agent: planner, Phase 1; parked under the recursion guard).

## Goal

Consolidate the duplicated `_stub_fleet_mutating_passes` watcher-test helper into one shared definition.

## Workflow gap

- **Bug observed:** the `_stub_fleet_mutating_passes` watcher-test helper is duplicated across `tests/test_autonomous_session_watch.py` and `tests/test_stalled_detector_and_gc.py` with diverged docstrings — silent-drift risk (pass lists currently match).
- **Why it is a workflow gap:** same silent-drift class as #1265's hermeticity guards — a new watcher pass added to one copy but not the other silently un-stubs fleet-mutating behavior in half the suite. Unlike the #1265 autouse fixtures this helper is call-explicit with call-site-relative ordering semantics, so consolidation touches every full-`main()` call site; belongs in its own minimal-diff task.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "_stub_fleet_mutating_passes" tests/test_autonomous_session_watch.py tests/test_stalled_detector_and_gc.py` → 2 definitions (test_stalled_detector_and_gc.py:50, test_autonomous_session_watch.py:1679) + multiple call sites in each (2026-07-12) — duplication present post-#1265.

## Proposed change (candidate diff sketch — refine in planning)

(none — synthesized from prose follow-up) Move one canonical `_stub_fleet_mutating_passes` into `tests/conftest.py` (or a shared test util module), delete the two copies, update all call sites; keep call-site-relative ordering semantics.

## Scope / surfaces

- Primary target: `tests/test_autonomous_session_watch.py`, `tests/test_stalled_detector_and_gc.py` (+ `tests/conftest.py` for the shared home)
- Grep for other copies before editing (`grep -rln '_stub_fleet_mutating_passes' tests/`).

## Constraints / invariants

- Workflow-surface only (workflow-invariant tests) — never experiment code, `configs/`, or `tasks/`.
- `uv run pytest tests/test_autonomous_session_watch.py tests/test_stalled_detector_and_gc.py` green; ruff passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: tests/test_autonomous_session_watch.py, tests/test_stalled_detector_and_gc.py
- fingerprint: 2a75a0425585

Origin (parked prose candidate on #1265, 2026-07-11T15:14:58Z): "target_file: tests/test_autonomous_session_watch.py, tests/test_stalled_detector_and_gc.py. proposed_change: consolidate the duplicated _stub_fleet_mutating_passes helper (pass lists match, docstrings diverged) into one shared definition. why_workflow_gap: same silent-drift class as #1265's guards — but call-explicit with call-site-relative ordering semantics, so consolidation touches every full-main() call site; belongs in its own minimal-diff task. confidence: medium."

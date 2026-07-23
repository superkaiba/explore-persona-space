---
title: 'daily-fix: fix red asw-stub pin test on main (conftest:288)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:2c3fafec7c77
- daily-auto-filed
created_at: '2026-07-22T06:44:58Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): tests/conftest.py:288 hasattr-guarded
  asw stub makes test_no_hasattr_asw_stub_guards red on pristine main'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1582 (emitting agent: code-reviewer, #1582 round 1). URGENT-ish: the pin test is red on pristine main, so every full-suite Step-9c run carries a pre-existing failure.

## Goal

Fix the `tests/conftest.py` hasattr-guarded autonomous-session-watch stub so `tests/test_no_hasattr_asw_stub_guards.py` passes on main again.

## Workflow gap

- **Bug observed:** `tests/conftest.py:288` (`if hasattr(asw, "_launch_guard_apply"):`) violates the #1303 no-hasattr-guarded-asw-stubs invariant its own pin test enforces — `tests/test_no_hasattr_asw_stub_guards.py::test_no_hasattr_guarded_asw_stubs_in_tests` is RED on pristine main.
- **Why it is a workflow gap:** a red workflow-invariant pin test on main poisons every Step-9c full-suite verdict (pre-existing-red noise other sessions must attribute around).
- **Confidence (emitter):** high (reviewer ran the test; offense byte-present on origin/main).
- verified-at-filing: `sed -n '284,292p' tests/conftest.py` → the hasattr guard is present at :287-288 (2026-07-22); `uv run pytest tests/test_no_hasattr_asw_stub_guards.py -x -q` → `1 failed in 0.66s` (re-reproduced at filing).

## Proposed change (candidate diff sketch — refine in planning)

One-line fix: replace the hasattr guard with a bare `monkeypatch.setattr(asw, "_launch_guard_apply", ...)`, or add an allowlist entry to the pin test if the guard is deliberate.

## Scope / surfaces

- Primary target: `tests/conftest.py`
- Also check `tests/test_no_hasattr_asw_stub_guards.py` allowlist semantics before choosing which side to fix.

## Constraints / invariants

- Workflow-surface only (workflow-invariant test infrastructure). Full `tests/test_no_hasattr_asw_stub_guards.py` + the touched watcher tests pass after the fix.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: 2c3fafec7c77

- workflow_fix_target: tests/conftest.py

Verbatim parked candidate (task #1582 events, 2026-07-21T13:26:12Z): "parked — running under EPM_WORKFLOW_FIX_SESSION / workflow_fix_target, see workflow-fix-on-bug § Recursion guard. source: prose-followup (code-reviewer r1). target_file: tests/conftest.py. bug_observed: tests/conftest.py:288 ('if hasattr(asw, \"_launch_guard_apply\"):') violates the #1303 no-hasattr-guarded-asw-stubs invariant its own pin test enforces — tests/test_no_hasattr_asw_stub_guards.py::test_no_hasattr_guarded_asw_stubs_in_tests is red on pristine main. proposed_change: one-line fix (bare monkeypatch.setattr or allowlist entry). confidence: high (reviewer ran the test; offense byte-present on origin/main)."

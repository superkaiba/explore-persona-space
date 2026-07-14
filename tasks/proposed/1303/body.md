---
title: 'daily-fix: asw test hasattr stub loops -> fail-loud'
kind: infra
tags:
- wf-fix
- wf-fix-fp:071d0ad9ac57
- daily-auto-filed
created_at: '2026-07-14T06:36:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): 5 pre-existing determinism-only
  ''if hasattr(asw, name)'' stub loops remain in tests/test_autonomous_session_watch.py
  (post-helper, no hermeticity gap)'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-13 parked-candidate routing pass (Step C) from a candidate parked on task #1295 (emitting agent: code-reviewer round 1, recursion-guard park at 2026-07-13T09:14:30Z).

## Goal

Clean up the 5 remaining determinism-only `if hasattr(asw, name)` stub loops in `tests/test_autonomous_session_watch.py` to bare `monkeypatch.setattr` (fail-loud), enabling a pinnable `grep -rn "if hasattr(asw" tests/` lint.

## Workflow gap

- **Bug observed:** 5 pre-existing determinism-only `if hasattr(asw, name)` stub loops remain in `tests/test_autonomous_session_watch.py` (post-#1295-helper; no hermeticity gap) — a silently-skipping stub pattern: a renamed watcher attribute makes the stub a no-op instead of failing loud.
- **Why it is a workflow gap:** #1295 consolidated the third inline pass-neutralization loop onto the shared conftest helper precisely to kill this class; the same-file siblings keep the class alive and block a mechanical lint pin.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "if hasattr(asw" tests/test_autonomous_session_watch.py` → 5 hits at lines 10012, 12113, 12893, 16261, 17307 — exactly the candidate's cited lines (2026-07-14 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Replace each `if hasattr(asw, name): monkeypatch.setattr(...)` loop with bare `monkeypatch.setattr(asw, name, ...)` (fail-loud on a renamed attribute), or route through the shared conftest neutralization helper #1295 landed; then add the grep-based lint/test pin so no new `if hasattr(asw` sites land in tests/.

## Scope / surfaces

- Primary target: `tests/test_autonomous_session_watch.py`
- Secondary: the shared conftest helper (reuse), a workflow-invariant pin test

## Constraints / invariants

- Test-surface only; full `uv run pytest tests/test_autonomous_session_watch.py` must pass.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: tests/test_autonomous_session_watch.py
- fingerprint: 071d0ad9ac57

Verbatim parked candidate (prose-followup, parked on #1295 events.jsonl at 2026-07-13T09:14:30Z):

> target_file: tests/test_autonomous_session_watch.py; bug_observed: 5 pre-existing determinism-only 'if hasattr(asw, name)' stub loops remain (lines 10012, 12113, 12893, 16261, 17307 — all post-helper, no hermeticity gap); proposed_change: same-class cleanup to bare monkeypatch.setattr (fail-loud), enabling a pinnable 'grep -rn "if hasattr(asw" tests/' lint; confidence: medium; related_task: #1295.

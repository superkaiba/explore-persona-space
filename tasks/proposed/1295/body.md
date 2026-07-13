---
title: 'daily-fix: consolidate 3rd inline loop in auth-outage test'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cb90b57b553e
- daily-auto-filed
created_at: '2026-07-13T06:44:45Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): a THIRD inline 22-name
  pass-neutralization loop (lines ~355-380) drives full asw.main([]), is missing boot_death_pass,
  and uses hasattr silent-skip instead of fail-loud raising=True — pre-existing on
  main, same #1267 drift class #1278 fixed for the other two files.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 Step C parked-candidate routing pass, from the prose follow-up parked on task #1278 (2026-07-12T07:33:34Z, recursion guard; emitting agent: #1278's round-1 code-reviewer bug-class sweep).

## Goal

Consolidate the third inline pass-neutralization loop in tests/test_auth_outage_guard.py onto the shared tests/conftest.py `_stub_fleet_mutating_passes` helper.

## Workflow gap

- **Bug observed:** a THIRD inline 22-name pass-neutralization loop (tests/test_auth_outage_guard.py ~355-380) drives full `asw.main([])`, is missing `boot_death_pass`, and uses `hasattr` silent-skip instead of fail-loud `raising=True` — pre-existing on main (byte-parity confirmed by #1278), the same #1267 drift class #1278 fixed for the other two files. Per #1278's own audit, a missing `boot_death_pass` in such a copy is "a live hermeticity gap" — a test invoking the full watcher main() could execute the boot-death lane unstubbed.
- **Why it is a workflow gap:** #1278 created the shared conftest helper precisely so per-file inline copies stop drifting; this remaining copy is the drift class live on a workflow-invariant test.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "_stub_fleet_mutating_passes\|boot_death_pass\|hasattr" tests/test_auth_outage_guard.py` → no shared-helper use, no `boot_death_pass`, `hasattr` skip at :381 (2026-07-13, post-#1278-merge).

## Proposed change (candidate diff sketch — refine in planning)

Replace the inline ~355-380 loop with the shared `tests/conftest.py` `_stub_fleet_mutating_passes` helper (which carries `boot_death_pass` and `raising=True`), per #1278's reference diff (PR #1031, squash 6c1a8f838abb).

## Scope / surfaces

- Primary target: `tests/test_auth_outage_guard.py`
- Reference implementation: #1278 (PR #1031).

## Constraints / invariants

- Workflow-surface only (watcher-invariant test). Lint + ruff pass. Recursion guard applies to the spawned session.

## Provenance

- fingerprint: cb90b57b553e

- workflow_fix_target: tests/test_auth_outage_guard.py

Verbatim origin: prose follow-up parked on #1278 events.jsonl at 2026-07-12T07:33:34Z (related_task #1278).

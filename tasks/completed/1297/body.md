---
title: 'daily-fix: issue658 test sys.modules stub leak'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-13T06:44:52Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): the module-scope sys.modules
  stub of issue658_common is never torn down (and the early-return guard silently
  accepts a pre-loaded real module), so stubs leak across test files in the same pytest
  session.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 Step C parked-candidate routing pass, from the prose follow-up parked on task #1037 (2026-07-05T13:04:24Z; emitting agent: #1037's implementer). NON-workflow-surface (experiment test hygiene) — filed `daily-auto-filed` only, no wf-fix tags.

## Goal

Fix the `sys.modules`-stub isolation bug in `tests/test_issue658_judge_e0_batch_migration.py` — the stub leaks across test files in the same pytest session.

## Bug

- **Observed:** `_install_issue658_common_stub()` registers a fake `issue658_common` module in `sys.modules` at import time (`MOD = _load_module()` at module scope) and never removes it; the guard `if "issue658_common" in sys.modules: return` also means a REAL `issue658_common` loaded earlier silently replaces the stub (and vice versa: any later test importing `issue658_common` in the same session gets this file's trivial stand-ins).
- **Confidence:** medium.
- verified-at-filing: `sed -n '50,80p' tests/test_issue658_judge_e0_batch_migration.py` (2026-07-13) — stub installed at module scope, no fixture teardown, early-return guard present.

## Proposed change (refine in planning)

Scope the stub with a proper fixture (install in a module-scoped fixture with teardown restoring the prior `sys.modules` entry), or namespace the module under test so the stub never occupies the shared `issue658_common` key.

## Scope / surfaces

- `tests/test_issue658_judge_e0_batch_migration.py` (experiment test — ordinary infra code-change pipeline applies).

## Constraints / invariants

- Tests keep passing in isolation AND in full-suite order; no ordering-dependent behavior remains.

## Provenance

Origin: prose follow-up parked on #1037 events.jsonl at 2026-07-05T13:04:24Z ("tests/test_issue658_judge_e0_batch_migration.py sys.modules-stub isolation bug (stubs leak across test files in the same pytest session)"), re-verified at filing time.

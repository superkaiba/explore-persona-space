---
title: 'daily-fix: guard backend_poll failure_classifier import'
kind: infra
tags:
- wf-fix
- wf-fix-fp:0916c348052d
- daily-auto-filed
created_at: '2026-07-14T06:36:38Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-13 problem sweep (route 2): backend_poll.py:1384 ''from
  failure_classifier import OUR_CODE_FRAME'' is an unguarded lazy scripts-local import
  of the exact class #1296 fixed, called upstream of the newly-guarded sites; sibling
  bare-lazy sites in pod_config.py'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-13 parked-candidate routing pass (Step C) from a candidate parked on task #1296 (emitting agent: code-reviewer round 1, recursion-guard park at 2026-07-13T10:09:10Z).

## Goal

Guard the remaining unguarded scripts-local lazy import in `scripts/backend_poll.py` (`from failure_classifier import OUR_CODE_FRAME`) with `_ensure_scripts_dir_on_sys_path()`, widen the lazy-import guard test to scripts-local modules generally, and ride the `scripts/pod_config.py` bare-lazy siblings on the same fix.

## Workflow gap

- **Bug observed:** `backend_poll.py:1384` `from failure_classifier import OUR_CODE_FRAME` is an unguarded lazy scripts-local import of the exact failure class #1296 fixed for the three `from runpod_api import` sites (called from `_maybe_escalate_runpod_cuda_ima` upstream of the newly-guarded sites); sibling bare-lazy `from runpod_api import` sites exist in `scripts/pod_config.py` (~L576 try/except-wrapped, ~L1549 bare).
- **Why it is a workflow gap:** #1296's guard test pins only the `runpod_api` import form, so the same crash class (library-context invocation without `scripts/` on `sys.path`) survives through the unpinned module names.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n "from failure_classifier import\|_ensure_scripts_dir_on_sys_path" scripts/backend_poll.py` → helper defined at :159, guards at :1569/:2221/:2397, unguarded `from failure_classifier import OUR_CODE_FRAME` at :1384; `grep -n "from runpod_api import" scripts/pod_config.py` → sites at ~:576 (try/except) and ~:1549 (bare) (2026-07-14 UTC)

## Proposed change (candidate diff sketch — refine in planning)

```
+ _ensure_scripts_dir_on_sys_path()
  from failure_classifier import OUR_CODE_FRAME
```
plus: widen `test_every_lazy_runpod_api_import_is_bootstrap_guarded` (tests/test_backend_poll.py) to scripts-local modules generally (`runpod_api|failure_classifier|issue664_\w+` + the plain-import form); add the equivalent bootstrap (or a guarded pattern) to the `pod_config.py` sibling sites.

## Scope / surfaces

- Primary targets: `scripts/backend_poll.py`, `tests/test_backend_poll.py`, `scripts/pod_config.py`

## Constraints / invariants

- Workflow-helper scripts only; `uv run pytest tests/test_backend_poll.py` passes; ruff clean on touched files.
- Recursion guard applies to the spawned session.

## Provenance

- workflow_fix_target: scripts/backend_poll.py, tests/test_backend_poll.py, scripts/pod_config.py
- fingerprint: 0916c348052d

Verbatim parked candidate (prose-followup, parked on #1296 events.jsonl at 2026-07-13T10:09:10Z):

> target_file: scripts/backend_poll.py, tests/test_backend_poll.py, scripts/pod_config.py. bug_observed: backend_poll.py:1384 'from failure_classifier import OUR_CODE_FRAME' is an unguarded lazy scripts/-local import of the exact class #1296 fixed (called from _maybe_escalate_runpod_cuda_ima:1664 upstream of the newly-guarded :2398 site); sibling bare-lazy sites at pod_config.py:576/:1549. proposed_change: add _ensure_scripts_dir_on_sys_path() above :1384; widen test_every_lazy_runpod_api_import_is_bootstrap_guarded to scripts-local modules generally (runpod_api|failure_classifier|issue664_\w+ + plain-import form); pod_config.py siblings ride the same fix. confidence: medium. related_task: #1296.

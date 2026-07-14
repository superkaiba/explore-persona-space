---
title: 'daily-fix: scripts_dir bootstrap before backend_poll import'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7f4e7ffb5459
- daily-auto-filed
created_at: '2026-07-13T06:44:48Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-12 problem sweep (route 2): backend_poll.py:1554 does
  a bare ''from runpod_api import get_pod_by_name'' without the scripts_dir sys.path
  bootstrap its sibling sites (:1709/:1771) establish — ModuleNotFoundError when imported
  as a module before scripts/ is on sys.path (originally cited test has since been
  removed; import-site inconsistency re-verified at filing).'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-12 Step C parked-candidate routing pass, from the candidate parked on task #710 (2026-06-28T09:29:30Z, recursion guard; emitting agent: #710's implementer).

## Goal

Add the established `scripts_dir` sys.path bootstrap before the bare `from runpod_api import get_pod_by_name` in `scripts/backend_poll.py` (now at line ~1554; originally cited as :565).

## Workflow gap

- **Bug observed (original):** a bare `from runpod_api import get_pod_by_name` without the `scripts_dir` sys.path bootstrap its sibling import sites establish; the editable install puts only `<repo>/src` on sys.path, so the RunPod async-wedge override path raises `ModuleNotFoundError` when `backend_poll` is imported as a module (tests) before anything has inserted `scripts/`.
- **Why it is a workflow gap:** the module's other `from runpod_api import ...` sites (now :2205/:2380 region) sit behind explicit `scripts_dir` bootstraps (:1709-1711, :1771-1773); the :1554 site does not — an import-order landmine on the #664/#667 billing-leak detection path.
- **Confidence (emitter):** high at emission; downgraded to LOW at filing — see the staleness note below. Filed per the standing directive; the spawned planner may deflect with a reasoned no-change report if the site is provably unreachable without scripts/ on sys.path.
- verified-at-filing: `grep -n "from runpod_api import" scripts/backend_poll.py` → bare import at :1554 with no module-top or local bootstrap confirmed; HOWEVER the candidate's cited failing test `test_backend_poll_script_produces_legacy_poll_pipeline_json_shape` NO LONGER EXISTS (`pytest tests/test_backend_poll.py::<name>` → "no tests ran"), so the original repro is gone (2026-07-13). The inconsistency between sibling import sites is the surviving concrete claim.

## Proposed change (candidate diff sketch — refine in planning)

```diff
     if getattr(handle, "backend", None) != "runpod":
         return result
+    scripts_dir = str(_Path(__file__).resolve().parent)
+    if scripts_dir not in sys.path:
+        sys.path.insert(0, scripts_dir)
     from runpod_api import get_pod_by_name  # live RunPod API (X-Team-Id baked in)
```
(the exact 4-line pattern already at backend_poll.py:1709-1711 / :1771-1773.)

## Scope / surfaces

- Primary target: `scripts/backend_poll.py`

## Constraints / invariants

- Workflow-surface only. Lint + ruff pass. Recursion guard applies to the spawned session.

## Provenance

- fingerprint: 7f4e7ffb5459

- workflow_fix_target: scripts/backend_poll.py

Verbatim origin: candidate block parked on #710 events.jsonl at 2026-06-28T09:29:30Z (related_task #710).

---
title: 'Fix thread-caps main red: issue1773_register_steer_stats imports matplotlib
  before load_dotenv'
kind: infra
tags:
- urgent-main-red
created_at: '2026-07-31T23:29:55Z'
has_clean_result: false
origin_prompt: 'Auto-filed from #1946 implementer round: origin/main red on tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  via scripts/issue1773_register_steer_stats.py matplotlib-before-dotenv (verified
  live rc=1).'
workflow: v1
---
## Overview / Motivation

Filed from the #1946 implementer round's gate-scope check: origin/main is currently RED on a workflow-invariant test, so every fleet session's Step 9c gate must re-classify it until fixed.

## Goal

Fix `scripts/issue1773_register_steer_stats.py` so `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is green on main: the script imports matplotlib (line 20) before any `load_dotenv()` call, violating the VM-entrypoint thread-caps/dotenv-ordering invariant (#847/#891 family).

## The defect

- **Bug observed:** `test_no_new_torch_before_dotenv_vm_entrypoints` FAILs on origin/main (rc=1, assert at tests/test_shared_vm_thread_caps.py:888) naming `scripts/issue1773_register_steer_stats.py`.
- **Cause:** matplotlib imported at module top (L20-23) with no `explore_persona_space.orchestrate.env.load_dotenv` before it; introduced by main commit `04e111a7ad` (task #1773), postdating the step9c baseline ledger (`main_sha 3b78a230f834`), so it reads as NEW red at every gate.
- verified-at-filing: `sed -n 1,30p scripts/issue1773_register_steer_stats.py` + `grep -n 'load_dotenv\|import matplotlib' scripts/issue1773_register_steer_stats.py` → matplotlib at L20/L23, zero load_dotenv hits; one bounded pytest run of the node → rc=1, FAILED (2026-07-31).

## Proposed change

2-line import reorder: add the project `load_dotenv()` (via `explore_persona_space.orchestrate.env`) before the matplotlib/numpy/scipy imports, matching every other `scripts/issue*_*.py` VM entrypoint the test enumerates. Run the single test node to green; the no-flags workflow_lint red on `scripts/issue1689_user_slot_capture.py:752` is a SEPARATE known-ledger item — do not touch it here.

## Scope / constraints

- One file: `scripts/issue1773_register_steer_stats.py` (experiment entrypoint — NOT workflow surface).
- Acceptance: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q` green from a clean checkout; ruff clean on the touched file.

## Provenance

Surfaced by the #1946 implementer round (gate-scope pin-sweep hit, 2026-07-31); #1773 is the introducing task.

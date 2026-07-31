---
title: 'daily-fix: fix red main: test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded
  — add the scripts-dir bootstrap (call `_en'
kind: infra
tags:
- urgent-main-red
created_at: '2026-07-31T06:33:14Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1092_build_corpus.py\n\
  bug_observed: tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded\
  \ is red on pristine origin/main (verified at main tip 563d6627cf, rc=1) — scripts/issue1092_build_corpus.py:2029\
  \ has a lazy `from issue779_common import TRAITS` with no scripts-dir bootstrap\
  \ in the preceding 12 lines (landed with merge cfba89b1c8; node absent from the\
  \ step9c baseline ledger refreshed 2026-07-30T17:50:10Z).\nwhy_workflow_gap: a live-red\
  \ workflow-invariant test on main forces every intervening session's Step 9c gate\
  \ to re-classify the red (#1643 class) until the offending experiment script gains\
  \ the #1296/#1304 bootstrap guard.\nproposed_change: add the scripts-dir bootstrap\
  \ (call `_ensure_scripts_dir_on_sys_path()` or `sys.path.insert(0, scripts_dir)`\
  \ per the #1296/#1304 convention) directly above the lazy import at scripts/issue1092_build_corpus.py:2029.\n\
  diff_sketch: |\n      if args.smoke:\n  +       _ensure_scripts_dir_on_sys_path()\
  \  # #1296/#1304: lazy scripts-local import guard\n          from issue779_common\
  \ import TRAITS\nconfidence: high\nrelated_task: #1852\nurgency: main-red\nfailing_test:\
  \ tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1852. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded` red on origin/main: add the scripts-dir bootstrap (call `_ensure_scripts_dir_on_sys_path()` or `sys.path.insert(0, scripts_dir)` per the #1296/#1304 convention) directly above the lazy import at scripts/issue1092_build_corpus.py:2029.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded is red on pristine origin/main (verified at main tip 563d6627cf, rc=1) — scripts/issue1092_build_corpus.py:2029 has a lazy `from issue779_common import TRAITS` with no scripts-dir bootstrap in the preceding 12 lines (landed with merge cfba89b1c8; node absent from the step9c baseline ledger refreshed 2026-07-30T17:50:10Z).
- **Failing node (router-verified):** `tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded -q` -> rc=1 at main @ e3fffa8ec5 (2026-07-31T06:33:10Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1092_build_corpus.py`
- Failing node: `tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: fd2276f1ec88
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1092_build_corpus.py
bug_observed: tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded is red on pristine origin/main (verified at main tip 563d6627cf, rc=1) — scripts/issue1092_build_corpus.py:2029 has a lazy `from issue779_common import TRAITS` with no scripts-dir bootstrap in the preceding 12 lines (landed with merge cfba89b1c8; node absent from the step9c baseline ledger refreshed 2026-07-30T17:50:10Z).
why_workflow_gap: a live-red workflow-invariant test on main forces every intervening session's Step 9c gate to re-classify the red (#1643 class) until the offending experiment script gains the #1296/#1304 bootstrap guard.
proposed_change: add the scripts-dir bootstrap (call `_ensure_scripts_dir_on_sys_path()` or `sys.path.insert(0, scripts_dir)` per the #1296/#1304 convention) directly above the lazy import at scripts/issue1092_build_corpus.py:2029.
diff_sketch: |
      if args.smoke:
  +       _ensure_scripts_dir_on_sys_path()  # #1296/#1304: lazy scripts-local import guard
          from issue779_common import TRAITS
confidence: high
related_task: #1852
urgency: main-red
failing_test: tests/test_backend_poll.py::test_every_lazy_scripts_local_import_is_bootstrap_guarded
wf_fix: false
<!-- /workflow-fix-candidate -->

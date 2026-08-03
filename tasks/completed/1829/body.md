---
title: 'daily-fix: fix red main: test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  — call explore_persona_space.orchestrate.e'
kind: infra
tags:
- urgent-main-red
created_at: '2026-07-29T13:23:38Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1773_analyzer_plots.py\n\
  bug_observed: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints\
  \ fails on pristine origin/main — scripts/issue1773_analyzer_plots.py has a module-top\
  \ heavy import at line 25 with no load_dotenv() call (verified by the #1791 Step\
  \ 9c gate run + pristine-scratch oracle strip, 2026-07-29).\nwhy_workflow_gap: a\
  \ live-red workflow-surface scan pin forces every in-flight session's Step 9c gate\
  \ / Step 10d TG legs to re-classify the red until fixed (the #1643 fleet-wide per-hour\
  \ cost shape). NOTE: the offending file itself is EXPERIMENT-script scope (wf_fix:\
  \ false) — route as a daily-fix/implementer task, not a wf-fix session.\nproposed_change:\
  \ call explore_persona_space.orchestrate.env.load_dotenv() before the heavy import\
  \ in scripts/issue1773_analyzer_plots.py per #847.\ndiff_sketch: |\n  + from explore_persona_space.orchestrate.env\
  \ import load_dotenv\n  + load_dotenv()\n    <before the module-top heavy import\
  \ at line 25>\nconfidence: high\nurgency: main-red\nfailing_test: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints\n\
  wf_fix: false\nrelated_task: #1791\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1791. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` red on origin/main: call explore_persona_space.orchestrate.env.load_dotenv() before the heavy import in scripts/issue1773_analyzer_plots.py per #847.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints fails on pristine origin/main — scripts/issue1773_analyzer_plots.py has a module-top heavy import at line 25 with no load_dotenv() call (verified by the #1791 Step 9c gate run + pristine-scratch oracle strip, 2026-07-29).
- **Failing node (router-verified):** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q` -> rc=1 at main @ cfde049a3c (2026-07-29T13:23:30Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1773_analyzer_plots.py`
- Failing node: `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: 67e489c5ed68
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1773_analyzer_plots.py
bug_observed: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints fails on pristine origin/main — scripts/issue1773_analyzer_plots.py has a module-top heavy import at line 25 with no load_dotenv() call (verified by the #1791 Step 9c gate run + pristine-scratch oracle strip, 2026-07-29).
why_workflow_gap: a live-red workflow-surface scan pin forces every in-flight session's Step 9c gate / Step 10d TG legs to re-classify the red until fixed (the #1643 fleet-wide per-hour cost shape). NOTE: the offending file itself is EXPERIMENT-script scope (wf_fix: false) — route as a daily-fix/implementer task, not a wf-fix session.
proposed_change: call explore_persona_space.orchestrate.env.load_dotenv() before the heavy import in scripts/issue1773_analyzer_plots.py per #847.
diff_sketch: |
  + from explore_persona_space.orchestrate.env import load_dotenv
  + load_dotenv()
    <before the module-top heavy import at line 25>
confidence: high
urgency: main-red
failing_test: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
wf_fix: false
related_task: #1791
<!-- /workflow-fix-candidate -->

---
title: 'daily-fix: fix red main: test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  — add the standard `from explore_persona_s'
kind: infra
tags:
- urgent-main-red
created_at: '2026-07-30T14:33:22Z'
has_clean_result: false
origin_prompt: "<!-- workflow-fix-candidate v1 -->\ntarget_file: scripts/issue1776_p3p4_supplement.py\n\
  bug_observed: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints\
  \ FAILs on origin/main — scripts/issue1776_p3p4_supplement.py imports torch at module\
  \ top (line ~38 region) with zero load_dotenv() calls, so the shared-VM #847 thread\
  \ caps never bind in-process.\nwhy_workflow_gap: the #1776 inline round landed a\
  \ VM entrypoint that violates the pinned torch-before-dotenv invariant, leaving\
  \ a live red every Step 9c gate must re-classify; verified at origin/main (`git\
  \ show origin/main:scripts/issue1776_p3p4_supplement.py | grep -c load_dotenv` →\
  \ 0).\nproposed_change: add the standard `from explore_persona_space.orchestrate.env\
  \ import load_dotenv; load_dotenv()` before the torch/numpy imports in scripts/issue1776_p3p4_supplement.py\
  \ so the pinned invariant test goes green on main.\ndiff_sketch: |\n  + from explore_persona_space.orchestrate.env\
  \ import load_dotenv\n  + load_dotenv()\n  import numpy as np\n  import torch\n\
  confidence: high\nrelated_task: #1827\nurgency: main-red\nfailing_test: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints\n\
  wf_fix: false\n<!-- /workflow-fix-candidate -->"
workflow: v1
---
## Overview / Motivation

Auto-filed by the watcher urgent-park router (#1681,
`autonomous_session_watch.urgent_wf_park_pass`) from an URGENT
(`urgency: main-red`) parked workflow-fix candidate raised on task
#1827. The named test is red on origin/main NOW — every
intervening session's Step 9c gate re-classifies it until this fix lands.

## Goal

fix `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` red on origin/main: add the standard `from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()` before the torch/numpy imports in scripts/issue1776_p3p4_supplement.py so the pinned invariant test goes green on main.

## Workflow gap

- **Bug observed (emitter's claim, candidate block):** tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints FAILs on origin/main — scripts/issue1776_p3p4_supplement.py imports torch at module top (line ~38 region) with zero load_dotenv() calls, so the shared-VM #847 thread caps never bind in-process.
- **Failing node (router-verified):** `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
- **Confidence (emitter):** high
- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q` -> rc=1 at main @ 32cb7f088b (2026-07-30T14:33:19Z) (FAILED — red confirmed)

## Proposed change (candidate diff sketch — refine in planning)

(see the verbatim candidate block under `## Provenance` — the router
forwards it unmodified and never synthesizes fields)

## Scope / surfaces

- Primary target: `scripts/issue1776_p3p4_supplement.py`
- Failing node: `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`

## Constraints / invariants

- NON-workflow-surface fix (`wf_fix: false` declared by the parking
  session — the /daily route-2 analogue); scope stays on the named
  target, never `tasks/` state.

## Provenance

- fingerprint: aadb6352736b
- routed-by: autonomous_session_watch urgent-wf-park-router (#1681)

<!-- workflow-fix-candidate v1 -->
target_file: scripts/issue1776_p3p4_supplement.py
bug_observed: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints FAILs on origin/main — scripts/issue1776_p3p4_supplement.py imports torch at module top (line ~38 region) with zero load_dotenv() calls, so the shared-VM #847 thread caps never bind in-process.
why_workflow_gap: the #1776 inline round landed a VM entrypoint that violates the pinned torch-before-dotenv invariant, leaving a live red every Step 9c gate must re-classify; verified at origin/main (`git show origin/main:scripts/issue1776_p3p4_supplement.py | grep -c load_dotenv` → 0).
proposed_change: add the standard `from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()` before the torch/numpy imports in scripts/issue1776_p3p4_supplement.py so the pinned invariant test goes green on main.
diff_sketch: |
  + from explore_persona_space.orchestrate.env import load_dotenv
  + load_dotenv()
  import numpy as np
  import torch
confidence: high
related_task: #1827
urgency: main-red
failing_test: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
wf_fix: false
<!-- /workflow-fix-candidate -->

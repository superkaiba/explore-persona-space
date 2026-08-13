---
title: scripts/issue2224_fold_figures.py breaks test_shared_vm_thread_caps on main
  (missing load_dotenv-before-torch preamble)
kind: infra
tags:
- from-2265
created_at: '2026-08-13T08:55:55Z'
has_clean_result: false
origin_prompt: 'Pre-existing red surfaced by #2265 implementer round: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  fails on pristine origin/main via scripts/issue2224_fold_figures.py'
workflow: v1
---
## Goal

Make `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` green on `main` again: `scripts/issue2224_fold_figures.py` (landed on main in commit `3743121c21`, task #2224 fold round) has a module-top heavy import (torch-class) with no preceding `orchestrate.env.load_dotenv()` call, so the shared-VM thread-cap invariant test fails on pristine origin/main.

## Why now

Found by the #2265 implementer round (2026-08-13) while running the Step 9c gate-scope union in a clean worktree cut from origin/main: the failure reproduces with the file blob byte-identical to origin/main, i.e. pre-existing and unrelated to #2265's diff. Every session whose diff maps `tests/test_shared_vm_thread_caps.py` inherits this red at its Step 9c gate; baselines mask it (`step9c_baseline.py compare` treats it as pre-existing red), which means the invariant it pins — no uncapped torch import on shared-VM entrypoints (#847/#891) — is silently unenforced fleet-wide until fixed.

## Fix shape (either, small)

- Two-line fix: add the `load_dotenv()`-before-torch preamble to `scripts/issue2224_fold_figures.py` per `.claude/rules/code-style.md` § Shared-VM CPU thread caps (the test's own remediation message names the expected form); OR
- if the script is deliberately pod-only, add it to the test's grandfather/exemption list with a one-line reason.

## Verification

`uv run pytest tests/test_shared_vm_thread_caps.py -x` green on main after the fix.

## Provenance

Surfaced in task #2265 `epm:results` v1 (§d Needs human eyeball); filed by the #2265 orchestrator session per the workflow-fix routing (orchestrator files, subagents never file/spawn).

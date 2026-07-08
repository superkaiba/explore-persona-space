---
title: Fix torch-before-dotenv in 34 main-side VM scripts (thread-caps invariant test
  red on trunk)
kind: infra
tags: []
created_at: '2026-07-08T13:49:32Z'
has_clean_result: false
origin_prompt: 'surfaced by /issue 1144 Step 9c: test_no_new_torch_before_dotenv_vm_entrypoints
  RED on pristine origin/main with 34 grandfather-escaped offenders; port task fixed
  only its own (issue1090_fu3_figures.py)'
workflow: v1
---
## Overview / Motivation

Surfaced by task #1144's Step 9c gate (epm:test-verdict v1, 2026-07-08): `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is RED on pristine origin/main (verified in a detached sparse scratch worktree at origin/main@586bf7aba1) with 34 offenders — VM figure/fit scripts that import torch/numpy at module top before `explore_persona_space.orchestrate.env.load_dotenv()`, so the shared-VM thread caps (#847/#891) never bind in-process. Most landed via Step-10d surgical/artifact-confirmed checkouts, which bypass pytest gates — the invariant test could not stop them.

Every Step 9c run whose selection includes this test now hits a pre-existing failure that each session must re-classify (the known-red ledger strips it only when the compare oracle is trustworthy; #1144 hit a dirty-root indeterminate and had to run a manual oracle).

## Goal

Make `test_no_new_torch_before_dotenv_vm_entrypoints` PASS on main: insert the conforming `load_dotenv()`-before-torch/numpy preamble in every current offender (mechanical, per-file ~2 lines), so the trunk invariant test is green and Step 9c compares stop carrying this known-red row.

## The offender list (34 scripts, from the pristine-main oracle run — re-derive at implementation time)

Extract fresh via: run `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` at main HEAD and take the violation list. As of origin/main@586bf7aba1 it spans scripts/issue779_* (8), issue811_* (5), issue833_* (8), issue928_mlc_figures.py, issue1073_neardup_sensitivity.py, issue1074_* (4), issue1090_{freeanalysis,fu1,lowlevel}_figures.py, issue1112_debiased_cosine.py, and others (full list in #1144 epm:test-verdict v1 + /tmp/oracle-threadcaps.log at filing time).

## Constraints

- Mechanical preamble insertion only (no logic changes); copy the conforming pattern from a non-flagged script.
- Do NOT extend the grandfathered allowlist — fix the scripts (the allowlist is for the frozen #895 block only).
- Touched-file ruff + the thread-caps test file as the acceptance gate; expect the test to flip green on main.
- Some offenders may carry a late `load_dotenv(` (line N > torch import) — MOVE the call above the import rather than adding a duplicate.

## Provenance

- origin: task #1144 Step 9c gate finding (epm:test-verdict v1); manual pristine-main oracle at origin/main@586bf7aba1.

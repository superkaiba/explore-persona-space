---
title: 'fix #847 torch-before-dotenv violations making test_no_new_torch_before_dotenv_vm_entrypoints
  red on main (fleet-wide Step 9c inheritance)'
kind: infra
tags: []
created_at: '2026-08-30T20:30:52Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2387 round 8: test_no_new_torch_before_dotenv_vm_entrypoints
  is red on origin/main content, naming four scripts/issue2617_* and issue2643_* files
  unchanged by that round; test file byte-identical to main''s. Every session''s Step
  9c gate inherits the red.'
workflow: v1
---
## Goal

Fix the `#847` violations that make
`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
red on `origin/main`, so every session's Step 9c gate stops inheriting a
failure none of them caused.

## The bug

The test is red on main's own content. It names four scripts:

    scripts/issue2617_*
    scripts/issue2643_*

(the exact four are printed in the failure output). Each imports `torch`
before the `dotenv` load the `#847` rule requires at a VM entrypoint.

The test file itself is byte-identical to main's copy, so this is not a test
regression — it is a real rule violation in shipped scripts.

## Why this needs its own task

It is fleet-wide, and it is invisible from where it bites.

Every `/issue` session's Step 9c gate runs the diff-linked test union. Any
session whose diff touches a file mapping to this test inherits a red that
its own payload did not cause. The session then has to distinguish "my change
broke this" from "main was already broken", and the honest answer requires
checking whether the named files are in its own diff. That is exactly the
`#1388` shape: lint-red scripts landed on main broke the Step 9c gate
fleet-wide, and each affected session paid the diagnosis cost separately.

Measured from #2387 round 8 (2026-08-30): the round-8 implementer hit it,
confirmed all four named files were unchanged by its round and present on
origin/main, confirmed the test file matched main byte-for-byte, and correctly
declined to attribute it to its own payload. That diagnosis is the cost being
paid repeatedly.

## Fix

1. For each named script, move the `torch` import after the dotenv load — the
   `orchestrate.env.load_dotenv` path, not a bare `dotenv` import (the
   `--check-dotenv-before-hf-import` lint rule governs the sibling case and the
   same reasoning applies).
2. Confirm `test_no_new_torch_before_dotenv_vm_entrypoints` passes on main
   after the fix, with no change to the test file.
3. Check whether the four scripts have live consumers that depend on
   import-time `torch` availability before deciding the import placement — a
   mechanical move that breaks a caller is not a fix.

## Worth considering while in here

The rule caught these only at the Step 9c gate, i.e. after the scripts had
landed. If the check can run as part of the ordinary no-flags
`workflow_lint.py` run, it would fail at the authoring session instead of at
every later session's gate. That is the difference between one session paying
and every session paying, and it is the same argument the `#2253`
`--check-prod-import-lockfile` rule already won for unresolvable imports.

## Acceptance

- `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
  passes on `main`, test file unmodified.
- Each named script's callers still work (state which were checked and how).
- If the pre-landing check is adopted, a test pins it.

## Provenance

Surfaced by #2387 round 8 while running the diff-linked test union for its
Step 9c pre-check. Not fixed there: the four scripts are outside that round's
scope and folding an unrelated repo-wide fix into a live review round would
contaminate the review.

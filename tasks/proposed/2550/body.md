---
title: 'workflow-fix: scripts/issue2254_firstk_ctxext_sensitivity.py imports numpy
  before load_dotenv, reddening test_shared_vm_thread_caps on main'
kind: infra
tags:
- wf-fix
created_at: '2026-08-24T19:53:03Z'
has_clean_result: false
origin_prompt: 'Surfaced by #2537 round 1+2: three independent reviewers plus the
  implementer each git-probed the same pre-existing-on-trunk red at Step 9c scope;
  confirmed at the repo root on pristine main.'
workflow: v1
---
## Goal

Fix the fleet-wide red on `main`: `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
fails on a pristine `origin/main` checkout because `scripts/issue2254_firstk_ctxext_sensitivity.py`
imports a heavy root at module top with no prior `load_dotenv()`.

## Evidence

Surfaced while running #2537's Step 9c gate scope. Independently reproduced at the
repo root on pristine `main` (not in any worktree, not with any branch changes applied):

```
uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q
```

fails with:

```
AssertionError: NEW heavy-import-before-load_dotenv VM entrypoint(s) — call
explore_persona_space.orchestrate.env.load_dotenv() BEFORE importing any
HEAVY_IMPORT_ROOTS root so the shared-VM thread caps (#847) bind in-process:
  scripts/issue2254_firstk_ctxext_sensitivity.py (module-top heavy import at line 33,
  first load_dotenv( at line None)
```

The offending line is `import numpy as np` at `scripts/issue2254_firstk_ctxext_sensitivity.py:33`,
with no `load_dotenv()` anywhere in the file. The script landed on `main` in #2254 round 5
(commit `fd8222d6f7`, "issue #2254 r5: intrusion-sensitivity recount for the measured-direction
position claim"). It is not on the test's grandfather list.

## Why this needs its own task

The failure is **pre-existing on trunk and foreign to any branch that trips it**. Both
independent reviewers on #2537 round 1 (the Claude `code-reviewer` and the Codex twin) ran
their own git-provenance probes and classified it `pre-existing-on-trunk`; the #2537
implementer hit it a third time. Confirmed again here: `scripts/issue2254_firstk_ctxext_sensitivity.py`
is byte-unchanged from the merge-base through the #2537 branch HEAD, and neither that
script nor `tests/test_shared_vm_thread_caps.py` appears in #2537's round diff.

The cost is per-session, not one-off: any session whose diff maps
`tests/test_shared_vm_thread_caps.py` into its Step 9c selection sees a red gate it did not
cause, then spends reviewer and orchestrator turns proving the red is foreign. That has now
happened at least four times on #2537 alone.

## Scope

1. Add `explore_persona_space.orchestrate.env.load_dotenv()` before the module-top heavy
   import in `scripts/issue2254_firstk_ctxext_sensitivity.py`, following whatever idiom the
   already-compliant `scripts/` entrypoints use (read a sibling that passes the test rather
   than inventing a form). The thread-cap rationale is #847: the caps must bind in-process
   BEFORE the heavy root is imported, so ordering is the whole point — moving the import
   under a `load_dotenv()` call is the fix, not silencing the assertion.
2. Confirm the test passes at the repo root on `main` after the change.
3. Check whether any OTHER `scripts/` entrypoint has landed in the same state since the
   test's grandfather list was last updated — a bounded sweep of the test's own violation
   report over the current tree is enough; an exhaustive proof is not required.

## Acceptance

1. `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q`
   passes on `main`.
2. `scripts/issue2254_firstk_ctxext_sensitivity.py` still runs (the import reorder did not
   break its entrypoint) — a `--help` or equivalent smoke suffices, since this task changes
   import ordering only.
3. If scope item 3 finds further violators, they are either fixed in the same round or named
   explicitly in the report with a reason for deferring.

## Do NOT

Do not add the file to the test's grandfather/exempt list as the fix. The grandfather list
exists for genuinely legacy entrypoints; this script landed after the rule and adding it
would convert a two-line ordering fix into a permanent carve-out, weakening the #847
invariant for every future entrypoint.

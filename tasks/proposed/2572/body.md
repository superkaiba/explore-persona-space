---
title: Fix fleet-wide red test_no_new_torch_before_dotenv_vm_entrypoints on scripts/issue2254_firstk_ctxext_sensitivity.py
kind: infra
tags:
- workflow-fix
created_at: '2026-08-25T10:59:39Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2569 unit-1 implementer and confirmed by the #2569
  orchestrator: scripts/issue2254_firstk_ctxext_sensitivity.py (landed fd8222d6f7)
  does a module-top heavy import with no load_dotenv(), reding the shared VM thread-caps
  invariant test fleet-wide for any session whose Step 9c gate selects it.'
workflow: v1
---
# Fix the fleet-wide red `test_no_new_torch_before_dotenv_vm_entrypoints` on `scripts/issue2254_firstk_ctxext_sensitivity.py`

## Goal

Restore green on the shared VM thread-caps invariant test. It is currently RED
on `main` for every session whose Step 9c gate selects it, which is any session
touching a VM entrypoint.

## The failure

`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
fails on the main-resident `scripts/issue2254_firstk_ctxext_sensitivity.py`
(landed `fd8222d6f7`): the module does a module-top heavy import (numpy / torch /
scipy class) with NO `load_dotenv()` call before it. Verified independently:
`grep -n 'load_dotenv' scripts/issue2254_firstk_ctxext_sensitivity.py` returns
NOTHING.

## Why it matters beyond one file

The test is a workflow INVARIANT, so it rides in the Step 9c selected set for
unrelated sessions. A red invariant test cannot be distinguished by a reviewer
from a red caused by the round's own payload without extra forensics, which is
exactly the ambiguity the payload-attribution rule exists to remove. Leaving it
red trains sessions to wave past a red gate.

## Fix

Add the same one-line fix the #2569 unit-1 implementer applied to its own module
after tripping the identical invariant: call `load_dotenv()` (via
`explore_persona_space.orchestrate.env.load_dotenv`, per the project rule that
new direct-upload / entrypoint scripts use that helper and never bare `dotenv`)
BEFORE the numpy / torch / scipy imports at module top.

## Acceptance

- `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints`
  passes.
- No behavior change to `issue2254_firstk_ctxext_sensitivity.py` beyond import
  ordering plus the dotenv call.
- A quick scan for SIBLING violators: if other main-resident entrypoints trip the
  same invariant, fix them in the same round rather than leaving a partially
  green gate.

## Provenance

Surfaced by the #2569 unit-1 implementer (pre-split build, plan v4) and
independently confirmed by the #2569 orchestrator, 2026-08-25. Not introduced by
#2569's diff: #2569's own module initially tripped the same invariant and was
fixed in `6766e74b1f`. Filed rather than fixed in place because
`issue2254_firstk_ctxext_sensitivity.py` belongs to another task and #2569 has no
mandate to edit it.

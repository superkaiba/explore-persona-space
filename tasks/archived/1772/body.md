---
title: 'workflow-fix: red main — add load_dotenv before heavy import in two figure
  scripts'
kind: infra
tags:
- wf-fix
- wf-fix-fp:redmain-dotenv-tc
created_at: '2026-07-28T21:10:53Z'
has_clean_result: false
origin_prompt: 'Observed by the #779 rb-nuisance-profile inline round 2026-07-28:
  test_no_new_torch_before_dotenv_vm_entrypoints RED on main from scripts/issue1481_coverage_fig.py
  + scripts/issue779_scaling_curve_fig.py'
workflow: v1
---
## Overview / Motivation

`tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is RED on origin/main as of 2026-07-28 ~21:10Z, verified by a fresh bounded pytest run at the repo root (rc=1, 23.7 s). A red main test degrades the `/issue` Step 9c test-verdict gate fleet-wide: every concurrent session must re-classify the failure as pre-existing against the baseline ledger.

## Goal

Restore `test_no_new_torch_before_dotenv_vm_entrypoints` to green on main by adding the missing `load_dotenv()` call before the module-top heavy import in the two offending scripts (the canonical thread-cap / shared-VM discipline), without changing either script's behavior.

## Workflow gap

- **Bug observed:** two experiment figure scripts committed earlier today perform a module-top heavy import (torch/numpy family) with no preceding `load_dotenv()`, which the shared-VM thread-cap invariant test forbids:
  - `scripts/issue1481_coverage_fig.py` (heavy import at line 8, no `load_dotenv(`)
  - `scripts/issue779_scaling_curve_fig.py` (heavy import at line 16, no `load_dotenv(`)
- **Why it is a workflow gap:** the invariant exists because an unguarded heavy import on the shared VM spawns default thread pools before the env caps are applied, oversubscribing the box for every concurrent session. The test is the enforcement; two scripts landed past it.
- **Confidence (filer):** high — reproduced directly.
- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -x -q` → FAILED, assertion lists exactly the two files above (2026-07-28); `git log -1 --oneline -- scripts/issue1481_coverage_fig.py scripts/issue779_scaling_curve_fig.py` → both landed 2026-07-28 (reported commits `0be28be947`, `40887389d4` — cited as reported by the observing round, not independently rev-parsed at filing).

## Proposed change (candidate diff sketch — refine in planning)

In each offending script, before the module-top heavy import, insert the canonical prelude used by every other VM entrypoint (see any `scripts/issue1482_*.py` for the exact shape):

```
+ PROJECT_ROOT = Path(__file__).resolve().parent.parent
+ sys.path.insert(0, str(PROJECT_ROOT / "src"))
+ from explore_persona_space.orchestrate.env import load_dotenv
+ load_dotenv()   # thread caps + credentials BEFORE numpy/torch
  import numpy as np   # (existing heavy import)
```

Then re-run the named test to green.

## Scope / surfaces

- Primary targets: `scripts/issue1481_coverage_fig.py`, `scripts/issue779_scaling_curve_fig.py`
- Do NOT change plotting behavior, output paths, or figure content.
- Re-run `uv run pytest tests/test_shared_vm_thread_caps.py -q` plus `uv run python scripts/workflow_lint.py` before landing.

## Constraints / invariants

- Workflow-surface-adjacent fix only; no experiment-logic change.
- `scripts/workflow_lint.py` (no flags) passes; ruff clean on touched files.

## Provenance

- workflow_fix_target: scripts/issue1481_coverage_fig.py,scripts/issue779_scaling_curve_fig.py
- fingerprint: redmain-dotenv-thread-caps-20260728

Surfaced by the #779 `rb-nuisance-profile` inline round (2026-07-28), which observed the red while its own payload lint gate PASSed (its file is not in the violation list). Filed by the orchestrator, not by that round.

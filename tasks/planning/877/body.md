---
title: 'Fix pre-existing thread-caps test failure: issue779_layer_sweep.py missing
  load_dotenv-before-torch'
kind: infra
tags: []
created_at: '2026-07-02T17:37:39Z'
has_clean_result: false
origin_prompt: 'Surfaced by #876 implementer: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  fails pre-existing on main via scripts/issue779_layer_sweep.py (no load_dotenv before
  torch import, no grandfather entry)'
---
## Overview / Motivation

Surfaced by task #876's implementer (2026-07-02): `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` FAILS pre-existing on pristine `main` — `scripts/issue779_layer_sweep.py` (commit fd1d5512e8, task #779) imports torch/numpy at module top with no `orchestrate.env.load_dotenv()` call before them and no grandfather entry. Every task whose Step 9c touched-file subset includes this workflow-invariant test inherits the red.

## Goal

Make `test_no_new_torch_before_dotenv_vm_entrypoints` pass on `main`: add `orchestrate.env.load_dotenv()` before the torch import in `scripts/issue779_layer_sweep.py` (preferred — the shared-VM thread-cap rule requires load_dotenv before torch import so OMP caps freeze correctly), OR add a grandfathered allowlist entry with an inline reason (the script self-caps via `torch.set_num_threads` at ~:486).

## Constraints

- One-liner class; do not change the script's behavior otherwise.
- Coordinate with the live #779 session (task at followups_running) — the file is on `main`; its worktree inherits on rebase.
- `uv run pytest tests/test_shared_vm_thread_caps.py` passes after the fix; ruff clean.

## Provenance

- Surfaced by: #876 implementer round 1 (pre-existing-on-main test failure, 0 introduced by the #876 diff).

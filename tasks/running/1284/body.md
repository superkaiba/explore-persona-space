---
title: 'daily-fix: issue1092 load_dotenv before heavy imports'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-12T06:52:47Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-11 problem sweep (route 2): pre-existing red on main:
  test_no_new_torch_before_dotenv_vm_entrypoints flags both issue1092 scripts (heavy
  module-top imports before/without load_dotenv), taxing every Step 9c gate run with
  a pristine-oracle classification round'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-11 from the problem sweep. NOT a workflow-surface gap — experiment-code hygiene on `main` that taxes every Step 9c test gate.

## Goal

Make `scripts/issue1092_bridge_refit.py` and `scripts/issue1092_figures.py` satisfy the torch/heavy-import-after-dotenv entrypoint contract, turning `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` green on `main`.

## Problem

- **Bug observed:** the pre-existing red on `main` — `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` flags `scripts/issue1092_bridge_refit.py` (module-top heavy import at line 27, no `load_dotenv` call anywhere in the file) and `scripts/issue1092_figures.py` (module-top numpy at line 64; `load_dotenv()` only at line 2470 inside main). At least two Step 9c gate runs today (#1257 07:55Z, #1273 07:54Z) each burned a pristine single-file oracle round classifying this red as pre-existing-on-main.
- **Why file it:** every future Step 9c run pays the oracle tax until main is green.
- verified-at-filing: `grep -c load_dotenv scripts/issue1092_bridge_refit.py` → 0; `grep -n "load_dotenv\|import numpy" scripts/issue1092_figures.py` → numpy at 64, load_dotenv at 2470/2472 (2026-07-12). No open task mentions issue1092 (`grep -rln issue1092 tasks/proposed/*/body.md` → none).

## Proposed change

Add the standard entrypoint preamble (`from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()`) before the first heavy import in both scripts (the #1145 `issue928_mlc_figures.py` fix is the worked precedent for this exact test), keeping behavior otherwise identical; confirm the target test passes on the branch.

## Scope / surfaces

- `scripts/issue1092_bridge_refit.py`, `scripts/issue1092_figures.py` (experiment scripts — minimal reorder only, no logic change).
- Gate: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -x` green.

## Constraints / invariants

- No behavior change beyond import/env ordering; ruff passes.

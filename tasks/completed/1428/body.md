---
title: Fix torch-before-dotenv in 10 main-resident figure scripts (fleet thread-caps
  test red)
kind: infra
tags: []
created_at: '2026-07-16T17:09:28Z'
has_clean_result: false
origin_prompt: 'Flagged by #1415 crash-fix implementer round 2026-07-16: tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints
  fails on main-resident figure scripts from #1092/#1315/#1336/#1345/#952 — needs
  a separate main-side infra fix.'
workflow: v1
---
## Overview / Motivation

Fleet-wide invariant test `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` is RED on pristine main: a fresh batch of main-resident figure/analysis scripts imports torch/matplotlib-heavy modules at module top BEFORE `load_dotenv()` (the #847 shared-VM thread-cap hook), postdating the prior offender-batch fixes (#1284, #1319, #1378). Surfaced by the #1415 crash-fix implementer round (2026-07-16); any session running this test (Step 9c gates, Step 10d mapped invariant-test legs' baselines) sees a red baseline.

## Goal

Make `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` PASS on main by moving `load_dotenv()` (from `explore_persona_space.orchestrate.env`) above the first heavy import in each offender script — the same recipe as #1284/#1319/#1378.

## Workflow gap / evidence

- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q` on the main repo root → 1 failed (2026-07-16T16:50Z), offenders enumerated in the assert:
  - scripts/issue1092_inline_compose_chain_figure.py (heavy import line 17)
  - scripts/issue1092_inline_fair_comparison_agreement_fig.py (line 23)
  - scripts/issue1092_inline_fair_comparison_fig.py (line 28)
  - scripts/issue1092_offvm_refit_figures.py (line 28)
  - scripts/issue1315_bare_plots.py (line 25)
  - scripts/issue1336_analyzer_figures.py (line 26)
  - scripts/issue1336_dedup_figure.py (line 20)
  - scripts/issue1336_increments_figure.py (line 20)
  - scripts/issue1345_clear_figs.py (line 44)
  - scripts/issue952_divtrain_figures.py (line 24)

## Proposed change

Per offender: insert `from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()` before the first torch/matplotlib/numpy-heavy import (match the pattern the #1284/#1319/#1378 fixes used; never bare `dotenv` — lint `--check-dotenv-before-hf-import`). Re-run the test to green. No behavior change beyond import order + the thread-cap setdefaults engaging.

## Constraints / invariants

- Experiment-script surface only; no workflow-surface edits.
- `uv run ruff check` clean on touched files; the single test green on main after merge.

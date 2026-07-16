---
title: Fix torch-before-dotenv in issue1073/issue1092 figure scripts (fleet test_shared_vm_thread_caps
  failure)
kind: infra
tags: []
created_at: '2026-07-16T05:09:55Z'
has_clean_result: false
origin_prompt: '#1315 fu-r2 implementer prose follow-up: two MAIN-side test_no_new_torch_before_dotenv_vm_entrypoints
  violators (issue1073_fig_linear_nonlinear.py, issue1092_inline_operator_figure.py)
  fail the fleet test; 2-line load_dotenv fix each'
workflow: v1
---
## Overview / Motivation

Filed from a prose follow-up surfaced by the #1315 fu-r2 implementer (2026-07-16). NOT a workflow-surface fix (experiment scripts) — a plain 2-line-each code fix that clears a fleet-wide failing test.

## Goal

Make `scripts/issue1073_fig_linear_nonlinear.py` and `scripts/issue1092_inline_operator_figure.py` call `explore_persona_space.orchestrate.env.load_dotenv()` BEFORE their heavy torch/transformers imports, clearing the fleet-wide `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` failure.

## Problem

`test_no_new_torch_before_dotenv_vm_entrypoints` fails on `origin/main` naming exactly these two VM entrypoint scripts (verified 2026-07-16: violator blobs byte-identical between main and the issue-1315 branch; every branch's Step-9c mapped-scan carries the failure as a baseline exclusion). A torch-before-dotenv importer freezes its thread pool before the shared-VM caps apply (#847/#891/#779 — `.claude/rules/code-style.md` § Shared-VM CPU thread caps).

## Fix shape (per script, ~2 lines)

Add at the top, before any torch/transformers/matplotlib-heavy import (the #847 pattern used by the sibling fix in `scripts/issue1315_lr1e5_plots.py` @ commit ff3f319ba2 on issue-1315):

```
from explore_persona_space.orchestrate.env import load_dotenv
load_dotenv()
```

## Acceptance

- `uv run pytest tests/test_shared_vm_thread_caps.py -q` → 19/19 pass on main (0 named violators).
- `uv run ruff check` clean on both touched scripts.
- No behavioral change to the figures the scripts produce.

## Constraints

- Experiment-code scripts: fix the import ordering only; do not refactor.
- verified-at-filing: `uv run pytest tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints -q` on the issue-1315 worktree names exactly these two scripts (1 failed), and `git diff origin/main -- scripts/issue1073_fig_linear_nonlinear.py scripts/issue1092_inline_operator_figure.py` is empty (blobs identical to main) (2026-07-16).

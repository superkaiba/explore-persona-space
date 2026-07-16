---
title: 'daily-fix: torch-before-dotenv in 2 issue scripts (#847)'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-15T06:51:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-14 problem sweep (route 2): torch imported before load_dotenv
  in issue1092_inline_figures.py:41 and issue823_weightspace_calibration.py:41, the
  #847 torch-before-dotenv class enumerated by test_no_new_torch_before_dotenv_vm_entrypoints'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-14 run: routed from a prose follow-up parked under the recursion guard on task #1311 (#847 torch-before-dotenv class). Experiment-code fix (NOT workflow surface), /daily route-2 channel.

## Goal

call orchestrate.env.load_dotenv() before heavy imports (torch) in the named VM entrypoints (#847 class)

## Bug

- **Observed:** torch imported before load_dotenv in issue1092_inline_figures.py:41 and issue823_weightspace_calibration.py:41, the #847 torch-before-dotenv class enumerated by test_no_new_torch_before_dotenv_vm_entrypoints
- verified-at-filing: `grep -n "^import\|^from\|load_dotenv"` on both files -> `import torch` at :41 in each with no prior `load_dotenv()` (per-target: 1 hit each) (2026-07-15). `tests/test_no_new_torch_before_dotenv_vm_entrypoints` is the enumerating pin (check its current allowlist/failures during planning). Retraction re-check on #1311 events after the park ts: none.

## Proposed change

Move/insert `from explore_persona_space.orchestrate.env import load_dotenv; load_dotenv()` ahead of the heavy imports in both scripts, matching the #847 convention; update or satisfy the enumerating pin test.

## Constraints

- No behavior change beyond import order; smoke: `uv run python -c "import ..."` both modules + the pin test passes.

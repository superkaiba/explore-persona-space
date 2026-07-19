---
title: Fix heavy-import-before-dotenv in issue1092 figure script (fleet-wide Step
  9c thread-caps red)
kind: infra
tags: []
created_at: '2026-07-18T15:50:37Z'
has_clean_result: false
origin_prompt: 'experiment-implementer report on #1426 sampled-rollout round (d):
  pre-existing fleet-wide step-9c red on scripts/issue1092_prefixend_monitoring_combined_fig.py
  (heavy import at line 19 precedes any load_dotenv( call; landed at e050ead51c)'
workflow: v1
---
## Overview / Motivation

Auto-filed from an experiment-implementer report on task #1426 (round sampled-rollout-robustness, 2026-07-18): `tests/test_shared_vm_thread_caps.py::test_no_new_torch_before_dotenv_vm_entrypoints` FAILS fleet-wide on `scripts/issue1092_prefixend_monitoring_combined_fig.py` — the file imports matplotlib (line 19) and numpy (line 20) before any `load_dotenv(` call (0 `load_dotenv(` calls in the file). Every branch's Step 9c mapped-test gate that touches a thread-caps-mapped file now trips on this pre-existing red (the #1426 round confirmed it fails identically with its own diff stashed).

## Goal

Fix the import order in `scripts/issue1092_prefixend_monitoring_combined_fig.py` so the thread-caps gate passes: add the project wrapper `from explore_persona_space.orchestrate.env import load_dotenv` + a `load_dotenv()` call BEFORE the heavy imports (the same shape as scripts/issue1426_analyzer_figures.py / issue1426_pooled_gradient.py, fixed in 62f983e36e), then confirm `uv run pytest tests/test_shared_vm_thread_caps.py` is green.

## Workflow gap

- **Bug observed:** pre-existing test red on main blocks every session's Step 9c gate whose diff maps to test_shared_vm_thread_caps.py.
- **Why filed as infra:** experiment-code file (out of workflow-fix auto-spawn scope) but it breaks the shared mechanical gate; 2-line fix, no science impact (figure script for completed #1092).
- **Confidence (emitter):** high
- verified-at-filing: `sed -n '1,25p' scripts/issue1092_prefixend_monitoring_combined_fig.py | grep -n 'import\|load_dotenv'` → matplotlib line 19 / numpy line 20, 0 `load_dotenv(` calls in file (grep -c = 0); no open task mentions the file (repo-wide task-body grep, 2026-07-18)

## Constraints

- Fix the SOURCE file only; run `uv run pytest tests/test_shared_vm_thread_caps.py` + ruff on the touched file.

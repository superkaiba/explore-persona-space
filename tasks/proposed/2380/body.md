---
title: 'workflow-fix: Step 9c selector misses importlib-loading test files — bare
  module-name literals map to no selector arm'
kind: infra
tags:
- wf-fix
created_at: '2026-08-19T03:06:31Z'
has_clean_result: false
origin_prompt: 'workflow-fix-candidate from task #2377 round-1 implementer: tests/test_janitor_tmp_scratch_sweep.py
  (125 tests) not selected for a diff editing scripts/clean_experiment_downloads.py
  — importlib _load("clean_experiment_downloads") carries no .py-suffixed literal,
  so stem-map/import-map/glob-scan arms never map it'
workflow: v1
---
# Step 9c selector misses test files that load their target module via importlib (bare module-name literals)

## Goal

Close the Step 9c test-selector blind spot for test files that load their
target script via importlib with a BARE module name: add a selector arm (or
extend the stem-map arm) in `scripts/select_step9c_tests.py` that matches
bare module-name string literals of edited scripts inside test files (the
`_load("clean_experiment_downloads")` idiom), or an explicit test↔module pin
registry for importlib-loading test files.

## Why

Observed on task #2377 (base origin/main `8d431ff16c`, n_tests=147):
`tests/test_janitor_tmp_scratch_sweep.py` — the primary behavior suite
(125 tests) for `scripts/clean_experiment_downloads.py` — was NOT selected
for a diff editing exactly that script. Root cause: the test loads its
target via importlib with a bare module name (`_load("clean_experiment_downloads")`),
so the stem-map / import-map / glob-scan arms — which key on `.py`-suffixed
literals or import statements — never map it. Any future
`clean_experiment_downloads.py` regression covered only by that suite passes
the Step 9c gate unselected. The #2377 implementer ran the suite manually
(green), but the gate itself has the gap.

## Acceptance criteria

1. For a diff touching `scripts/clean_experiment_downloads.py`, the selector
   selects `tests/test_janitor_tmp_scratch_sweep.py` (the driving instance).
2. The arm is general: a test file carrying a bare module-name string literal
   matching an edited script's stem (importlib `_load(...)` idioms) is
   mapped — or, if a pin-registry design is chosen instead, the registry is
   lint-enforced so new importlib-loading test files must register.
3. No over-selection regression: common English words as module stems must
   not glob-match unrelated tests (guard the arm on the literal appearing in
   a string-literal/loader context, or on a minimum stem specificity).
4. Regression test pinning the new arm (fire + non-fire branches).

## Provenance

Surfaced by the task #2377 round-1 implementer (workflow-fix-candidate block
in its `epm:results` v1 marker, confidence: medium). Filed by the #2377
orchestrator session per `.claude/rules/workflow-fix-on-bug.md` (auto-file +
spawn). Dedup checked against #2297/#2309/#2310/#2315 (all gate-LAUNCHER /
marker-contract fingerprints on the same file — distinct bugs).

workflow_fix_target: scripts/select_step9c_tests.py

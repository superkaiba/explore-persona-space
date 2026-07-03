---
title: 'workflow-fix: select_step9c_tests must include touched tests/*.py in subset'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e3013d01ea8c
created_at: '2026-07-02T09:51:31Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation on #851 Step 9c: select_step9c_tests.py --base
  main omitted the branch''s own new test files (tests/test_graded_judge.py, tests/test_eval_margin.py)
  from the printed pytest command with no untested-touched-file WARN; orchestrator
  appended them manually.'
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #851 (Step 9c test-verdict gate, 2026-07-02).

## Goal

`scripts/select_step9c_tests.py` must include touched `tests/*.py` files from the branch diff in its printed pytest command.

## Workflow gap

- **Bug observed:** on issue-851 (`git diff main...HEAD` touching `tests/test_graded_judge.py` + `tests/test_eval_margin.py` among 7 files), `uv run python scripts/select_step9c_tests.py --base main` printed only the 32-file workflow-invariant subset — the branch's OWN new test files were absent from the run list, and stderr carried NO `untested touched file:` WARN for them (or for the touched src/scripts files they cover).
- **Why it is a workflow gap:** the Step 9c gate's `touched` scope exists to verify the diff; omitting the diff's own test files means a branch whose new tests fail would PASS the gate unless the orchestrator manually appends them (as #851's orchestrator did). Silent under-selection of the very tests written for the change defeats the gate.
- **Confidence (emitter):** high (observed live; the printed command is in #851's `epm:test-verdict` marker).

## Proposed change (candidate diff sketch — refine in planning)

```
+ In select_step9c_tests.py's subset computation:
+   touched = git diff --name-only <base>...HEAD
+   test_files = [f for f in touched if f.startswith("tests/") and f.endswith(".py")]
+   subset = workflow_invariant_subset | mapped_tests_for_touched_source_files | set(test_files)
+ Also: emit the documented `untested touched file: <path>` WARN for touched
+ source files with no mapped/co-named test, which did not fire on #851.
+ Add a regression test: a diff adding tests/test_foo.py must appear in the
+ printed command.
```

## Scope / surfaces

- Primary target: `scripts/select_step9c_tests.py`
- Secondary: its pinning test (add one if none exists).

## Constraints / invariants

- Workflow-surface only. `scripts/workflow_lint.py` passes; ruff on touched files passes.
- The printed command shape (single `uv run pytest <files> -v --tb=short` line) is consumed verbatim by the /issue Step 9c prose — keep it stable.

## Provenance

- workflow_fix_target: scripts/select_step9c_tests.py
- fingerprint: e3013d01ea8c

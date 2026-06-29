---
title: 'daily-fix: living_docs.py check degrade on bad task, not crash nightly pass'
kind: infra
tags:
- wf-fix
- wf-fix-fp:1e8c70da472b
- daily-auto-filed
created_at: '2026-06-29T06:38:43Z'
has_clean_result: false
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-06-28 problem sweep (route 2 — behavior change, files for independent review).

## Goal

Make `scripts/living_docs.py check` degrade gracefully (skip + warn) on a registry/filesystem-inconsistent task instead of hard-crashing the whole pass.

## Workflow gap

- **Bug observed:** Tonight `uv run python scripts/living_docs.py check` raised `FileNotFoundError: task #696 registry says 'tasks/approved/696' but that dir is missing` (inside `_relates_to_index` → `tw.find_task_path(696)`) and exited 1, producing NO drift report at all.
- **Why it is a workflow gap:** `living_docs.py check` is now a LOAD-BEARING nightly `/daily` step (folded in from `/weekly` per #713). A single drifted task silently kills the entire Living-docs drift pass every night — exactly the silent-failure class the protocol exists to catch. The drift linter should be resilient to one bad task, not all-or-nothing.
- **Confidence (emitter):** high (reproduced tonight)

## Proposed change (refine in planning)

- In `_relates_to_index` (and any sibling that resolves task paths), wrap the per-task `find_task_path(...) / "body.md"` read in a try/except that, on `FileNotFoundError` / missing-dir, emits a warning naming the task and continues — so the check returns a real drift report and a nonzero/zero exit reflecting actual drift, not a crash.
- The bad task itself should surface as a drift finding (registry/fs inconsistency), not as an exception.

## Scope / surfaces

- Primary: `scripts/living_docs.py`
- Add a test (a fixture registry pointing at a missing dir) asserting `check` warns + continues rather than raising.

## Constraints / invariants

- Workflow-surface only; do NOT mutate task state to "fix" the bad task — just degrade gracefully (the registry repair is the sibling task on `scripts/task.py`).
- `scripts/workflow_lint.py --check-references` + ruff on touched files pass.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/living_docs.py
- fingerprint: 1e8c70da472b

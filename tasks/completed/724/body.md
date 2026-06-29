---
title: 'daily-fix: task.py audit --repair + reconcile 13 drifted registry tasks'
kind: infra
tags:
- wf-fix
- wf-fix-fp:b8b8ac361292
- daily-auto-filed
created_at: '2026-06-29T06:38:31Z'
has_clean_result: false
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-06-28 problem sweep (route 2 — behavior change, files for independent review). Found while running the nightly `living_docs.py check` drift pass.

## Goal

Add a registry repair/rebuild mode to `task.py audit` (and the underlying `task_workflow`) that reconciles `tasks/REGISTRY.json` to the on-disk task layout, and reconcile the 13 currently-drifted tasks.

## Workflow gap

- **Bug observed:** `uv run python scripts/task.py audit` is REPORT-ONLY. Tonight it reported 13 registry/filesystem inconsistencies it cannot fix: `#696` registry path says `tasks/approved/696` but the task is at `tasks/completed/696`; and `#698, #699, #700, #701, #702, #703, #704, #705, #706, #707, #708, #709` are on disk (mostly `completed/`, two `archived/`, one `planning/`) but absent from the registry entirely. Confirmed: `697`/`715` are present in `REGISTRY.json` but `706` is not.
- **Why it is a workflow gap:** `task.py` is the canonical task-state API and `REGISTRY.json` is the index the dashboard list view + every registry-reader consumes. Drifted/unregistered terminal tasks are permanently invisible there (terminal tasks are never re-mutated, so they never self-heal), and the stale `#696` PATH crashes `living_docs.py check` (now a load-bearing nightly `/daily` step per #713). `audit` names the problem but offers no reconcile path.
- **Likely cause:** the documented concurrent-repo-root-committer hazard — a `git pull --rebase` flattened away a `REGISTRY.json` update from a parallel session (the same class as #711). Worth confirming.
- **Confidence (emitter):** medium

## Proposed change (refine in planning)

- Add `task.py audit --repair` (or a `task_workflow.reconcile_registry()` reachable via the canonical API, flock-guarded + single commit) that, for each drift class: rewrites a stale `path` to where the task actually is on disk, and adds an on-disk-but-unregistered task to the registry with its disk status. Report what it changed; never delete a task.
- Run it once to reconcile the current 13.
- Consider a watcher/cron hook so future drift is auto-reconciled (or at least surfaced) rather than silently accumulating.

## Scope / surfaces

- Primary: `scripts/task.py`, `src/explore_persona_space/task_workflow.py`
- Add/extend a test under `tests/test_task_workflow*.py` pinning the reconcile behavior.

## Constraints / invariants

- Workflow-surface only. Registry reconcile points entries to where tasks ALREADY are on disk — it must never move a task or change a task's status.
- `scripts/workflow_lint.py --check-references` + ruff on touched files pass.
- Runs under `EPM_WORKFLOW_FIX_SESSION=1` — do NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: scripts/task.py, src/explore_persona_space/task_workflow.py
- fingerprint: b8b8ac361292

---
title: 'workflow-fix: clean_experiment_downloads degrades gracefully off-main'
kind: infra
tags:
- wf-fix
- wf-fix-fp:4928d4778b51
created_at: '2026-07-03T10:53:33Z'
has_clean_result: false
origin_prompt: 'Orchestrator observation on #841 att-6: between-phase incremental
  cleanup crashes on pod/GCE issue-N clones (repo_root main-branch guard via _active_consumer_protected_issues
  -> tasks_dir); the CLAUDE.md-prescribed between-phase cleanup silently never runs
  on remote lanes. Fix: skip the cross-task protected-issues read off-main with a
  logged WARN; keep VM behavior unchanged.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator observation on task #841 (attempt-6 GCP run log).

## Goal

Make `clean_experiment_downloads.py` degrade gracefully on a non-main checkout (pod/GCE `issue-<N>` clone): skip the tasks_dir-dependent protected-issues read instead of crashing on `repo_root()`'s main-branch guard, so the CLAUDE.md-prescribed between-phase incremental cleanup actually runs on remote lanes.

## Workflow gap

- **Bug observed:** on #841's attempt-6 GCP instance, the between-phase `clean_experiment_downloads.py <N> --incremental --apply` crashed: `_active_consumer_protected_issues` (line ~451) → `tasks_dir()` → `repo_root()` → `_ensure_managed_main_worktree` raised `RuntimeError: primary checkout /workspace/eps-issue-841 is on 'issue-841' and has no local main branch...`. The dispatcher survived (cleanup step non-fatal) but the cleanup silently never runs on ANY pod/GCE lane.
- **Why it is a workflow gap:** CLAUDE.md § Disk hygiene REQUIRES multi-phase runs to call the incremental cleanup between phases ("the run knows the phase is done"), and pods/GCE clones always sit on `issue-<N>` branches — so the prescription and the implementation contradict on every remote lane. The protected-issues read exists to protect OTHER tasks' caches on the SHARED VM; on a single-task remote instance there are no sibling consumers, so skipping that read (with a logged note) is safe and correct.
- **Confidence (emitter):** high (live traceback on #841 att-6; root cause read directly from task_workflow.py's branch guard).

## Proposed change (candidate diff sketch — refine in planning)

- In `_active_consumer_protected_issues` (or its caller), wrap the `tasks_dir()` resolution: on the specific `RuntimeError` from the main-branch guard (or any repo_root resolution failure), log one WARN ("non-main checkout — skipping cross-task protected-issues read; single-task remote instance assumed") and return an empty protected set, so the cleanup proceeds scoped to the named issue's own caches.
- Keep the VM behavior unchanged (main checkout resolves as today).
- Add a regression test: monkeypatch repo_root to raise the guard error; assert clean_issue_downloads still runs and only touches the named issue's cache dirs.

## Scope / surfaces

- Primary target: `scripts/clean_experiment_downloads.py`
- Grep before editing: `grep -n "tasks_dir\|repo_root" scripts/clean_experiment_downloads.py` — cover every resolution site in the file, not just line 451.

## Constraints / invariants

- Workflow-surface only. NEVER weaken the store//eval_results protection or the nested-store parity guard; the skip applies ONLY to the cross-task protected-issues enumeration.
- Existing tests stay green; add the regression test.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/clean_experiment_downloads.py
- fingerprint: 4928d4778b51

Surfaced observation (verbatim log excerpt, #841 att-6): "File scripts/clean_experiment_downloads.py, line 451, in _active_consumer_protected_issues → base = tasks_dir() → ... RuntimeError: primary checkout /workspace/eps-issue-841 is on 'issue-841' and has no local `main` branch to route task.py writes through; create `main` (or check it out) before running task.py." followed by "[phase=stage0] start" (dispatcher survived; cleanup never ran).

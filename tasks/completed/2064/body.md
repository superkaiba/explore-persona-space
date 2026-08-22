---
title: 'workflow-fix: registry drift at an allocated id breaks task.'
kind: infra
tags:
- wf-fix
- wf-fix-fp:68326b22ed2a
- daily-auto-filed
created_at: '2026-08-04T06:51:16Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): tasks/proposed/2052 existed
  on disk with no REGISTRY entry, so the id allocator re-issued 2052 and every task.py
  new in the fleet failed with a bare FileExistsError (taking file_infra_task.py down
  with it); there is no FileExistsError handler on that path (0 hits in task.py and
  task_workflow.py).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep (miner1, session 201e2896, task #1482).

## Goal

A `tasks/REGISTRY.json` entry missing for an on-disk task folder must not break `task.py new` fleet-wide: detect the drift at the allocation site and self-heal (or fail with the repair command), instead of re-issuing the same colliding id to every caller.

## Workflow gap

- **Bug observed:** task folder `tasks/proposed/2052` existed on disk with no REGISTRY entry, so the id allocator kept re-issuing 2052 and **every `task.py new` in the fleet failed** with `FileExistsError: ... tasks/proposed/2052`, taking `scripts/file_infra_task.py` down with it ("`task.py new` exited 1; task NOT filed"). Two firing events at 2026-08-03T20:10:50Z and 20:11:58Z (session 201e2896, rows 3194/3198); repaired in-session by `task.py audit --repair --apply` → `AUDIT PASS` at 20:11:58Z.
- **Why it is a workflow gap:** the allocator derives the next id from `registry["highest_id"]` (`task_workflow.py:815`), so a lost registry write makes the on-disk folder invisible to allocation while remaining visible to `mkdir`. There is no handler on that path: `grep -c FileExistsError scripts/task.py src/explore_persona_space/task_workflow.py` → **0 / 0**. The failure is fleet-wide (every filer, every session) and silent as to cause — the traceback names a path collision, not a registry drift, so each caller has to rediscover `audit --repair`. The once-daily registry-drift observer pass (#1439) is report-only and ran too slowly to matter here.
- **Confidence (emitter):** high for the symptom + the absent handler; the ROOT CAUSE is open.
- verified-at-filing: `grep -c 'FileExistsError' scripts/task.py src/explore_persona_space/task_workflow.py` → `0` and `0` (2026-08-04) — absence claim, verified in both plausible target files. `grep -n 'highest_id' src/explore_persona_space/task_workflow.py` → the allocator reads `registry.get("highest_id", 0)` at L815 and defaults `{"highest_id": 0, "tasks": {}}` at L774. `task.py audit --repair` exists (`scripts/task.py:1026` `cmd_audit`, flag at L1724). Registry state now: `task.py audit` → PASS (the in-session repair held).
- unverified hypothesis — verify at plan time: HOW the registry write was lost. `task.py new` holds `flock` on `~/.task-workflow/lock` for the whole mutation, so a concurrent-write race should be impossible — meaning either the folder was created outside `task.py` (a hand-`mkdir`, an interrupted mutation between `mkdir` and the registry commit), or the flock was not held across both steps. The planner should establish this before choosing between self-heal and fail-loud; a self-heal that papers over a genuine lost-write bug would hide it.

## Proposed change (candidate sketch — refine in planning)

```
at the allocation/creation site, on FileExistsError at the allocated id:
  - detect the specific drift (folder present, no registry entry)
  - EITHER auto-run the existing reconcile (audit --repair --apply) once and
    retry the allocation, OR fail with a message naming the exact repair
    command instead of a bare FileExistsError traceback
  - either way, log the drift so the root cause stays visible
```

Ordering note for the planner: fail-loud-with-the-command is the conservative option and is strictly better than today; auto-repair is the convenient one but should wait on the root-cause finding above.

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (allocation + registry write), `scripts/task.py` (the `new` entrypoint's error surface).
- Both are named workflow surface despite the general `src/**` exclusion.

## Constraints / invariants

- `task.py` mutations stay flock'd and commit once; a repair-then-retry must not double-commit or widen the lock scope.
- Never allocate over an existing on-disk folder.
- `uv run pytest tests/test_task_workflow*.py tests/test_workflow*.py` passes.

## Provenance

- fingerprint: 68326b22ed2a

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: PLACEHOLDER

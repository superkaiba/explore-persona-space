---
title: 'daily-fix: re-resolve task path at marker-append time'
kind: infra
tags:
- wf-fix
- wf-fix-fp:c26845d1b9e8
- daily-auto-filed
created_at: '2026-07-20T06:48:10Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-19 problem sweep (route 2): cached pre-move path appended
  into old status folder (dup #1524 husk)'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-19 (route 2) from transcript-mined problems (see evidence in ## Provenance).

## Goal

Harden marker appends in `src/explore_persona_space/task_workflow.py` so a cached pre-move task-folder path cannot append into an OLD status folder — re-resolve the folder path from the registry UNDER the flock at append time (or assert the target dir is the registry-current one).

## Workflow gap

- **Bug observed:** a mid-run `epm:clarify` append (commit 950079a7) landed in `tasks/proposed/1524` AFTER the folder had moved to `tasks/completed/1524`, creating a duplicate-folder husk on origin/main that the post-merge guard flagged and a session had to hand-remove via scratch worktree (preserving the unique event row first).
- **Why it is a workflow gap:** a concurrent session holding a cached path can recreate an old status folder on append; the husk-reap cron mitigates but the append-side race remains open.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'def _append_jsonl_line' src/explore_persona_space/task_workflow.py` → :3961 (context read: appends to the PATH IT IS GIVEN; no registry re-resolve at append time under the flock — absence claim about the guard, in-target read is the evidence). Incident: session 30ba385e (task #1524) @ 01:15 UTC 2026-07-19, duplicate `tasks/proposed/1524` husk on origin/main from commit 950079a7.

## Proposed change (candidate diff sketch — refine in planning)

(none — sketch: inside the flock in the marker-append path, re-resolve `find(N)`/registry path; if it differs from the caller's path, append to the registry-current events.jsonl and warn)

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (+ `scripts/task.py` if the CLI caches paths)
- Tests: `tests/test_task_workflow*.py`.

## Constraints / invariants

- Workflow-surface rules apply where the target is workflow surface; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- Recursion guard applies where tagged wf-fix (workflow_fix_target Provenance line below).

## Provenance

- sha-verify (filing-time, #1467): `950079a7` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.
- sha-verify (filing-time, #1467): `30ba385e` cited in commit context does NOT resolve as a commit in this repo at filing time — treat as a transcript/session reference, not a commit.

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: 573c6a31988a

Mined evidence: post-merge guard output 'DUPLICATE folders on origin/main: tasks/proposed/1524 — scratch-worktree removal needed' (session 30ba385e, 2026-07-19).

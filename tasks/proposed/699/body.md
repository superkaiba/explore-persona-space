---
title: 'workflow-fix: atomic post_event append (events.jsonl partial-line corruption)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:postevent-atomic-append-corruption
created_at: '2026-06-28T06:44:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-06-27 problem sweep: #653 session b8d3dbc3 — non-atomic
  post_event left a 129-byte partial JSON line that crashed every events.jsonl reader
  + dashboard and stalled /issue resume >2h'
---
## Overview / Motivation

Auto-filed by the `/daily` problem sweep (2026-06-27) from a workflow-surface bug
observed in the #653 autonomous session (transcript b8d3dbc3).

## Goal

Make `task_workflow.post_event` crash-safe so a killed/OOM'd/cron-timed-out writer
cannot leave a truncated JSON line that breaks every downstream reader.

## Workflow gap

- **Bug observed:** `post_event` appends an event line via `with path.open("a")`
  under flock, but the write is NOT atomic against SIGKILL/OOM/cron-timeout
  mid-write. A watcher killed mid-append left a 129-byte partial JSON line
  (`BAD line 374: Unterminated string`) in `tasks/.../653/events.jsonl`, which
  then crashed every `task.py view` / `list_events` / dashboard read and stalled
  the `/issue 653` resume for >2h.
- **Why it is a workflow gap:** `events.jsonl` is canonical task-workflow state;
  a single non-atomic append makes the entire task (and the dashboard) unreadable.
- **Confidence (emitter):** high (reproduced corruption + reader crash in-session).

## Proposed change (refine in planning)

Two complementary fixes (planner picks one or both):
1. **Atomic append:** build the full `json.dumps(event) + "\n"` in memory and
   write it in a single `os.write(fd, ...)` (POSIX guarantees an atomic write
   for a buffer under PIPE_BUF; for larger lines, write to a temp + append, or
   `O_APPEND` single-syscall write) so a kill mid-write never leaves a partial line.
2. **Tolerant reader:** `list_events` / readers skip (and log) an unterminated
   trailing line instead of raising, so an already-corrupted file is still readable.

Add a concurrency/crash regression test under `tests/test_task_workflow*.py`
(simulate a truncated trailing line; assert readers tolerate it; assert a
concurrent append never interleaves a partial line).

## Scope / surfaces

- Primary target: `src/explore_persona_space/task_workflow.py` (`post_event`,
  `list_events` / the events.jsonl reader).
- Add a regression test in `tests/test_task_workflow*.py`.

## Constraints / invariants

- Preserve the existing `flock` on `~/.task-workflow/lock` + single-commit semantics.
- Workflow-surface only — never experiment code.
- `uv run pytest tests/test_task_workflow*.py` passes; ruff on touched files passes.

## Provenance

- workflow_fix_target: src/explore_persona_space/task_workflow.py
- fingerprint: postevent-atomic-append-corruption

---
title: 'Bound the session-auto-respawn lane: 7 respawns, zero progress'
kind: infra
tags: []
created_at: '2026-08-05T10:28:59Z'
has_clean_result: false
origin_prompt: 'Observed during the 2026-08-05 infra drain: #2004 accumulated 7 session-auto-respawn
  markers over 7h34m with no real progress since its 02:54Z completion-audit PASS;
  #2022 accumulated 4 over 4h20m since its 06:08Z PASS. Neither merged. Four sibling
  tasks in the same window needed zero respawns and one (#1988) completed cleanly,
  so this is a per-task stall the respawn lane fails to bound, not a systemic pipeline
  defect.'
workflow: v1
---
## Overview / Motivation

Two tasks are stuck in an unbounded watcher-respawn loop after passing their
completion audit. The watcher keeps respawning their sessions; the respawned
sessions post no real progress and stall again. Nothing escalates, so the
loop is open-ended and the tasks hold infra concurrency slots indefinitely.

Observed 2026-08-05 during a manual drain of the proposed-infra backlog.

## Goal

Bound the `session-auto-respawn` lane for the post-completion-audit stall
shape: after N consecutive respawns with NO intervening real (non-watcher)
marker, stop respawning and escalate (loud marker + push + sidecar row)
instead of looping. Today the loop appears unbounded in practice.

## Verified at filing (2026-08-05T10:28Z, read from events.jsonl)

- **#2004** — status `reviewing`, 25 events.
  Last REAL (non-watcher) marker: `2026-08-05T02:54 epm:completion-audit` PASS.
  Seven `session-auto-respawn` markers since: 06:03, 06:23, 07:43, 08:03,
  08:23, 10:03, 10:23. **7h34m with zero real progress.** No `epm:merged`.
- **#2022** — status `reviewing`, 21 events.
  Last REAL marker: `2026-08-05T06:08 epm:completion-audit` PASS.
  Four `session-auto-respawn` markers since: 08:23, 08:43, 10:03, 10:24.
  **4h20m with zero real progress.** No `epm:merged`.
- Respawn counts obtained by counting `session-auto-respawn` occurrences in
  each task's `task.py view <N> --json` output; "real marker" = an event whose
  note does NOT contain `autonomous_session_watch`.

## NOT a systemic pipeline defect — scope is these two tasks

Four of the six tasks in flight at the same time needed ZERO respawns, and
one of them (#1988) merged and completed cleanly during this window
(`epm:merged` 09:42, `completed` by 09:45). So the pipeline demonstrably
runs end-to-end; this is a per-task stall that the respawn lane fails to
bound, not a broken pipeline. Do NOT widen the fix on a systemic premise.

## Open questions for the planner (not pre-decided)

1. Why is this lane effectively unbounded at 7 respawns when other watcher
   respawn arms carry an episode belt (`STALLED_MAX_RESPAWNS`) plus a
   per-issue per-UTC-day cap? Candidate: each respawn creates a NEW session,
   and the episode state is advancement-cleared or session-keyed, so the
   counter resets every cycle and the belt never binds. Verify before fixing.
2. Is the right bound "N consecutive respawns with no intervening real
   marker" (progress-keyed) rather than a raw per-day count? The
   progress-keyed form is what distinguishes this shape from a healthy
   session that is respawned once and then works.
3. What should the terminal action be — escalate-only (marker + push +
   sidecar, task left `reviewing` for a human), or transition to `blocked`
   with a typed reason? Escalate-only is the more conservative default and
   matches the keep-running wedged-owner arm (#1582).
4. Is the underlying stall itself worth diagnosing separately (why does a
   session that passed its completion audit fail to reach Step 10d merge)?
   That may be a distinct task; this one is about bounding the loop.

## Scope / surfaces

- `scripts/autonomous_session_watch.py` — the `session-auto-respawn` lane and
  its episode/cap state.
- Its tests under `tests/test_autonomous_session_watch*.py`.
- `.claude/rules/background-automation.md` — the watcher pass documentation,
  if the bound changes documented behavior.

Grep before editing: `grep -rn "session-auto-respawn" scripts/ tests/ .claude/`

## Constraints / invariants

- 0 GPU-h.
- ESCALATE-ONLY unless the planner argues otherwise: this arm must not start
  killing sessions that are legitimately slow. The project's wedge heuristics
  based on elapsed silence have a poor track record (five false positives on
  2026-08-04); the reliable signal here is "N respawns with no intervening
  real marker", not elapsed time.
- Failing tests are rewritten to the new contract with a stated reason, never
  deleted.
- `scripts/workflow_lint.py` (no-flags) passes; ruff on touched files passes.

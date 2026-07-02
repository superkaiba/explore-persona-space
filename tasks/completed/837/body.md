---
title: 'workflow-fix: followup-round-repark fires while the round is still executing'
kind: infra
tags:
- wf-fix
created_at: '2026-07-02T06:33:01Z'
has_clean_result: false
origin_prompt: 'Periodic monitor observation: watcher reparked #778 mid-round twice
  (04:43, 06:13) while the null battery was still computing'
---
## Goal

Fix `autonomous_session_watch.py`'s followup-round-repark pass so it never reparks a `followups_running` task whose round is still executing.

## Workflow gap

- **Bug observed (twice on #778, 2026-07-02):** at 04:43 and again at 06:13 the watcher posted `[autonomous_session_watch:followup-round-repark] same-issue follow-up round complete (round-end ...)` and flipped #778 followups_running -> awaiting_promotion while the corrected-monitoring round was mid-execution — first while the analyzer fold had not run (results uploaded but body untouched), then again while the 1000-draw null-battery process (pid 456063) was actively computing at ~300% CPU. Each repark orphaned the round; recovery required manual/watcher respawns (05:01 replacement session; 06:29 re-dispatch).
- **Why it is a workflow gap:** the repark predicate treats some early round-end signal (likely the epm:results / upload-verification marker, or the round's session going idle between stage dispatches) as round COMPLETE. Round-complete must additionally require the round's terminal signal: the fold landed (an epm:analysis marker newer than the round entry, or the epm:same-issue-followup-run satisfier posted with the folded-body commit), AND no live round process (bg battery/analysis pid) attributable to the round.
- **Confidence (emitter):** high (two reproductions same day, same task).

## Proposed change (refine in planning)

In the followup-round-repark pass: gate the repark on (a) an epm:analysis (or round-end satisfier) marker NEWER than the round-entry status-change, and (b) no round-attributed live process/stage-dispatch marker within the last N minutes (a stage-dispatch newer than the newest round-end signal = round still active). Log-and-skip otherwise.

## Scope / surfaces

- Primary target: `scripts/autonomous_session_watch.py` (followup-round-repark pass)
- Tests: extend the watcher tests with a mid-round repark regression case.

## Provenance

- workflow_fix_target: scripts/autonomous_session_watch.py
- Observed on task #778, rounds at 2026-07-02T04:43:26Z and 2026-07-02T06:13:12Z.

---
title: 'daily-held: re-drive stranded #1622 merge (PR #1399)'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-25T06:51:20Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 3): Task 1622 completed 07-23
  with epm:done but no epm:merged; PR 1399 is still an open draft after the watcher
  killed its own 15:43Z recovery respawn ten minutes in, and the once-per-episode
  respawn budget is consumed'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (session 76a8649e, task #1622). HELD as needs-human — the recovery action is safe but TIMING is a judgment call: re-spawning before the watcher reconcile fix lands (filed tonight as the reconcile-kills-young-respawn item) risks the same 10-minute kill; the once-per-episode auto-respawn budget for this episode is consumed, so nothing recovers it automatically.

## State (verified at filing)

- #1622: `epm:done` 2026-07-23, no `epm:merged`; PR #1399 OPEN, draft=true (gh probe 2026-07-24).
- 15:43:11Z completed-unmerged-respawn spawned recovery; 15:53:38Z session-reconcile-stop killed it mid-lint-gate.

## Options

1. Re-spawn now: `uv run python scripts/spawn_session.py spawn-issue --issue 1622 --auto` (watch that it survives >10 min; the reconcile pass may kill it again until the fix lands).
2. Wait for the reconcile fix task to merge, then re-spawn.

Either way the merge itself is the standard no-gate Step 10d auto-merge; only the re-drive timing needs you (or the PM).

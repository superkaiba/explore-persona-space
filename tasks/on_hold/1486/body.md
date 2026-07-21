---
title: 'ruff residual burn-down: ~33 errors + 17 format-dirty files (post-#1023 scoping)'
kind: infra
tags: []
created_at: '2026-07-18T00:15:10Z'
has_clean_result: false
parent_id: 1023
origin_prompt: 'task #1023 decision (2026-07-17): option 1 deferred+descoped; revive
  trigger <3 live issue sessions or 2026-08-01'
workflow: v1
---
## Overview / Motivation

Deferred + descoped remainder of task #1023's ruff-debt decision (2026-07-17).
After #1023's config scoping landed (pyproject per-file-ignores on frozen
experiment paths + extend-exclude of artifact dirs), the visible repo-wide
ruff debt dropped 2,226 -> ~33. This task is the residual burn-down, deferred
to a quiet window because it edits src/ + eps/experiments/ files concurrently
touched by live branches.

## Scope (the exact residual, measured 2026-07-17 at branch commit dff56be0bc)

1. ~33 visible ruff errors: 16 src/explore_persona_space (12 sagan_progress.py,
   4 axis/), 11 F401 in eps/experiments/_factor_screen/, 3 F821 + 1 F401 +
   1 F601 in frozen scripts, 1 B006 in experiments/. (~19 are safe-auto-fixable.)
2. 17 ruff-format-dirty files (7 eps/experiments/, 8 scripts/, 1 src/, 1 tests/).
3. Disposition question for the 3 frozen-script F821s (real undefined-name
   bugs in write-frozen artifacts): fix vs leave visible — decide at run time.
4. Growth re-check duty from #1023's decision record: if the visible count
   grows >+5/week sustained over 2 weeks, file the workflow-fix candidate for
   a merge-time ruff ratchet on experiment merges (they currently never run ruff).

## Revive trigger

Revive (set-status proposed) when `spawn_session.py list` shows <3 live issue
sessions, OR on 2026-08-01, whichever comes first. Parked at on_hold so the
watcher's proposed_infra_sweep does NOT auto-dispatch it into the busy window
the deferral exists to avoid.

## Provenance

- parent: #1023 (ruff-debt disposition decision; option 1 DEFERRED+DESCOPED)
- source: plan v2 §4 row (f) + §P3, tasks/*/1023/plans/v2.md

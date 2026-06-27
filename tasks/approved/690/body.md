---
title: 'Auto-dispatch proposed infra tasks without a PM STATUS pass: file-time spawn
  + always-on watcher sweep'
kind: infra
tags: []
created_at: '2026-06-27T20:36:44Z'
has_clean_result: false
origin_prompt: fix this. but also add something so that these infra fixes just get
  dispatched automatically by whoever files them.
---
## Goal

A filed `proposed` `kind:infra` task currently only auto-runs if a LIVE PM session
performs a STATUS pass (`research-pm.md` §444 "Standing rule — infra auto-dispatch").
With no PM session in the loop, the task orphans indefinitely — incident: code-review
fast-follow task #684 was filed by the #682 session and sat at `proposed` for ~17h
until a human hand-spawned it. The always-on 10-min `autonomous_session_watch.py` cron
only re-drives `blocked` + `no_compute_available` tasks (#642); it deliberately skips
`proposed`. Close the gap from BOTH ends.

## Required changes

### 1. File-time auto-dispatch (primary mechanism)
Whoever FILES a ripe, auto-dispatchable `kind:infra` task should IMMEDIATELY dispatch it
(`spawn_session.py spawn-issue --issue <N> --auto`) instead of relying on PM pickup. Add
a mechanism that makes this the default and hard to forget — e.g. a `task.py new`-adjacent
helper or a thin `scripts/` wrapper that files + dispatches in one call — plus the
instruction wired into the routing surfaces (`CLAUDE.md` routing, `workflow-fix-on-bug.md`,
`research-pm.md`) that "filing a ripe infra fix == dispatching it." It MUST no-op
gracefully when no Happy daemon is reachable (pod-side / headless filing) — those cases
fall through to #2. Design the exact mechanism in the plan (do not bake session-spawning
into `task.py`'s state-mutation core if it breaks the pod-side "never shell out to task.py"
rule — prefer the orchestration layer).

### 2. Always-on watcher safety-net sweep (backstop)
Add a pass to `autonomous_session_watch.py` (the 10-min cron) that sweeps ripe `proposed`
`kind:infra` tasks and dispatches them via `spawn-issue --auto`, catching anything filed
by a context that could not self-dispatch (pods, manual `task.py new`, a crashed filer).
It MUST reuse the SAME scope + gating the PM rule uses: the 3-concurrent-auto-dispatched-
session cap, `holds`/predicate gating, `on_hold` exclusion, and skip any task that already
has a live issue-mapped session (no double-dispatch).

## Constraints / acceptance

- **Idempotent:** never spawn a 2nd session for a task that already has a live one
  (dedup on issue-mapped sessions). This is the most important invariant.
- **Shared concurrency cap:** respect the existing 3-session cap shared with PM
  auto-dispatch; don't blow past it when both paths are active.
- Don't dispatch `proposed` tasks that aren't `kind:infra`, that are `on_hold`, or that
  have unmet `holds`/predicates.
- Honor autonomous-mode conventions (`EPM_AUTONOMOUS_SESSION`).
- The existing PM-session infra auto-dispatch rule keeps working unchanged — this ADDS
  paths, it does not remove the PM one.
- **Tests** pin: (a) file-time dispatch fires on a ripe infra filing; (b) the watcher
  sweep dispatches an orphaned `proposed` infra task; (c) neither path double-dispatches a
  task that already has a live session; (d) the concurrency cap is respected.

## Provenance
Surfaced from the #684 incident (2026-06-27): a code-review fast-follow infra task filed
by the #682 session sat at `proposed` ~17h because no PM STATUS pass ran. User directive:
"fix this ... and add something so that these infra fixes just get dispatched
automatically by whoever files them."

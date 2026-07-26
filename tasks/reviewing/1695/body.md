---
title: 'daily-fix: refusal/transport-aware session recovery'
kind: infra
tags:
- wf-fix
- wf-fix-fp:6270f41d104e
- daily-auto-filed
created_at: '2026-07-26T07:04:34Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-25 problem sweep (route 2): A turn-1 Overloaded 529
  death on the #1689 autonomous session was mis-classified as a boot-refusal and left
  about three hours between task creation and a working session, and a separate refusal
  that killed an orchestrator turn one second before a tick fire still read HEALTHY
  from tick_triage so recovery was incidental rather than mechanical.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the `/daily` 2026-07-25 problem sweep. Two recovery lanes mis-read the
cause of a dead session: a transport-class 529 was classified as a refusal, and a
refusal that killed a turn still read HEALTHY.

## Goal

Make `tick_triage` return `STALE-REDRIVE` when an api-error assistant row is newer
than the latest marker, and split the watcher's boot-death shape into `boot-refusal`
versus `boot-transport` with a collapsed re-dispatch threshold for the transport case.

## Workflow gap

1. **A 529 mis-labelled as a refusal cost ~3 h.** Task #1689 was created at
   2026-07-25T18:54:59Z and a session spawned at 18:55:45Z. Its very first assistant
   turn was an API-error row — *"API Error: Repeated 529 Overloaded errors. The API is
   at capacity"* (`isApiErrorMessage: true`, the only assistant row in the 15-line
   file) — and it never acted again. The watcher's boot-death lane stopped it at
   19:33:34Z (registration age 38 min ≥ 30 min) with
   `shape=boot-refusal … refusal-killed boot turn, #1287`. The replacement session only
   booted at 21:57:16Z: **~3 h 1 min from task creation to real work**, of which ~2 h
   24 min was the post-stop grace before re-dispatch. A 529 is transport-class and
   freely retryable; a usage-policy refusal is not. Labelling the first as the second
   points diagnosis at the wrong ladder and forfeits the fast retry.
2. **A refusal that killed a turn still read HEALTHY.** In #1687 (`b656f7fa`) the
   orchestrator turn was refusal-killed at 07:54:25Z, one second before a scheduled
   `/issue-tick 1687` fire, exactly at the Step 4→5 handoff (draft PR opened,
   code-reviewer not yet spawned). `tick_triage.py` returned
   `HEALTHY status=running, marker age 1m — chain alive` at 07:54:31Z — whose contract
   is "end the turn immediately". The pipeline only resumed because the tick turn went
   on to spawn the reviewer anyway, **off-contract**. Honouring HEALTHY would have left
   the task idle to the next 45-min tick. Marker-age staleness cannot see a refusal
   that just killed the driving turn, and the #1074/#1209 wedge lanes need ≥3 failed
   wakes or 20 min of silence — both far slower than the signal already sitting in the
   transcript tail.
- **Confidence (emitter):** high on both observations; medium on the transport
  threshold value (5 min is the parked suggestion, not a measured optimum).
- verified-at-filing: absence confirmed in the named targets —
  `grep -c 'isApiErrorMessage\|api.error' scripts/tick_triage.py` → **0 hits** (the
  triage has no api-error awareness at all). The watcher's shape vocabulary is
  evidenced by the marker text quoted above, read from #1689's own events. Incident
  timings computed from transcript timestamps (task create 18:54:59Z, spawn 18:55:45Z,
  stop 19:33:34Z, replacement boot 21:57:16Z). Landed-fix history check
  `git log --oneline --since='7 days ago' -- scripts/tick_triage.py
  scripts/autonomous_session_watch.py` → the wave touched the watcher via #1668
  (`b173175e15`) and #1681's urgent-park router (`167141479c`); neither adds api-error
  awareness to triage nor splits the boot-death shape. (2026-07-25)

## Proposed change (refine in planning)

```
  scripts/tick_triage.py:
+ scan the tail of the session transcript for an assistant row with
+ isApiErrorMessage: true whose timestamp is NEWER than the latest marker;
+ when present, return STALE-REDRIVE (reason: api-error-after-marker)
+ instead of HEALTHY. Bounded tail read, same shape as the existing
+ #1629 human-active screen (which already reads the transcript).

  scripts/autonomous_session_watch.py boot-death lane:
+ classify the failing row: usage-policy text -> shape=boot-refusal (unchanged);
+ 529 / overloaded / 429 / timeout / connection -> shape=boot-transport
+ for boot-transport, collapse the registration-age threshold
+ (EPM_BOOT_TRANSPORT_MIN_AGE_MIN, default ~5) and re-dispatch immediately
+ rather than falling through to the proposed_infra_sweep grace.
```

The `#1629` human-active screen is the precedent for (1): `tick_triage` already reads
the transcript tail for a recency signal, so this is one more predicate on a read that
already happens — not a new I/O path.

## Scope / surfaces

- `scripts/tick_triage.py` (the new predicate + its verdict reason string).
- `scripts/autonomous_session_watch.py` (shape split + threshold).
- `.claude/skills/issue-tick/SKILL.md` — its verdict vocabulary is documented there;
  a new reason string must appear in the branch table.
- Keep the two halves in one task: they are the same mis-classification seen from the
  two ends (triage says healthy when it isn't; the watcher says refusal when it isn't).

## Constraints / invariants

- Preserve every existing day-cap / episode-belt bound on watcher re-dispatch — a
  faster transport lane must not become an unbounded respawn loop (the existing
  triggers are all day-capped by design, #1241).
- Fail toward the status quo: an unreadable transcript tail returns today's verdict,
  never a spurious STALE-REDRIVE (re-driving a healthy session is its own cost).
- A HEALTHY tick must stay ONE Bash call — do not turn the healthy path into a
  multi-probe turn.
- `scripts/workflow_lint.py --check-references` / `--check-asks` pass; ruff passes;
  the watcher + tick pin tests stay green.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a
  `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- workflow_fix_target: scripts/tick_triage.py
- fingerprint: 6270f41d104e
- Source: `/daily` 2026-07-25 transcript sweep, sessions `979c2d1c` + `5c5a89e8`
  (#1689) and `b656f7fa` (#1687).

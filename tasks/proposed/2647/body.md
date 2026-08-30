---
title: Add a runtime backstop that reports a stale Step 9c known-red ledger
kind: infra
tags: []
created_at: '2026-08-30T17:50:31Z'
has_clean_result: false
parent_id: 2645
origin_prompt: 'Deferred from #2645 plan v1 section 7: every alert channel for the
  nightly Step 9c ledger refresh lives inside the wrapper, so non-execution is unreportable;
  a staleness read from outside covers the whole class, not just the executable-bit
  cause.'
workflow: v1
---
---
kind: infra
---

# Add a runtime backstop that reports a stale Step 9c known-red ledger

## Goal

Make a stale `.claude/cache/step9c-baseline.json` ledger REPORTABLE at runtime, so that a nightly refresh which stops running is detected by something OUTSIDE the refresh itself.

## Why this is separate from #2645

#2645 fixed the immediate cause: `scripts/cron_step9c_ledger_refresh.sh` was committed non-executable, so 18 consecutive nightly fires died `Permission denied`, and it added the mechanical pin (a committed test asserting every crontab-referenced `scripts/cron_*.sh` is 0755 in the git index) that makes that exact cause non-recurring.

What #2645 deliberately did NOT fix is the reason the outage was INVISIBLE for 18 nights. Every alert channel the wrapper has — the per-day log under `logs/step9c_ledger_refresh/`, the audit sidecar row, the Telegram push on non-zero rc — lives INSIDE the wrapper. A wrapper that never executes cannot report its own non-execution. The only evidence was a raw redirect log in a different repo's log directory that nothing reads.

The executable-bit pin closes ONE cause. It does not close the class. A crontab line edited to a wrong path, the crontab wiped, cron itself not running, the wrapper deleted, `uv` vanishing from PATH, a rename that outruns the crontab — each produces the identical signature (a silently stale ledger and no alert), and each is invisible to a static index-mode test.

The corresponding item in #2645's task body was phrased "Consider whether..." and was deliberately absent from its `## Acceptance`, so deferring it was a scope call, not an acceptance dodge. The plan critic reviewed that deferral and judged it legitimate on the grounds that the deferred work is comparable in size to the whole of #2645.

## The read this needs

The backstop is a STALENESS read, not a process-liveness read: it asks whether the ledger's `refreshed_at` is older than expected, which is agnostic to WHY the refresh stopped, and is therefore the only shape that covers the whole class. `scripts/step9c_baseline.py` already exposes `ledger_age_hours(ledger)` (returns hours since `refreshed_at`, or `None` when unparseable), so the measurement primitive exists; what is missing is a consumer outside the wrapper that reads it on a schedule and escalates.

Note the two distinct thresholds — do not conflate them. The in-gate freshness window is 24 h (a ledger older than that triggers the in-session lazy refresh, which is the ~31-40 min mid-gate cost sessions pay). A BACKSTOP alert threshold should be meaningfully looser than 24 h, because a single missed night is normal noise (a fleet-busy night, a host reboot, a concurrent in-session refresh holding the flock) while a multi-day gap is the actual signal. Picking the alert threshold is part of this task's design work, not an inherited constant.

## Sketch, not a specification

The natural home is a pass in `scripts/autonomous_session_watch.py`, which already runs every 10 minutes and already owns the escalate-only surfacing idiom. A pass there needs, per the established convention in that file: its own episode state file, a dedup TTL so one stale episode produces one push rather than one per 10-minute tick plus a periodic re-alert, its own `EPM_DISABLE_*` kill switch, a row in the docstring pass inventory (which `workflow_lint.py --check-asw-docstring-pass-count` pins, so the docstring and the code must move together), and tests.

Posture should be ESCALATE-ONLY, mirroring the sibling audit passes: report the stale ledger, never auto-trigger a ~31-40 min refresh from inside a 10-minute watcher tick.

That is a sketch of the likeliest shape. The design is this task's own work, and an implementation that reaches the goal by another route is fine.

## Acceptance

1. A staleness read of the Step 9c baseline ledger runs on a schedule from OUTSIDE `scripts/cron_step9c_ledger_refresh.sh`, so a wrapper that never executes is still detected.
2. Exceeding the chosen staleness threshold produces exactly one deduped alert per episode, with a re-alert cadence, not one per tick.
3. The pass is escalate-only: it never triggers the refresh itself.
4. It has a kill switch, and the docstring pass inventory and its pin test agree with the code.
5. Committed tests cover the fresh case, the stale case, and the unparseable/missing-ledger case.

## Provenance

Deferred from #2645 plan v1 section 7. Filed with `--no-dispatch`: the account was session-limited at filing time, so an auto-spawned session would have died on the limit immediately. Dispatch when convenient via `spawn_session.py spawn-issue --issue <N> --auto`.

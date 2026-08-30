---
title: 'cron_step9c_ledger_refresh.sh is committed non-executable — #2114''s nightly
  refresh has failed Permission denied for 18 nights'
kind: infra
tags: []
created_at: '2026-08-30T05:53:13Z'
has_clean_result: false
origin_prompt: 'Found from #2387 while checking Step 9c ledger freshness: ledger 77.9h
  stale, cron armed but wrapper mode 100644 vs 100755 siblings'
workflow: v1
---
## Goal

Make #2114's nightly Step 9c known-red ledger refresh actually run, and add the
mechanical pin that would have caught it, so a cron wrapper can never again ship
without its executable bit.

## The bug

`scripts/cron_step9c_ledger_refresh.sh` is committed mode **100644**. Every
sibling cron wrapper is **100755**:

    100644  scripts/cron_step9c_ledger_refresh.sh   <-- not executable
    100755  scripts/cron_pod_audit.sh
    100755  scripts/cron_vm_disk_guard.sh
    100755  scripts/cron_daily_healthcheck.sh
    100755  scripts/cron_codex_auto_upgrade.sh

The crontab invokes the path directly:

    31 5 * * * /home/.../scripts/cron_step9c_ledger_refresh.sh >> ~/my-goat/logs/cron_step9c_ledger_refresh.log 2>&1

so every fire dies instantly:

    /bin/sh: 1: /home/.../scripts/cron_step9c_ledger_refresh.sh: Permission denied

**18 consecutive nights** of that line in the redirect log (measured
2026-08-29). The cron has never worked since #2114 shipped it.

## Why it stayed invisible

The failure is doubly silent, and the second half is the interesting part.

The wrapper is carefully designed to be loud: per-day log, an audit sidecar row
at `.claude/cache/step9c-refresh-cron-events.jsonl` on rc != 0, a once-per-day
sentinel-gated Telegram push, and ALWAYS exit 0. **Every one of those channels
lives inside the script.** When the script itself cannot execute, none of them
fire — the alerting is downstream of the thing that is broken.

So the only evidence is a raw redirect log in a DIFFERENT repo's log dir
(`~/my-goat/logs/`), which nothing reads. The sidecar's newest row is from
2026-08-11 (an unrelated rc=2), which actively reads as "quiet since mid-August"
rather than "never ran".

## Impact

The ledger goes stale, and Step 9c pays for it. Measured on #2387 today:
`refreshed_at = 2026-08-26T23:56:30Z`, **77.9 h** old against a 24 h freshness
window. #2114's own rationale is that a stale ledger costs a session ~31-40 min
of in-gate refresh (#2105, #1992, #2106) — that is the cost every session has
been paying, fleet-wide, for 18 days, for a cron that exists specifically to
prevent it.

## Fix

1. `git update-index --chmod=+x scripts/cron_step9c_ledger_refresh.sh` and
   commit. A bare `chmod +x` is NOT sufficient — git tracks the mode, so the
   fix has to land in the index or it will not survive a fresh checkout.
2. Verify the next fire actually runs (a dated log under
   `logs/step9c_ledger_refresh/` and a fresh `refreshed_at`), rather than
   assuming the chmod was enough.
3. Add the missing mechanical pin — the real deliverable. A test asserting that
   every `scripts/cron_*.sh` referenced by a crontab line is committed 0755
   would have caught this at review time and covers the whole class going
   forward. `tests/test_cron_step9c_ledger_refresh.py` exists and is thorough
   about the wrapper's BEHAVIOR, but never asserts the file can be executed —
   which is why five review rounds on #2114 passed it.
4. Consider whether the class needs a runtime backstop too: a wrapper whose
   alert channels are all internal has no way to report its own non-execution.
   An external liveness read (the ledger's own `refreshed_at` age, which the
   watcher could observe) is the natural place, since it is the quantity
   actually cared about and is independent of whether the wrapper ran.

## Acceptance

- The wrapper is committed 0755 and a subsequent nightly fire produces a dated
  log plus a fresh `refreshed_at`.
- A committed test fails if any crontab-referenced `scripts/cron_*.sh` is not
  0755 in the index.
- Existing `tests/test_cron_step9c_ledger_refresh.py` stays green.

## Provenance

Found from #2387 (cron-wrapper push-timeout bounding, a #2196 class sweep) while
checking ledger freshness before that task's Step 9c gate — the staleness is what
prompted looking at the cron at all. Not fixed in-session deliberately: #2387's
worktree is mid-review-round on an unrelated diff, and folding an unrelated mode
change into it would contaminate that review.

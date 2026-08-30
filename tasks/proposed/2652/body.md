---
title: Re-size the Step 9c ledger-refresh timeout — it runs at 99% of its own fence
kind: infra
tags: []
created_at: '2026-08-30T22:52:59Z'
has_clean_result: false
origin_prompt: 'Discovered while harvesting #2645''s A1 fire: the baseline ledger
  records duration_s 4308.6 against refresh_timeout_s 4350.0 (41s margin, 0.95%) for
  the 2026-08-26 run, and the 2026-08-30 run timed out at the fence. The refresh''s
  real cost is ~72 min, not the ~31-40 min several docs quote.'
workflow: v1
---
---
kind: infra
---

# Re-size the Step 9c ledger-refresh timeout — it runs at 99% of its own fence

## Goal

Make the nightly Step 9c known-red baseline refresh complete reliably, by correcting a fence that is sized with under 1% margin against the work it bounds — and correct the ~31-40 min cost figure that several docs and plans still quote.

## The measurement

Two datapoints, both from artifacts on disk, no estimation:

- **2026-08-26, SUCCEEDED at 4308.6 s.** Recorded inside the baseline ledger itself: `.claude/cache/step9c-baseline.json` carries `pytest_summary.duration_s = 4308.6` alongside `refresh_timeout_s = 4350.0`. That is **41.4 s of headroom, 0.95% of the fence**, over a universe of 6,416 tests across 143 `test_universe` entries.
- **2026-08-30, TIMED OUT at 4350 s.** `logs/step9c_ledger_refresh/2026-08-30.log`: `pytest exceeded 4350.0s — killing the process group` / `refresh pytest timed out after 4350.0s — NO ledger write`, wrapper `exit=2`, wall 72 min 59 s.

A job that finishes 41 seconds inside its own kill fence is not a job with a healthy margin; it is a job that fails on the next test added, the next slow import, or any contention on a VM shared with roughly 15 concurrent sessions. Four days after that near-miss it went over.

## Why this matters beyond one failed run

The refresh writes NO ledger on timeout, by design. So every timeout leaves the previous ledger in place and ageing. The in-gate consumer treats a ledger older than 24 h as stale and falls back to an in-session lazy refresh — which invokes the SAME command with the SAME fence, so it times out too, having burned another ~72 min of a session's wall clock. The failure mode is self-perpetuating: once the fence is tight enough to trip, neither the nightly nor the in-gate fallback can restore freshness, and every session reaching Step 9c pays ~72 minutes to discover that.

This was discovered while verifying #2645, which fixed a different bug in the same path (the wrapper was committed non-executable, so it never ran at all for 18 nights). #2645's fix is proven working — the wrapper now executes and its alert channels fire correctly — and that is exactly how this defect became visible: converting a silent non-execution into a loud, self-reporting timeout is what surfaced it. The two are independent defects in the same chain, and #2645 deliberately did not widen its scope to cover this one.

## The stale cost figure, which should be corrected wherever it appears

Multiple places carry ~31-40 min as the refresh's cost, traceable to #2105, #1992, and #2106. The measured cost is ~4300 s, roughly 72 min — about double. Known carriers to check and correct:

- `scripts/cron_step9c_ledger_refresh.sh` header comment ("sessions then pay the ~31-40 min refresh cost mid-gate")
- #2645 plan v1 section 11, which inherited it
- any other plan or doc quoting the range (grep for `31-40`)

A figure that under-states a wall by 2x will keep producing mis-sized fences and mis-sized session expectations until it is corrected at the source.

## Approach — the design decision is this task's own work

Candidate directions, not a specification:

1. **Raise the fence with real margin.** Cheapest, and treats the symptom. If chosen, size it off the measured ~4300 s with enough headroom to absorb both universe growth and shared-VM contention rather than picking another round number; and consider making the timeout a function of the measured baseline rather than a hardcoded constant, so it cannot silently re-tighten as the suite grows.
2. **Shrink or shard the universe.** 6,416 tests in one serial pytest invocation is the actual cost driver. Sharding across processes, or splitting the known-red baseline into independently-refreshable slices, attacks the wall rather than the fence. More work, better durability.
3. **Make timeout non-destructive.** Today a timeout discards everything and writes nothing, so a run that got 95% of the way through yields zero. Incremental or partial-shard ledger writes would let a timed-out run still advance freshness. This composes with either option above.

Whichever is chosen, the fence should stop being a number with no stated relationship to a measurement.

## Acceptance

1. The nightly refresh completes and writes a ledger on a normally-loaded shared VM, demonstrated by at least one real dated log showing `exit=0` and a `refreshed_at` that advances.
2. The fence's value is justified against a MEASURED wall with stated margin, in a comment or a derivation — not a bare constant.
3. The ~31-40 min figure is corrected wherever it appears, and the corrected number cites its measurement.
4. If timeout remains destructive (no partial write), that is stated as a deliberate choice with its cost acknowledged, rather than left implicit.

## Provenance

Discovered 2026-08-30 while harvesting #2645's A1 verification fire. Evidence is two on-disk artifacts: the ledger's own `duration_s`/`refresh_timeout_s` pair from the 2026-08-26 run, and the 2026-08-30 dated log's timeout lines. Filed with `--no-dispatch` — the fix involves a design choice among the three directions above, so it wants a plan rather than an immediate autonomous run. Dispatch via `spawn_session.py spawn-issue --issue <N> --auto` when it is worth the compute.

---
title: poll_pipeline ETA tripwire self-disables on plan §9 calendar-latency rows;
  range cells silently parse as lower bound
kind: infra
tags: []
created_at: '2026-08-07T15:58:49Z'
has_clean_result: false
origin_prompt: 'Surfaced by the #2162 orchestrator: poll_pipeline logged ''no parseable
  §9 planned_wall_h for #2162; phase-ETA tripwire disabled (fail-safe)'' on every
  tick of a healthy run; the plan HAS the column, and the offending row is ''| P9
  stage-2 judge | <=24 calendar | 0 | Batch API |''.'
workflow: v1
---
# Poller ETA tripwire silently self-disables on any plan whose §9 `planned_wall_h` column carries a calendar-latency row

## Goal

Make `poll_pipeline.py`'s phase-ETA tripwire survive plan §9 compute tables that legitimately contain non-numeric latency cells, and stop a range cell from being silently parsed as its lower bound. Today the tripwire disables itself fleet-wide on (probably) every plan that uses the Anthropic Batch API for judging, which is most `kind: experiment` plans.

## Evidence (observed live on #2162, 2026-08-07)

`poll_pipeline.py` emitted, on every tick of a healthy 8x H100 run:

```
INFO poll_pipeline: no parseable §9 planned_wall_h for #2162; phase-ETA tripwire disabled (fail-safe)
```

The plan is NOT missing the column — `tasks/running/2162/plans/plan.md` line 351 is
`| component | planned_wall_h | planned_gpu_h | parallelism | basis |`.

Running the parser directly against the plan text isolates the offending row:

```
_md_planned_wall_rows(plan) RAISED _UnparseableWallRow:
  | P9 stage-2 judge (≈36.3k calls) | ≤24 calendar | 0 | Batch API | same instrument |
```

The `planned_wall_h` cell is `≤24 calendar`. `_LEADING_FLOAT_RE` does not match a leading `≤`, so
`_md_planned_wall_rows` raises `_UnparseableWallRow`, and the caller maps that to `None` —
**disabling the whole tripwire**, by deliberate design (the docstring at `scripts/poll_pipeline.py`
~L3872-3881 says it raises rather than return a partial sum, "AC #2").

## Two distinct defects

1. **Hard self-disable.** ONE non-numeric cell anywhere in the table turns off ETA-deviation
   detection for the ENTIRE run. The fail-safe is defensible in isolation (never a partial sum),
   but the trigger is a LEGITIMATE plan authoring choice, not a malformed plan: a Batch-API judge
   phase's cost is calendar latency (SLA 2-24 h), not GPU wall-hours, and writing a number there
   would be false.

2. **Silent lower-bound mis-parse — worse than the disable.** The sibling row in the same table,
   `| P6 judge waves (pilot + gate-3 sync + batch) | 2–24 calendar | 0 | ... |`, has a cell
   beginning with the digit `2`, so `_LEADING_FLOAT_RE` DOES match and it parses as `2.0` —
   contributing a bogus 2 h to the planned total and silently discarding the `–24 calendar`
   remainder. Had the P9 row been written `2–24 calendar` instead of `≤24 calendar`, the tripwire
   would have stayed ENABLED with a wrong (understated) budget, which fires spurious ETA-deviation
   posts instead of disabling. So the current behavior is bimodal and both branches are wrong:
   a leading digit gives a silently wrong budget; a leading `≤` gives no budget at all.

## Why this is systematic, not #2162-specific

Every `kind: experiment` plan that routes judging through the Batch API (the standing CLAUDE.md
mandate for large judge sets) has at least one phase whose honest wall figure is a calendar-latency
range or bound. The planner has no documented way to express "this phase costs no GPU wall-time,
it costs up to 24 h of calendar latency" in a `planned_wall_h` cell, so authors write prose. The
poller then either disables or mis-sums. Worth checking how many recent plans' tripwires are
silently off.

## Proposed fix (direction, not a mandate — the spawned session's planner + critics decide)

- **Poller side:** recognize a documented set of non-wall sentinels in a `planned_wall_h` cell
  (e.g. a cell with no leading float that matches a `calendar` / `n/a` marker) and SKIP that row
  rather than disabling the tripwire, while still hard-raising `_UnparseableWallRow` on a
  genuinely malformed numeric cell (preserving the AC #2 intent: never a silent partial sum from a
  MALFORMED row, but do not conflate "deliberately not a wall figure" with "malformed").
- **Reject ambiguity rather than guessing:** a range cell (`2–24`, `2-24`) must NOT parse to its
  lower bound. Either raise, or require an explicit single figure. The current silent lower-bound
  behavior is the more dangerous of the two defects.
- **Planner side:** give §9 an explicit convention for calendar-latency phases — e.g. a separate
  `calendar_latency_h` column, or a required literal in the `planned_wall_h` cell — and state it in
  `.claude/rules/plan-compute-sizing.md` so the poller contract is authored against rather than
  discovered. Whatever literal is chosen must be pinned by a test on BOTH sides so the two surfaces
  cannot drift.
- Add regression fixtures for: a `≤24 calendar` cell, a `2–24 calendar` cell, a genuinely malformed
  cell (must still raise), and an all-numeric table (must still sum).

## Impact / severity

Monitoring-only; no data or spend at risk directly. Stall detection (pid liveness + log mtime) is
unaffected and still fired correctly throughout #2162. The loss is the ETA-deviation auto-post,
which is the mechanism that would catch a throughput blowout on a multi-hour run. #2162 retains
independent protection (its plan's own throughput pilot gate re-derives the fence at 2x the
pilot-extrapolated wall), so this is a degraded-defence-in-depth finding, not an active incident.

## Provenance

Surfaced by the #2162 orchestrator while the run was healthy in P2. Parser located at
`scripts/poll_pipeline.py` `_md_planned_wall_rows` (~L3872-3906), caller `_parse_plan_wall_budget` /
`_plan_total_wall_h_for_issue`; both returned `None` for #2162's plan. `_html_planned_wall_rows`
returned `[]` (the plan's table is markdown).

---
title: Parenthesized §9 planned_wall_h cell silently disables the poller phase-ETA
  tripwire
kind: infra
tags:
- workflow-fix
- poll-pipeline
created_at: '2026-08-07T13:33:46Z'
has_clean_result: false
origin_prompt: 'poll_pipeline tick 1 on #2163 logged ''no parseable §9 planned_wall_h;
  phase-ETA tripwire disabled (fail-safe)''; root cause is the conditional GPU row''s
  parenthesized (1.5) cell.'
workflow: v1
---
# A parenthesized §9 `planned_wall_h` cell silently disables the poller's phase-ETA tripwire

## Goal

Stop a single non-leading-float `planned_wall_h` table cell from silently disabling the
`poll_pipeline.py` phase-ETA tripwire for an entire run — either by tolerating the common
conditional-value spellings, or by catching them at PLAN time so the degradation is impossible to
ship unnoticed.

## The bug

`poll_pipeline._md_planned_wall_rows` scans the plan §9 compute table for the `planned_wall_h`
column and takes each cell's LEADING float via
`_LEADING_FLOAT_RE = re.compile(r"\s*([0-9]+(?:\.[0-9]+)?)")`. If ANY located data row's cell yields
no leading float it raises `_UnparseableWallRow`, and the caller fails safe by disabling the
phase-ETA tripwire for the whole run, logging one INFO line:

    no parseable §9 planned_wall_h for #<N>; phase-ETA tripwire disabled (fail-safe)

Failing safe is the right direction — a mis-parsed wall would mis-fire the tripwire, which is worse.
The defect is that it is **silent and all-or-nothing**: one cosmetically-formatted cell in one row
costs the tripwire for every phase of a multi-hour run, and nothing surfaces it except a human
reading a single INFO line in one poll tick's output.

## Observed on #2163 (2026-08-07)

Plan v5 §9 writes the CONDITIONAL GPU cell's wall in parentheses to mark it as
fires-only-if-triggered — a reasonable and readable convention:

    | P6-GPU conditional cell (1x H100 fp64 eigh + solve + score) | (1.5) | (1.5 realized; 4 booked) | ... |

`(1.5)` starts with `(`, so there is no leading float, so the tripwire went off for the whole ~6 h
run. Every other row parsed fine. The run proceeded on its PRIMARY protection — the driver's own
per-phase `_pilot_gate`, which is strictly tighter (per-phase, aborts rc=7 past 2x that phase's
planned wall) than the poller's whole-run total — so nothing was actually at risk here. But that was
luck of this plan's design, not a property of the guard.

`verify_plan.py` passed the plan PASS / 0 FAIL / 0 WARN, so plan time gave no signal either.

## Why this is worth fixing rather than a one-off plan edit

The parenthesized-conditional spelling is a natural way to write a maybe-phase, and §9 rows also
carry ranges (`5-6`), tildes (`~0.5`), and `TBD`-ish values in practice. Any of those in the
`planned_wall_h` column silently costs the tripwire fleet-wide. Fixing the single #2163 cell would
restore one run's backstop and leave the class open for the next plan.

## Proposed fix (either or both; the planner should adjudicate)

**(a) Parser tolerance.** Extract the first float ANYWHERE in the cell rather than requiring it at
the leading position — `re.search` over the cell with the existing float pattern — so `(1.5)`,
`~1.5`, `1.5-2` and `1.5 (conditional)` all yield `1.5`. Keep `_UnparseableWallRow` for cells with
no float at all (genuinely `TBD` / `N/A`), and consider degrading PER ROW (drop the unparseable row
from the total, keep the tripwire armed on the rest) instead of disabling globally — a total that
omits one conditional phase is more permissive, i.e. still fail-safe.

**(b) Plan-time check.** Add a `verify_plan.py` check that every §9 `planned_wall_h` data cell
parses under whatever rule the poller uses, WARNing (or FAILing) with the offending row named. This
is the durable half: it moves the discovery from "someone read a poll tick" to "the plan gate said
so before dispatch". Pin poller and verifier to ONE shared parse helper so they cannot drift.

**(c) Make the degradation loud.** When the tripwire does disable itself, have the poller post a
one-line `epm:progress` (or fold it into the existing tick advisory path) naming the unparseable row,
so the reduced-backstop condition lands in the durable record rather than only in stdout.

## Acceptance criteria

1. A §9 table containing a `(1.5)` cell keeps the phase-ETA tripwire ARMED for the parseable rows.
2. A cell with no float at all still fails safe, and the failure names the offending row.
3. `verify_plan.py` surfaces an unparseable `planned_wall_h` cell at plan time, with the row named.
4. The poller and `verify_plan.py` share one parse helper (no duplicated regex).
5. If the tripwire does disable itself at runtime, that fact reaches the task record, not just stdout.
6. Tests pin (1), (2), (3); `workflow_lint.py` passes.

## Provenance

Filed by the #2163 orchestrator after reading the tripwire-disabled line in poll tick 1. #2163 was
not remediated by a plan v6 on purpose: its per-phase pilot gates are live and tighter, and a
mid-run plan version would muddy which version the run executed under for a redundant backstop.

workflow_fix_target: scripts/poll_pipeline.py

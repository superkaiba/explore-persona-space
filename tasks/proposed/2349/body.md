---
title: 'plan-compute-sizing.md: a 1-unit pilot is not a sizing basis when the phase''s
  own design admits unequal unit costs (1.5x grid under-projection, #2329)'
kind: infra
tags: []
created_at: '2026-08-17T15:14:15Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'orchestrator-surfaced during /issue 2329: the gate-4 pilot measured
  blocks[:1] on a 234-block work-conserving queue whose plan states ''unequal block
  costs absorbed by work conservation''; projected 5.97h/48 GPU-h vs realized ~9.5-10h/~80
  GPU-h, with the cap raise excluded by measurement'
workflow: v1
---
## Goal

`.claude/rules/plan-compute-sizing.md` requires a MEASURED 1-cell pilot as the
per-call sizing basis, and covers heterogeneity ACROSS families (the "HETEROGENEOUS
FAN-OUTS / per-family pilot floor" block at ~L475: families whose budget/grid/cell-size
multipliers differ by >~4x each get their own pilot). It does NOT cover
**within-family unit-cost dispersion**: a phase whose units are the SAME family, with
no budget/grid multiplier to scale by, but whose per-unit costs are nonetheless
unequal. A 1-unit pilot there is a biased basis by construction, and the rule's
escape hatch (">=2x the worst-case extrapolation, COMPUTED") is unavailable because
there is no multiplier ratio to compute from.

## Evidence (#2329 P3 grid, 2026-08-17)

The grid phase is 234 (cell x slot x arm) blocks pulled from a SHARED
work-conserving claim queue. The plan states the design rationale explicitly:
"unequal block costs absorbed by work conservation" (plan §9, P3 row). So the plan
KNEW the units were unequal.

The gate-4 throughput pilot ran `blocks = blocks[:1]` (`issue2329_run.py`, pilot
branch) and extrapolated by rollout count:

    projected_wall_h = per_rollout * n_rollouts / width / 3600

Realized vs projected:

| quantity | pilot projection | realized (early) |
|---|---|---|
| s/rollout/worker | 4.1983 | ~6.15 |
| grid wall | 5.97 h | ~9.5-10 h |
| grid GPU-h @ width 8 | ~48 | ~80 |

That is a ~1.5x under-projection of the single largest phase, on a task whose
registered total was 45 GPU-h.

**The cap raise is excluded as the cause, by measurement** — the grid's own realized
distribution (4,860 rows) shows only 5.86% of rows exceed the old 2048 cap, worth
~+7% wall, matching the pre-registered estimate from two independent bases. The
residual ~1.36x traces to the pilot's sampled unit, not the cap.

**Honest scoping of the consequence:** the >=2x fence rule WORKED — the fence was
11.93 h and the realized ~9.5-10 h wall fits inside it, so nothing was killed. The
damage is confined to (a) a wrong ETA communicated across many status updates, and
(b) a GPU-h projection that under-counted the largest phase by ~1.5x, which is what
plan approval and expectation-setting rest on. This is a sizing-accuracy gap, not a
run-safety gap.

## Proposed fix

Add a WITHIN-FAMILY DISPERSION clause to the pilot-basis section, triggered by
design-level evidence rather than by a measured multiplier:

- **Trigger:** the phase's own design acknowledges unequal unit costs — a
  work-conserving/claim/dynamic queue, an explicit "unequal costs" note, or a unit
  set spanning a structural size axis (e.g. variable pairs-per-cell after
  exclusions). If the plan says work conservation absorbs cost inequality, a 1-unit
  pilot is disqualified as the sizing basis by that admission.
- **Requirement:** sample K >= 3 units chosen to span the cost range (or the
  cheapest + most expensive by a stated structural proxy), and size on the MEAN with
  the observed max/min ratio reported in the §9 row; a single unit may still set the
  fence, never the projection.
- **Escape:** if K>=3 is impractical, state the projection as a RANGE (1-unit basis x
  observed dispersion) rather than a point estimate, and mark the row
  `dispersion-unsampled`.
- **Reporting:** the realized max/min per-unit ratio goes in the clean-result /
  report Compute row next to the projection, so the next task inherits a real
  dispersion figure instead of re-deriving it.

Related but distinct precedent already in the rule: the per-family pilot floor
(across-family, multiplier-computable). This clause is its within-family sibling,
where no multiplier exists.

Sibling observation worth a line in the same edit: #2329's Phase B measured a 1.39x
per-worker wall spread and Phase A 2.2x on the SAME workload shape, and per-unit cost
drifted up to 21% WITHIN a single worker's run — so unit-cost dispersion in this
codebase is routinely large enough to matter, and early-window rates are biased
estimators of it.

## Acceptance

- `plan-compute-sizing.md` carries the within-family dispersion clause with its
  design-level trigger, the K>=3 requirement, the range escape, and the reporting duty.
- The distinction from the existing across-family per-family pilot floor is stated so
  the two are not conflated.
- No change to the >=2x fence rule (it worked here).

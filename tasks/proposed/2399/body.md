---
title: 'verify_plan: assert the conditional-branch GPU-hour upper bound fits the declared
  budget and self-fence (plus compound wall-cell parse)'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-19T21:29:53Z'
has_clean_result: false
origin_prompt: 'surfaced by codex-efficiency-critic during #2329 q35_ladder_decay
  critique round 1: plan v5 books 6 GPU-h with a 9-10h fence while its own G2/G5 gates
  permit a ~10.8 GPU-h worst case; verify_plan returned PASS because the fence was
  compared against the baseline projection'
workflow: v1
---
# verify_plan: assert the plan's CONDITIONAL-branch GPU-hour upper bound fits its declared budget and self-fence

## Provenance

workflow_fix_target: scripts/verify_plan.py

Surfaced during the #2329 `q35_ladder_decay` post-approval critique panel
(round 1) by the `codex-efficiency-critic` twin (branch-coverage arithmetic,
`mechanizable: yes`) and, as a same-file sibling, by the Claude
`efficiency-critic` (compound wall-cell parse; an agent-memory note for it
landed at `.claude/agent-memory/efficiency-critic/compound_wall_cell_parse_check.md`).

## The bug this would have caught

Plan #2329 v5 books **6 GPU-h** (4.6 base + 1.4 contingency) with a
**9-10 h** self-fence, and separately registers gates that PERMIT more work:
G2 aborts only at nearly 2x the baseline rollout rate, and G5 permits
regenerating the grid. Taking the plan's own 5.12 GPU-s/rollout basis:
baseline generation is ~0.60 h (420 rollouts) + ~1.88 h (1,320 rollouts); at
the G2 abort threshold the base rises to ~7.1 GPU-h; a broad G5 regeneration
adds ~1.88 h at baseline or ~3.75 h near the G2 threshold — a permitted worst
case of ~10.8 GPU-h, which crosses BOTH the 6 h booking and the 9-10 h fence.

Nothing caught it mechanically: `verify_plan.py --issue 2329` returned PASS,
and the Claude efficiency lens read the fence as compliant because it compared
the fence against the BASELINE projection (4.6 h) rather than against the
worst case the plan's own gates permit. The gap is structural — the budget is
computed on the happy path while the gates define a larger reachable set.

**Sibling shape (same file, same §9 table).** A compound wall cell such as
`0.5 VM + <=24 calendar` parses as `0.5` through
`plan_wall_budget.parse_wall_cell`, silently dropping a 24 h Batch SLA from the
phase-ETA tripwire budget (the same under-fence shape as #2162's
`2-24 calendar` cell). Consequence there is a FALSE lateness escalation during
a healthy in-SLA wait rather than overspend, but the mechanism — a §9 cell
whose parsed number understates the registered bound — is the same.

## Proposed checks

1. **Branch-coverage**: enumerate the plan's conditional GPU branches (retry /
   abort-threshold / regeneration gates), compute the cumulative upper-bound
   GPU-hours and wall using the plan's own per-unit basis, and FAIL when that
   upper bound exceeds either the declared `gpu_hours_total` or the registered
   self-fence. The remedy the plan author picks is either to book/fence the
   full branch set or to make the budget binding (a budget-derived abort
   threshold plus an aggregate regeneration cap) — the check only has to force
   the choice.
2. **Compound wall-cell parse**: parse every §9 wall cell and FAIL (or WARN)
   when a cell's parsed value is less than a `calendar`/`SLA`-tagged bound
   appearing in the SAME cell; the fix is to put the larger bound in the wall
   column and the smaller figure in the basis column.

## Acceptance criteria

1. A fixture plan whose gate-permitted upper bound exceeds its booked
   GPU-hours FAILs, with the message showing the derived upper bound and which
   limit it crosses.
2. A fixture whose branches fit both limits PASSes.
3. A compound wall cell `0.5 VM + <=24 calendar` is flagged; a plain cell is
   not.
4. Both checks are in the no-flags default `verify_plan.py` run.
5. Report the delta over a sample of committed plans; a plan with no
   conditional GPU branches must be unaffected.
6. Where the branch set cannot be enumerated mechanically, WARN rather than
   FAIL.

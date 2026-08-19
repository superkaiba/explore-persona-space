---
title: 'workflow-fix: re-verify premise before spending when a task resumes from a
  long park; dedup screen for infra filings'
kind: infra
tags: []
created_at: '2026-08-19T17:49:45Z'
has_clean_result: false
origin_prompt: '/issue 2321 — the authorized live repack halted at its own cap probe
  because sibling task #2332 had already done the work while #2321 sat parked at blocked;
  no workflow gate re-checked the premise on resume.'
workflow: v1
---
# workflow-fix: a task resuming from a long park must re-verify its premise before spending, and duplicate `kind: infra` filings need a dedup screen

## Goal

Close two gaps that let tasks **#2321** and **#2332** independently plan, implement, review, and
(for #2332) execute the *same* piece of work — and let #2321's user authorization for ~494k
irreversible Hub deletions be granted, and a pod provisioned, against a premise that a sibling
task had already invalidated two days earlier.

Nothing was lost: #2321's own driver measured the live file count as its first act and refused
off-cap. The point of this task is that the save came from one task's hand-written guard, not
from the workflow.

## What happened (2026-08-19, evidence in #2321's `epm:failure v1`)

- #2321 "Repack the 10 largest prefixes in the HF data repo to recover ~610k of the
  1,000,000-file cap" (`kind: infra`) finished implementation 2026-08-16 and parked at `blocked`
  at its Step 10 completion audit, awaiting a user decision on the deletion-bearing live phase.
- #2332 "Repack small-file HF prefixes to reclaim ~610k file-count slots without breaking
  readers" (`kind: infra`) — same goal, same measured top-10 inventory (610,356 files), same
  target prefixes, **no parent link either way** — executed on 2026-08-17, deleting 494,456
  verified-packed originals across 7 of #2321's 10 prefixes, and landed on main
  (`87640d1a1b`, "492,786 file-count slots freed") including its own packed-path reader shim
  (`orchestrate/packed_prefix.py` + the `hub.py` hook).
- On 2026-08-19 the user authorized #2321's live phase. The resumed session went from Step 0
  straight to provisioning a pod and launching the repack. The premise ("repo at 999,999, at
  cap, rejecting uploads fleet-wide") was two days stale: the repo was at **454,804 / 1,000,000**
  with 545,196 slots free, and 7 of the 10 prefixes were already packed.
- Cost: #2321's ~12,300-line branch is probably dead work; a pod was provisioned and torn down;
  and the user spent a decision on a cost/benefit tradeoff that had already inverted.

## Gap 1 (primary): no premise re-verification on resume-from-park

The planner's live-sibling check and the artifact-reuse search both run at **plan time**. #2321's
plan ran them correctly on 2026-08-16 and found nothing — #2332 did not exist yet. Nothing re-runs
them when a task wakes up days later. `.claude/skills/issue/SKILL.md` Step 0 loads state and
dispatches the next action; there is no "is this still needed?" gate in between.

The existing instrument-supersession duty (SKILL.md § 9a-ter) is the closest relative but does not
cover this: it is scoped to *measurement instruments* (judge rubrics, scorers), fires at inline-round
dispatch, and asks "is a stronger instrument in flight?" — not "has this entire task's work already
been done by a sibling?"

Proposed shape (the implementing session should re-derive, not take this as settled):

- At Step 0, compute the park duration: the gap between the latest transition INTO the current
  active status and the preceding transition OUT of an active status. When that park exceeds a
  threshold (~24 h is the obvious starting point; #2321's was ~3 days) AND the next pipeline action
  is spend-bearing or mutation-bearing (pod provision, GPU launch, Hub write, deletion), require a
  **premise re-check** before dispatching: (a) re-measure whatever quantities the plan's §1 Goal and
  its acceptance criteria are stated in terms of, and (b) re-run the duplicate/live-sibling scan
  including **terminal** siblings, not just live ones. Post the result as a marker; on a material
  divergence, park rather than proceed.
- (b) is the load-bearing half and the part #2321's plan got structurally wrong even at plan time:
  its live-sibling check scoped itself to *live sessions and unmerged branches*. #2332 was neither
  by 08-19 — it was `completed` and merged, which is exactly the state that makes a duplicate
  *most* certain to have already done the work. A sibling scan that only looks for in-flight work
  cannot see finished work.
- Cheap generic premise probe worth considering: re-run the plan's own §9 preconditions and any
  measured figure the acceptance criteria quote. #2321's driver did precisely this and it cost
  ~3 minutes of Hub listing.

## Gap 2: no dedup screen for `kind: infra` filings

The routing rule keys duplicate detection on *question identity* for experiments ("would the
result rewrite THIS issue's Takeaways?"). For `kind: infra` there is no equivalent screen at
`task.py new`: #2321 and #2332 have near-identical titles, cite the same measured inventory, and
target the same prefixes, and nothing flagged it at creation or at either plan.

A title/goal-similarity screen at `task.py new --kind infra` — surfacing the top few existing
tasks (any status, including `completed`) whose title or goal overlaps, for the filer to
acknowledge or dismiss — would have caught this at zero cost. Note the existing wf-fix dedup
predicate work (#1180, #1399, #1483, #1687) covers `daily-fix:`/`workflow-fix:` title tokens
only; ordinary infra titles are unscreened.

## Acceptance criteria

1. A task whose next action is spend- or mutation-bearing, resuming after a park longer than the
   chosen threshold, cannot reach that action without a recorded premise re-check.
2. The duplicate/sibling scan used by that re-check (and by the planner) includes terminal
   siblings — `completed`, `archived`, and merged branches — not only live sessions and unmerged
   branches.
3. `task.py new --kind infra` surfaces title/goal-overlapping existing tasks across all statuses
   at creation.
4. Tests pin all three, including a regression fixture reproducing the #2321/#2332 shape: task A
   parked with a plan-time sibling scan that found nothing, sibling B completes and merges the
   same work, A resumes and must be stopped by the re-check rather than by its own driver.
5. Whatever threshold and scope are chosen are documented in the rule surface the planner and
   `/issue` Step 0 both read.

## Explicitly out of scope

The disposition of #2321 itself (archive as duplicate / salvage the non-colliding parts / finish
the one remaining prefix) is a user decision parked on that task; do not act on it here. Also out
of scope: the two competing pack formats and reader shims now implied by #2321's branch vs main —
that resolves itself once #2321's disposition is settled.

## Provenance

Filed by the #2321 orchestrator session `cmt0cn4pxet5wye0ujmdrpz1b` on 2026-08-19 immediately
after parking #2321 at `blocked`. Full evidence — measured counts, the per-prefix deletion table
read from the live repo's own commit titles, and the `hub.py` shim collision — is in #2321's
`epm:failure v1`.

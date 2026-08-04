---
title: 'workflow-fix: cross-session lane pivot must resolve the owni'
kind: infra
tags:
- wf-fix
- wf-fix-fp:7be5012a67de
- daily-auto-filed
created_at: '2026-08-04T06:53:02Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-03 problem sweep (route 2): A #1345 lane pivot performed
  by a non-owning session provisioned pod-1345-charcapa and recorded it only as a
  durable epm:progress marker; the owning session never learned and kept waiting on
  a ~20h fellows queue while the 8xH200 sat 74+ min at 0% GPU (~$54). The failover
  rule places no owner-resolution duty on a pivoting session (its 8 owner mentions
  are all the #1667 wedge liveness guard).'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-03 (route 2: behavior/logic change → independent review) from the nightly problem sweep (miner8, sessions 954cd1a4 / 09fee09f, task #1345).

## Goal

A session that pivots a task's compute lane and provisions a pod on a task ANOTHER live session owns must actually reach that owner — a durable marker alone does not wake it, and the owner keeps waiting on the lane the pivot abandoned while the new pod bills.

## Workflow gap

- **Bug observed:** a lane pivot for #1345 was performed by a session that did not own #1345; it provisioned the 8×H200 `pod-1345-charcapa` and posted the pivot as a durable `epm:progress` marker at 2026-08-03T18:14:24Z. The OWNING session (09fee09f) never learned of it and kept waiting on a fellows SLURM job queued ~20h out. The pod sat **74+ minutes at 0% GPU across all 8 H200s with no logs (~$54)** before a third session diagnosed it: "The pod is waiting for nothing. It's orphaned. The pivot was done by a *different* session ... That session never learned about the RunPod pivot at all" (rows L925/L943). Resolution required stopping the owner session at 19:42:54Z and `scancel`ling fellows job 18419.
- **Why it is a workflow gap:** `.claude/rules/compute-backend-failover.md` documents the pivot/failover mechanics but places no duty on the pivoting session to STEER the owner. Its only owner-related language is the #1667 wedge **owner-liveness guard** (L1377–1393) — a watcher-side check on whether to auto-terminate a wedged pod, which is a different concern and fires on a different path. And per CLAUDE.md § teammate coordination, an independent Happy session is NOT SendMessage-addressable, so a cross-session pivot's ONLY channel is a durable task marker — which this incident shows the owner does not poll in time. The result is the #664 idle-burn class arriving by a new route.
- **Confidence (emitter):** high for the incident; medium for the fix shape (the channel question is genuinely constrained — see below).
- verified-at-filing: `grep -in 'owner\|SendMessage\|steer' .claude/rules/compute-backend-failover.md` → **8** hits, ALL within the #1667 wedge owner-liveness-guard block (L1377–L1393, read at compose time) — none imposes a notify duty on a pivoting session. Absence claim verified in-target with the near-miss language read rather than counted. `uv run python scripts/pod.py list-ephemeral` (2026-08-04) → `pod-1345-charcapa` no longer live (the incident's pod was terminated in-session).
- unverified hypothesis — verify at plan time: the right mechanism. An independent Happy session cannot be SendMessage'd, so the candidate options are (i) the pivoting session STOPS the owner and takes ownership explicitly (what this incident eventually did by hand), (ii) a pre-pivot ownership check that refuses to pivot a task with a live foreign owner, or (iii) an owner-side duty to re-read task markers before continuing a long queue wait. These have materially different blast radii and the planner should choose deliberately rather than adding a notify line that no channel delivers.

## Proposed change (candidate sketch — refine in planning)

```
in .claude/rules/compute-backend-failover.md (pivot recipe):
  BEFORE provisioning on a task you do not own, resolve the owner
  (spawn_session.py list / the Step-0 single-orchestrator registry):
    - live foreign owner -> either stand it down explicitly (stop + record) or
      refuse the pivot; never provision into a split-ownership state
    - record the disposition on the task
```

## Scope / surfaces

- Primary target: `.claude/rules/compute-backend-failover.md` (pivot recipe).
- Adjacent: CLAUDE.md § teammate coordination clause (a) (one implementer per file set → one owner per task's compute), and the /issue detached-handoff block.

## Constraints / invariants

- Must not introduce a pivot path that leaves two sessions believing they own one pod (the #1739 `pod-1739-ext` two-owner incident the same day is the sibling failure).
- Must not block a legitimate rescue pivot when the owner is provably dead.
- Workflow-surface only.

## Provenance

- fingerprint: 7be5012a67de

- workflow_fix_target: .claude/rules/compute-backend-failover.md

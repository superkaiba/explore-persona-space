---
title: 'workflow-fix: width re-eval on FLOP-bound mid-run deviation relaunch'
kind: infra
tags:
- wf-fix
- wf-fix-fp:e20e0f5d5faf
created_at: '2026-07-15T12:19:17Z'
has_clean_result: false
origin_prompt: 'Orchestrator-observed during #1092 offvm refit progress check (user
  asked: ''are you sure it wouldn''t be faster to change mid-run?''): the mid-run
  2x deviation rule''s negative-signature branch has no width re-evaluation step;
  the relaunch kept 4-box width and left ~5-6h wall on the table.'
workflow: v1
---
## Overview / Motivation

Auto-filed by the workflow-fix-on-bug protocol from an orchestrator-observed gap during the #1092 offvm-battery-refit follow-up round (emitting agent: orchestrator, user-chat session).

## Goal

Add a width-re-evaluation step to the mid-run compute-deviation branch: when the ≥2× deviation's vectorize signature check is NEGATIVE (FLOP-bound, engine already batched) on an embarrassingly-parallel unit grid with checkpoint/restore machinery live, the relaunch decision MUST evaluate re-sharding the REMAINING units across a wider fleet (credits-unconstrained default: wall-clock is the scarce resource) instead of defaulting to continue_as_is at fixed width.

## Workflow gap

- **Bug observed:** #1092 offvm refit (2026-07-15): after a 2.57× wall deviation (planned 5 h/box → projected 12.84 h/box) with a recorded NEGATIVE signature check, the crash-fix relaunch restored checkpoints but kept the original 4-box width for a FLOP-bound grid of 128 embarrassingly-parallel units — re-sharding the ~112 remaining units across 8–12 boxes at the relaunch point (restore made this cheap) would have cut ~5–6 h of wall-clock; the original v6 grid itself ran 12 boxes.
- **Why it is a workflow gap:** `.claude/rules/vectorize-many-cell-fits.md` § "Mid-run trigger — a 2× compute-deviation forces the vectorize check NOW" prescribes only the overhead-vs-FLOP signature check and (on a POSITIVE signature) the kill+relaunch-on-batched-twin calculus; the NEGATIVE-signature branch resolves to continue_as_is with no prompt to re-evaluate fleet WIDTH, even when a relaunch is already happening (crash-fix round) and the unit grid shards embarrassingly. CLAUDE.md's "saturate every provisioned GPU / wide-by-default" guidance covers plan-time width, not the mid-run deviation-relaunch point.
- **Confidence (emitter):** high
- verified-at-filing: `grep -n -iE 'deviation|re-?shard|width|relaunch|wider' .claude/rules/vectorize-many-cell-fits.md .claude/rules/crash-fix-rounds.md` → 20 hits across the 2 files; 0 hits describe width/re-shard re-evaluation in the Mid-run trigger's negative-signature branch (absence-of-guard claim — the 0-hit in-target result IS the evidence; the § Mid-run trigger section at vectorize-many-cell-fits.md:257-268 names only the signature check + vectorize path) (2026-07-15)

## Proposed change (candidate diff sketch — refine in planning)

```
.claude/rules/vectorize-many-cell-fits.md § Mid-run trigger:
+ On a NEGATIVE signature (FLOP-bound, engine already batched), the deviation
+ response is NOT automatically continue_as_is: if the phase is an
+ embarrassingly-parallel unit grid (independent cells/units) AND
+ checkpoint/restore machinery exists (or the deviation surfaces at a
+ relaunch point where restore is already happening), EVALUATE re-sharding
+ the REMAINING units across a wider fleet — state the arithmetic
+ (remaining box-hours ÷ candidate width + provision/stage/restore overhead
+ vs staying at width) in the epm:compute-deviation action note; wall-clock
+ is the scarce resource, credits are not (CLAUDE.md wide-by-default).
+ continue_as_is at fixed width requires the recorded arithmetic showing
+ widening does NOT win (e.g. remaining wall < re-shard overhead + round cost).
(mirror one-line pointer in .claude/rules/crash-fix-rounds.md relaunch section)
```

## Scope / surfaces

- Primary target: `.claude/rules/vectorize-many-cell-fits.md`
- Secondary: `.claude/rules/crash-fix-rounds.md` (relaunch decision pointer); check `workflow.yaml § pivot_criteria` (`compute_deviation_over_2x`) for a one-line consistency edit.
- Grep the workflow surface for the pattern before editing (`grep -rln 'compute_deviation_over_2x\|Mid-run trigger' .claude/ CLAUDE.md scripts/`) and update every hit; list them in the plan.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; `--check-lessons-index` passes if the rule's LESSONS.md row wording changes; if `workflow.yaml` or `CLAUDE.md` change, they stay consistent with the rule file.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` and carries a `workflow_fix_target:` Provenance line — it MUST NOT auto-route any of its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- workflow_fix_target: .claude/rules/vectorize-many-cell-fits.md
- fingerprint: e20e0f5d5faf

Orchestrator-observed gap (no formal candidate block; synthesized per the workflow-fix-on-bug prose-followup rule). Evidence trail: #1092 events.jsonl — epm:failure (p6-pilot-gate-abort, 2026-07-15T05:23:18Z), the two epm:compute-deviation markers (ratio 2.57, signature_check: negative, action: continue_as_is), and the 05:56–06:35Z relaunch cluster-launched markers at unchanged width 4.

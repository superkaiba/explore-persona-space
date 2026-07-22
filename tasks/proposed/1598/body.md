---
title: 'daily-fix: teammate report SendMessage as final turn action'
kind: infra
tags:
- wf-fix
- wf-fix-fp:acbf7f96dc7f
- daily-auto-filed
created_at: '2026-07-22T06:47:30Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-21 problem sweep (route 2): fan-out teammates go idle
  without delivering their report SendMessage; six nudges across three sessions in
  one morning'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-21 from 6 nudge events across 3 chat sessions (transcripts e0e69399 / f4b1d707 / 12462773).

## Goal

Make fan-out teammate spawns deliver their reports reliably: a teammate's FINAL action before ending its turn must be the SendMessage carrying its report; an idle ping without a delivered report is an incomplete turn.

## Workflow gap

- **Bug observed:** fan-out teammates (docs/tasks/papers scouts) finished their work but went idle WITHOUT SendMessage-ing their findings; the orchestrator had to explicitly nudge each — 6 nudge SendMessage firings across 3 sessions in one morning (1 in e0e69399, 2 in 12462773, 3 in f4b1d707; counted as deduplicated SendMessage tool_use invocations asking a teammate to deliver its report), ~1-3 min added latency per session + 6 extra orchestrator turns.
- **Why it is a workflow gap:** the teammate-coordination guidance (CLAUDE.md § "Orchestrator vs subagent re-invocation") covers channel choice and idle-ping semantics but never states the delivery contract for the TEAMMATE side — nothing requires the report SendMessage to be the final act of the turn, so teammates end turns with work done and report undelivered.
- **Confidence:** high (6 same-morning firings).
- verified-at-filing: `grep -c 'FINAL action' CLAUDE.md` → 0 (absence claim, in-target 0-hit is the evidence, 2026-07-22); the teammate-coordination bullet exists at CLAUDE.md:101 (channel-scope note added by tonight's route-1 commit `f985fec873` — a DIFFERENT clause; no overlap with this delivery-contract gap).

## Proposed change (candidate diff sketch — refine in planning)

Add to the CLAUDE.md teammate-coordination bullet (and/or the spawn-brief convention the fan-out orchestrators follow): when a spawn brief asks a teammate to report via SendMessage, the teammate's FINAL action before ending its turn MUST be that SendMessage; orchestrator-side, treat an idle notification with no delivered report as an incomplete turn to nudge once, then read the agent result directly.

## Scope / surfaces

- Primary target: `CLAUDE.md` (§ Orchestrator vs subagent re-invocation, teammate-coordination bullet)

## Constraints / invariants

- Workflow-surface only; keep the bullet's existing (a)/(b)/(c) structure and incident citations intact.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: acbf7f96dc7f

- workflow_fix_target: CLAUDE.md

Origin evidence: transcripts e0e69399 (06:05:52Z "The tasks-sweep agent went idle without delivering its report — pinging it"), f4b1d707 (06:05:36Z "All three sweepers nudged to deliver their reports"), 12462773 (06:05:09Z + 06:07:25Z).

---
title: 'daily-fix: act on consistency edit-locus WARN vs live siblin'
kind: infra
tags:
- wf-fix
- wf-fix-fp:cae7dcc962ca
- daily-auto-filed
created_at: '2026-07-28T07:03:22Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): #1718''s consistency-checker
  WARNed at plan time (11:47Z) of a same-file edit-locus conflict with a live sibling;
  the WARN was folded and unheeded, and the exact conflict materialized at merge (14:28Z)
  costing the conflict-resolution round'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Session 4d615395 (#1718), 2026-07-27 (miner D P6; WARN 11:47Z, conflict 14:28Z).

## Goal

An edit-locus WARN naming a live sibling should change merge sequencing, not just be folded into plan prose.

## Workflow gap

- **Bug observed:** the Phase-2 consistency-checker explicitly warned that this plan edits the same file region as a live sibling task; the warning was folded as prose and the predicted conflict materialized at Step 10d merge time, costing the resolution round on the fleet's hottest file.
- **Why it is a workflow gap:** no orchestrator duty consumes the edit-locus WARN — it informs the plan text and nothing else; the cheap consequence (hold the merge until the named sibling lands, then rebase) is never taken. NOTE: serializing merges adds latency — the planner should weigh scope (only same-file-region WARNs against siblings at reviewing/later) and may deflect with a reasoned no-change if the latency cost dominates.
- **Confidence (emitter):** medium
- verified-at-filing: unverified hypothesis — verify at plan time: the WARN->conflict chain is read from the session's own transcript (miner: inferred — not probed); confirm the consistency-checker WARN text + the conflict hunks from #1718's events/plan before implementing.

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/issue/SKILL.md` (post-plan-approval or Step 10d pre-merge): on a consistency edit-locus WARN naming a live sibling task at status reviewing or later, record the sibling id and HOLD the Step 10d merge until that sibling posts epm:merged, then rebase and proceed. Bounded: applies only to same-file WARNs, with a hard cap (e.g. wait <= one gate cycle) so a stuck sibling never wedges the task.

## Scope / surfaces

- Primary target: `.claude/skills/issue/SKILL.md`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: cae7dcc962ca

- workflow_fix_target: .claude/skills/issue/SKILL.md

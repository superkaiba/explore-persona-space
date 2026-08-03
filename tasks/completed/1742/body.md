---
title: 'daily-fix: Step 9c urgent-park emission made mechanical'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f0ffb697062f
- daily-auto-filed
created_at: '2026-07-28T06:58:42Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 2): on the mandate''s first
  live day (landed via #1713 at 09:41Z), at least 7 sessions (#1724 #1725 #1729 #1732
  #1735 #1719 #1739) classified the planner.md main-red as pre-existing and NONE emitted
  the mandatory routable urgency: main-red candidate; the red lived ~17h and was only
  routed by the nightly /daily'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-27 problem sweep (transcript mining, 44 in-window
transcripts). Sessions d07cbe1f/6666ef8f/513fca53/d71a9e1e/f56bbb5d/fe17b703 + #1739's events, 2026-07-27 (miners B/D/E/G/H convergent).

## Goal

Make the Step 9c urgent-park emission mechanical (tool-printed trigger line) instead of a prose duty sessions forget.

## Workflow gap

- **Bug observed:** seven sessions with DIRECT Step 9c evidence of the live planner.md 40900>40000 main-red each stripped it as pre-existing and terminated with prose-only dispositions ('Needs human eyeball', 'the nightly sweep can route it') — zero `urgency: main-red` candidate blocks emitted (transcript-wide token scans, miner-probed per session). The #1713 mandate was in the SKILL text each session loaded.
- **Why it is a workflow gap:** a prose MUST that fires at the busiest moment of the pipeline (gate verdict) reliably loses to recency; the compare tool already computes exactly the triggering condition (stripped pre-existing red on a workflow-surface node) and can print the demand line mechanically.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -c 'URGENT-PARK-REQUIRED' scripts/step9c_baseline.py` -> 0, compose time. Emission-skip evidence: miner-probed transcript scans (0 emitted parks across the 7 sessions) + `tasks/completed/{1724,1725}/events.jsonl` scans -> 0 candidate markers.

## Proposed change (candidate diff sketch — refine in planning)

(a) `scripts/step9c_baseline.py compare`: when a stripped pre-existing failure's node matches a workflow-surface pattern (tests/test_workflow*, workflow_lint), print `URGENT-PARK-REQUIRED: <node id>` in the verdict JSON/stdout. (b) `.claude/skills/issue/SKILL.md` Step 9c Verdict block: on seeing that line, emitting the routable parked candidate block (or verifying one already exists on the events stream of ANY task, via a bounded grep) is a numbered mandatory step, not prose.

## Scope / surfaces

- Primary targets: `scripts/step9c_baseline.py`, `.claude/skills/issue/SKILL.md` (Step 9c Verdict block)
- Companion context: #1741 (sweep park-predicate fix) + #1740 (the planner.md red itself) — this task is the third leg: emission.

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run + `--check-asks` pass on touched files;
  ruff passes where applicable.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT
  auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: f0ffb697062f

- workflow_fix_target: scripts/step9c_baseline.py, .claude/skills/issue/SKILL.md

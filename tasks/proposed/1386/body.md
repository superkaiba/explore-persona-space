---
title: 'daily-fix: hub Xet queue-full transient retry'
kind: infra
tags:
- daily-auto-filed
created_at: '2026-07-16T07:19:41Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-15 problem sweep (route 2): Xet maximum queue size
  reached not classified transient'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-15 parked-candidate routing pass (Step C) from a recursion-guard-parked prose follow-up on task #1359 (emitting agent: Methodology critic, Phase 2). NOT a workflow-surface fix — `src/` library code (the #1359 session correctly noted it is out of workflow-fix scope); filed as an ordinary infra fix task.

## Goal

Add the Xet "maximum queue size reached" error text to `_is_transient_upload_error`'s substring list in `src/explore_persona_space/orchestrate/hub.py` (~L854), so the INNER upload retry absorbs the #1315 Xet transient class for every caller — weighing the docstring's permanent-failure budget-burn tradeoff.

## Workflow gap

- **Bug observed:** #1315's upload hit the Xet "maximum queue size reached" transient, which `_is_transient_upload_error` does not classify as transient — the inner retry does not absorb it, so each caller must handle it ad hoc.
- **Why it is a workflow gap (analogue):** upload transient classification is shared infrastructure; a missing transient class re-surfaces as per-task upload flakiness.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'maximum queue size' src/explore_persona_space/orchestrate/hub.py` → 0 hits (absence claim — the 0-hit result IS the evidence); `grep -n '_is_transient_upload_error' src/.../hub.py` → definition at :854 (2026-07-16 UTC)

## Proposed change (candidate diff sketch — refine in planning)

Append the Xet queue-full substring to the transient list in `_is_transient_upload_error`, with a test; the planner weighs the docstring's stated tradeoff (a permanent failure matching the substring burns retry budget).

## Scope / surfaces

- Primary target: `src/explore_persona_space/orchestrate/hub.py`
- Secondary: the matching test file for upload transient classification.

## Constraints / invariants

- Fail-fast discipline: only genuinely transient classes belong in the substring list.

## Provenance

parked prose follow-up (verbatim, from #1359 events.jsonl 2026-07-15T19:30:47Z): "Suggestion: add the Xet 'maximum queue size reached' text to _is_transient_upload_error's substring list (src/explore_persona_space/orchestrate/hub.py:846-884) so the INNER retry absorbs the #1315 Xet class for every caller — weighing the docstring's permanent-failure budget-burn tradeoff."

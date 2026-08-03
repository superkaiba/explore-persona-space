---
title: 'daily-fix: label unverified claims in route-2 bodies'
kind: infra
tags:
- wf-fix
- wf-fix-fp:f15904190f8c
- daily-auto-filed
created_at: '2026-07-25T06:50:00Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-24 problem sweep (route 2): Four of six route-2 bodies
  in the 07-23 wave carried refutable factual premises - wrong incident root-cause
  in 1655, wrong API-behavior assumption in 1662, refuted timing premises in 1646
  and 1660 - each burning a fact-checker or critic correction round in the spawned
  session'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-24 problem sweep (sessions 1a210c3c/6dbde886/6d2780e7/d92dbcd2, tasks #1655/#1662/#1646/#1660).

## Goal

Filed bodies should carry honest epistemic labels on claims the filer could not mechanically verify, so spawned sessions treat them as hypotheses.

## Workflow gap

- **Bug observed:** 4 of 6 sessions in the 07-24 infra wave had to correct filed-body premises: #1655 (fact-check: "commit b66910d748 shows the account list was intact" — filed root-cause wrong), #1662 (fact-checker: bare-form terminate DOES sweep suffixed pods — filed assumption wrong), #1646 (filed "~200s alone" premise refuted by measurement), #1660 (filed "3600s" a mis-recollection vs train_behavior_fullft.py:650). Counted from each session's pipeline-record messages. Each burned a correction round pre-approval.
- **Why it is a workflow gap:** the verified-at-filing mandate covers grep-verifiable claims; timings, incident mechanisms, and API-behavior premises are asserted as fact with no labeling convention, so planners inherit them as premises.
- **Confidence (emitter):** high.
- verified-at-filing: n/a — behavioral observation from the four sessions' pipeline records (task events #1655/#1662/#1646/#1660, 2026-07-24); the SKILL.md route-2 section (read 2026-07-25) has no hypothesis-labeling clause (absence bind, context read).

## Proposed change (candidate diff sketch — refine in planning)

In `.claude/skills/daily/SKILL.md` route 2 (and the wf-fix body template note): claims not verified mechanically at compose time — timings, incident root-causes, API behavior — are written as `unverified hypothesis — verify at plan time: <claim>`, never as bare fact. One sentence + example in the template.

## Scope / surfaces

- Primary target: `.claude/skills/daily/SKILL.md` (route-2 composition; optionally `.claude/rules/workflow-fix-on-bug.md` body template)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py` no-flags run passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — recursion guard applies.

## Provenance

- fingerprint: f15904190f8c

- workflow_fix_target: .claude/skills/daily/SKILL.md

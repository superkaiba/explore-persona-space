---
title: 'daily-fix: extend ad-hoc provenance rule (2 clauses)'
kind: infra
tags:
- wf-fix
- wf-fix-fp:07b219d40592
- daily-auto-filed
created_at: '2026-07-17T06:57:26Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-16 problem sweep (route 2): two gaps in the ''Ad-hoc
  results summaries state per-arm provenance'' bullet surfaced today: (a) the #1345
  examples dashboard silently display-substituted ''ARIA''->''Assistant'' in text
  presented as model generations (user had to ask whether the story really said ARIA);
  (b) a #1092 chat table compared prefix-R2 vs context-R2 arms scored against DIFFERENT
  targets and the user had to flag ''these results'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-16 from two user corrections in transcript mining (09f28ede ~08:35Z; ea3a7991 ~06:08Z).

## Goal

Close the two disclosure gaps today's user corrections exposed in the ad-hoc results-summary rule.

## Workflow gap

- **Bug observed:** two gaps in the 'Ad-hoc results summaries state per-arm provenance' bullet surfaced today: (a) the #1345 examples dashboard silently display-substituted 'ARIA'->'Assistant' in text presented as model generations (user had to ask whether the story really said ARIA); (b) a #1092 chat table compared prefix-R2 vs context-R2 arms scored against DIFFERENT targets and the user had to flag 'these results seem wrong' before a matched-target comparison was built
- **Why it is a workflow gap:** Both incidents required the user to detect a silent presentation choice; the rule exists to make those disclosures automatic.
- **Confidence (emitter):** high (two user corrections in one day)
- verified-at-filing: `grep -n 'Ad-hoc results summaries' CLAUDE.md` -> L185 (bullet present); `grep -c 'display-substitution\|matched-target' CLAUDE.md` -> 0 (absence of both clauses)

## Proposed change (candidate diff sketch — refine in planning)

extend the CLAUDE.md ad-hoc provenance bullet with: any display-substitution of raw generation text needs inline per-passage disclosure (dashboards included), and any cross-arm metric table shown in chat states matched-target/matched-corpus (or names the mismatch)

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Grep the workflow surface for the pattern before editing and update every hit; list them in the plan.

## Constraints / invariants

- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under a `workflow_fix_target:` Provenance line — it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 07b219d40592


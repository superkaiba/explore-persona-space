---
title: 'daily-fix: ad-hoc result summaries state data provenance'
kind: infra
tags:
- wf-fix
- wf-fix-fp:47be51273896
- daily-auto-filed
created_at: '2026-07-08T07:00:14Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-07 problem sweep (route 2): Thomas had to ask twice
  in one evening whether #825 separator-round completions were on-policy (eaec45a8
  16:24Z "did we allow the model to generate on policy for the separators?"; 16:51Z
  "this is all with on-policy completions?") — ad-hoc summaries omitted per-arm completion
  provenance. Related: 0103f471 01:09Z ("wait this is on single stochastic sample?")
  — rollout-count/sampling recipe absent fr'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-07-07 (route 2) from the nightly transcript problem sweep.

## Goal

add a one-line rule: any ad-hoc results summary shown to the user states per-arm data/completion provenance (on-policy vs teacher-forced vs canned; generation recipe) in its setup line

## Workflow gap

- **Bug observed:** Thomas had to ask twice in one evening whether #825 separator-round completions were on-policy (eaec45a8 16:24Z "did we allow the model to generate on policy for the separators?"; 16:51Z "this is all with on-policy completions?") — ad-hoc summaries omitted per-arm completion provenance. Related: 0103f471 01:09Z ("wait this is on single stochastic sample?") — rollout-count/sampling recipe absent from ad-hoc figures.
- **Why it is a workflow gap:** Provenance questions recur precisely because ad-hoc artifacts sit outside the v4 spec's what-is-plotted-EXACTLY discipline.

## Proposed change

One-line CLAUDE.md addition near the clean-result / reporting guidance; reviewer judges exact wording/placement.

## Scope / surfaces

- Primary target: `CLAUDE.md`
- Grep the workflow surface for the pattern before editing and update every hit.

## Provenance

- Evidence: eaec45a8 16:24Z + 16:51Z; 0103f471 01:09Z.

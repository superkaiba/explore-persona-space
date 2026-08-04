---
title: 'daily-fix: projected-vs-realized + fallback-slice claims'
kind: infra
tags:
- wf-fix
- wf-fix-fp:38d25ba714a8
- daily-auto-filed
created_at: '2026-08-03T07:05:12Z'
has_clean_result: false
origin_prompt: '/daily 2026-08-02 problem sweep (route 2): Four verify-before-asserting
  failures in one day that existing clauses do not name: (a) ''we never ran 3.2M batch
  calls'' asserted post-/compact about a real completed $9,505 wave -- composed off
  the approval/projection marker; refuted only after Thomas''s ''are you sure - dig
  deeper (with a subagent)'' (session 0ac15c23, 14:02-14:42Z). (b) A hallucination
  NQ-Open headline ''+0.220'' was read from a FALL'
workflow: v1
---
## Overview / Motivation

Auto-filed by /daily 2026-08-02 (route 2: behavior/logic change -> independent review) from the nightly problem sweep (miners 2/3, sessions 0ac15c23/55419495, tasks #1773/#1739).

## Goal

Claims about what ran, at what size, and what is banked are grounded in completion artifacts and relocation sweeps at compose time.

## Workflow gap

- **Bug observed:** Four verify-before-asserting failures in one day that existing clauses do not name: (a) 'we never ran 3.2M batch calls' asserted post-/compact about a real completed $9,505 wave -- composed off the approval/projection marker; refuted only after Thomas's 'are you sure - dig deeper (with a subagent)' (session 0ac15c23, 14:02-14:42Z). (b) A hallucination NQ-Open headline '+0.220' was read from a FALLBACK slice (U=250,L=2,500) because operating-slice cells didn't exist yet; real operating-slice deltas +0.018/-0.003 (session 55419495, corrected 08:34Z). (c) Three consecutive wrong claims about SAE label coverage from a single local-dir count; the full 127,605/128,512 inventory surfaced only after 3 corrections in ~4 min (0ac15c23, 03:04-03:11Z). (d) A ~$1.5-3k re-judge was priced to the user on a false 'per-draw scores are averaged away' absence claim; the relocation sweep that refuted it ran only after dispatch (55419495, 14:27-15:35Z).
- **Why it is a workflow gap:** The existing arms cover numbers, absence claims, and coverage extremum sweeps, but not the projection-vs-realization ambiguity (approval markers carry the same figures as completion records) nor fallback-slice labeling, and the absence-sweep is not sequenced BEFORE spend questions.
- **Confidence (emitter):** high (all four incidents probed by miners with verbatim user rows + refuting artifacts)
- verified-at-filing: `grep -c -iE 'projected.vs.realized|completion artifact' CLAUDE.md` -> 0; `grep -c -iE 'fallback slice|operating slice' CLAUDE.md` -> 0; the relocation-sweep clause exists but has no before-pricing sequencing (read at compose time).

## Proposed change (refine in planning)

extend the verify-before-asserting arm: (i) PROJECTED-VS-REALIZED -- any claim about whether/at-what-size a batch wave or run executed is read from the run's COMPLETION artifact (drop report, batch-dispatch checkpoint, spend record), never the approval/projection marker (the same figure legitimately appears as both projection and fact); (ii) a headline number measured on a FALLBACK/partial slice names the slice and the 'fallback' label in its setup line; (iii) coverage/width claims about banked artifact SETS (labels, judgments, per-draw scores) require the relocation sweep BEFORE any regeneration is priced or an AskUserQuestion spends the user's attention on it.

## Scope / surfaces

- Primary target: `CLAUDE.md`

## Constraints / invariants

- Workflow-surface rules apply; `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- This session runs under `EPM_WORKFLOW_FIX_SESSION=1` (Provenance `workflow_fix_target:` line) -- it MUST NOT auto-route its own subagents' workflow-fix candidates (recursion guard).

## Provenance

- fingerprint: 38d25ba714a8

- workflow_fix_target: CLAUDE.md


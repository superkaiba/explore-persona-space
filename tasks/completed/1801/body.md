---
title: 'daily-fix: llm-judging — rule-23 point, tally split, rubric '
kind: infra
tags:
- wf-fix
- wf-fix-fp:2b526e83d4c5
- daily-auto-filed
created_at: '2026-07-29T07:11:57Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): three 2026-07-28 judge-instrument
  gaps: max_tokens=400 still truncation-censored 5.4% of #1739 sycophancy draws (extra
  recovery re-judge pass); a healthy judge wave was killed on a misread 28.9% ''drop
  rate'' that was mostly instructed REFUSAL tokens folded into an undifferentiated
  tally in the driver log; #1482''s judged persona_related axis was a rubric artifact
  (20/40 ''persona'' features were bare'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Sources: group-A P4 + P11, group-J P2.

## Goal

Fold three measured 2026-07-28 judge-instrument lessons into the llm-judging rule + the batch-judge driver log.

## Workflow gap

- **Bug observed:** (a) #1739: at max_tokens=400 (above rule 23's ~300 floor) 5.4% of sycophancy draws were still truncation-censored, costing a recovery re-judge pass (~10k items). (b) A healthy judge wave was killed and relaunched (~6 min) on a misread 28.9% 'drop rate' — mostly instructed-REFUSAL tokens the rubric itself requested, folded into one undifferentiated tally in the driver's log line even though `judge_tallies` already splits the category. (c) #1482: the judged `persona_related` axis shipped a rubric artifact — 20/40 top 'persona' features were bare language-ID features; the user caught it ~2h later and rubric amendments were posted in-flight (the durable instrument is #1773). (Percentages are session tallies — verify at plan time.)
- **Why it is a workflow gap:** rule 23's floor is rubric-dependent and the rule lacks the newly measured point; the driver log hides an existing tally split; section C's rubric guidance has no name-your-confusable-neighbors rule for judged CATEGORY axes.
- **Confidence (emitter):** medium
- verified-at-filing: `grep -n 'give any reasoning rubric' .claude/rules/llm-judging.md` → line 197 (~300 floor present, no 400-token measured point); `grep -n 'judge_tallies' src/explore_persona_space/eval/batch_judge.py` → present (split exists, unsurfaced in the driver line) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

Three small edits: rule-23 sentence + measured citation; one driver log-line change surfacing REFUSAL as its own category; one section-C guideline sentence with the #1482 example.

## Scope / surfaces

- Primary targets: `.claude/rules/llm-judging.md`, `src/explore_persona_space/eval/batch_judge.py` (log line only)

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 2b526e83d4c5

- workflow_fix_target: .claude/rules/llm-judging.md


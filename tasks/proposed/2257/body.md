---
title: Characterize the cross-trait hallucination increase from normal-code training
  (issue 2221 insecure_code/normal cell)
kind: analysis
tags: []
created_at: '2026-08-12T23:38:11Z'
has_clean_result: false
parent_id: 2221
origin_prompt: 'analyzer Step 6.5 free-analysis proposal on #2221 (2026-08-12): ''Characterize
  the cross-trait hallucination increase from normal-code training using the existing
  trait-eval rollouts + judge scores'''
workflow: v1
---
# Characterize the cross-trait hallucination increase from normal-code training (issue 2221 insecure_code/normal cell)

## Goal

Determine which mechanism explains the cross-trait hallucination increase in issue #2221's *normal-version* insecure-code fine-tune — graded hallucination 73.5 vs base 23.5, with no vulnerable code in its training mix: (a) an emergent-misalignment-flavored cross-trait effect of code training, or (b) generic drift from the grid's largest training mix (3,357 rows; the same size confound that dominates the parent's headline). 0 GPU-h: the disambiguation uses existing trait-eval rollouts + judge scores only — no new training, no new generation.

## Context

- Parent #2221 (real-data twin of the Persona Vectors finetuning-shift monitoring suite) observes the anomaly in its Results ("Only hallucination has usable outcome range") and names both candidate mechanisms; the analyzer tagged this follow-up `cost_class: free-analysis`, `est_gpu_hours: 0`, `headline_affecting: no`, `question_relation: substantially-different` (new question → child task; auto-run bands are same-question only, so this was filed, not run).
- Redundancy screen: **not-redundant** (`epm:followup-value-critique v1` on #2221, 2026-08-12). Nearest existing title, #459, is archived without a clean result and differs in construct (harmful EM datasets → misalignment profile, vs a benign control mix → hallucination).

## Inputs (verified present at filing)

- `data/issue_2221/p6/eval_rollouts/insecure_code_normal.json` (worktree `issue-2221`)
- `data/issue_2221/p6/judge/insecure_code_normal_hallucination.json`
- HF `superkaiba1/explore-persona-space-data` → `issue2221_realtwin/raw_completions/trait_eval/` (all 25 models' trait-eval pools, judge raw included)

## Candidate discriminating reads (sketch, not a plan)

- Dose curve within the insecure-code family: hallucination score across normal/mild/severe (all ~3,357 rows) vs the hallucination family's own cells (~1,533 rows) — under mechanism (b) score should track mix size across families; under (a) the code family should sit above the size trend.
- Content read: what the 73.5 cell confabulates (code-adjacent confabulation vs the same generic confabulation profile as the severe-hallucination fine-tune), from the stored rollout text.
- Per-prompt-family firing pattern overlap with the severe-hallucination fine-tune's judge-positive set.

## Provenance

Filed 2026-08-12 by the #2221 orchestrator from the analyzer's Step 6.5 free-analysis proposal (verbatim: "Characterize the cross-trait hallucination increase from normal-code training using the existing trait-eval rollouts + judge scores"). Filing ≠ spawning: left at `proposed` for manual triage per the substantially-different routing rule.

---
title: 'planner/critic: a torch-MLP LOCO fit is NOT a cheap CPU-only analysis step'
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:16Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: a torch-MLP / gradient-descent LOCO fit was mis-routed to the VM CPU default under the "CPU-only analysis phases" rule.

## Goal

Stop mis-routing gradient-descent fits to the VM CPU default under the "CPU-only analysis phases" rule — a torch-MLP LOCO fit is GPU-worthy, not cheap closed-form stats.

## Problem (from /daily 2026-06-27)

#658 `_fit_mlp_loco` (300-epoch AdamW per cell) ran on CPU, needlessly GPU-starved; Thomas flagged "why are we running this on CPU?" (session f721a47c). The "CPU-only analysis phases" rule deliberately keeps cheap closed-form stats (bootstrap, permutation) off pods, but a torch-MLP / gradient-descent fit is not cheap stats — it benefits materially from a GPU.

## Proposed change

Add a planner.md §9 + critic.md Methodology-lens note that a torch-MLP / gradient-descent LOCO fit (as opposed to a closed-form bootstrap/permutation analysis) is GPU-worthy and routes to a pod / GCP, NOT the VM CPU default. The planner sizes the fit's compute character; the critic REVISEs a gradient-descent fit silently placed on the VM CPU.

## Scope / target files

- `.claude/agents/planner.md`
- `.claude/agents/critic.md`

## Constraints

- Workflow-surface only — no experiment code, configs, or tasks/.
- Lint gate green (`workflow_lint.py`).
- Keep consistent with the existing "CPU-only phases don't hold GPU pods" rule in CLAUDE.md (refine, don't contradict).

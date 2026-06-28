---
title: 'analyzer: confirm all planned control arms present before authoring a verdict'
kind: infra
tags:
- daily-held
created_at: '2026-06-28T07:13:24Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog: the analyzer authored a verdict on a partial grid missing a control/baseline arm.

## Goal

Prevent the analyzer authoring takeaways on a partial grid that is missing a planned control / baseline arm.

## Problem (from /daily 2026-06-27)

#658 near-premature "complete": the apparent 3/10 Betley PASS was a multiple-comparison artifact beaten by a random-projection control (0/10 after the control landed), but the analyzer ran on a partial grid before the control arm landed (session 3149da48). Authoring a verdict before the control arm exists nearly shipped a false positive.

## Proposed change

analyzer.md guard — before authoring takeaways, assert that every planned control / baseline arm is present in `eval_results/`. A missing control arm BLOCKS the verdict: park (don't author), surface the missing arm, wait for it to land. This is the planned-vs-actual-coverage discipline applied at verdict time, not just at clean-result-critic time.

## Scope / target files

- `.claude/agents/analyzer.md`

## Constraints

- Workflow-surface only — no experiment code, configs, or tasks/.
- Lint gate green (`workflow_lint.py`).
- Keep consistent with the existing planned-vs-actual coverage rules (verify_task_body.py check 11b, clean-result-critic planned-vs-actual lens) — this is the upstream analyzer-side guard.

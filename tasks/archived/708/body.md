---
title: 'verify #664 base tokenizer — Mistral-Small regex warning on a Qwen-2.5-7B
  project'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-06-28T07:14:07Z'
has_clean_result: false
origin_prompt: /daily 2026-06-27 held backlog
---
## Overview / Motivation

Filed from /daily 2026-06-27 held backlog (NEEDS THOMAS — results-integrity call): #664 prints a Mistral-Small tokenizer warning on a Qwen-2.5-7B project.

## Goal

Confirm #664 is evaluating the intended Qwen-2.5-7B base, not a mis-tagged merge / wrong base model.

## Problem (from /daily 2026-06-27)

#664 prints a Mistral-Small-3.1-24B tokenizer "incorrect tokenization" regex warning on EVERY eval cell, on a Qwen-2.5-7B project (session 09e41486). This suggests a mis-tagged merge or a wrong base model being loaded — which would invalidate #664's evals.

## Proposed change

NEEDS THOMAS — results-integrity call. Verify the base tokenizer is the intended Qwen-2.5-7B before trusting #664's evals. Do NOT silently set `fix_mistral_regex=True` — that would paper over a possible wrong-model bug rather than diagnose it. Investigate: which checkpoint/merge #664 loaded, whether its tokenizer is Qwen or Mistral, and whether the eval numbers are on the intended model.

## Scope / target files

- Investigation on #664 (results integrity).
- `.claude/` only IF a guardrail results (e.g. a preflight base-tokenizer assertion).

## Constraints

- DO NOT auto-dispatch — PM surfaces to Thomas; could invalidate #664's results.
- No silent `fix_mistral_regex=True` workaround.
- If a guardrail lands later: workflow-surface only, lint gate green.

---
title: 'Correct #664 vLLM gotchas: chunk-it red herring + r13 tokenizer-reload root
  cause'
kind: infra
tags: []
created_at: '2026-06-28T19:37:30Z'
has_clean_result: false
origin_prompt: 'PM-session audit of #664: chunk-it mis-diagnosis was promoted to gotchas.md
  while the r13 tokenizer-reload root cause was never durably captured. Hold on predicate-664-lands;
  correct via #712''s supersedes path when #664 finishes.'
---
## Overview / Motivation

During #664's multi-relaunch vLLM saga, the durable record captured the WRONG
root cause: `.claude/rules/gotchas.md` carries a "single large `llm.generate()`
deadlocks the v1 EngineCore — chunk it" entry attributed to #664, but for #664
chunking + `enforce_eager` + `prefix_caching=0` ALL failed (v16 hung
identically). The ACTUAL r13 root cause was per-call
`AutoTokenizer.from_pretrained` reloads inside `_render()` starving generation
(fixed with `functools.lru_cache`) — "NOT a vLLM deadlock" per the v17
`run-launched` note. That correct cause is in no durable file.

The mechanism to fix this cleanly now exists (task #712 added
root-cause-confirmed capture + the `supersedes:` field).

## Goal

Correct the durable record: annotate the chunk-it `gotchas.md` entry as a
red herring for #664, and add the r13 tokenizer-reload root cause (with the
"if chunking + enforce_eager + prefix_caching=0 all fail, profile a single
`_render()` call — suspect per-call tokenizer/model reloads" diagnostic),
using the #712 `supersedes:` path.

## HOLD — predicate-664-lands

Do NOT run until #664 reaches a terminal state. The #664 vLLM root cause has
already evolved 3× (prefix-caching → CUDA graphs → V1 engine → tokenizer
reload); writing the "final" correction while the session is still
root-causing risks documenting a cause that changes again. Wait for #664 to
land, then capture the confirmed cause.

## Scope / surfaces

- `.claude/rules/gotchas.md` — annotate the chunk-it / EngineCore-deadlock
  entry (red-herring caveat for #664) + add the tokenizer-reload root cause.
- Prefer routing through the #712 `supersedes:` failure-lesson path if the
  #664 session posts the corrected lesson on landing (then this task verifies
  the durable record is correct rather than re-authoring it).

## Constraints / invariants

- Workflow-surface only.
- Keep the genuinely-real vLLM lessons (crash-orphan EngineCore exact-PID reap,
  #601/#653) intact — only the #664 large-batch-deadlock attribution is the
  red herring.

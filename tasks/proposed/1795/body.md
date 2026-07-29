---
title: 'daily-fix: validate batch-judge custom_id grammar pre-submit'
kind: infra
tags:
- wf-fix
- wf-fix-fp:584cb811c222
- daily-auto-filed
created_at: '2026-07-29T07:08:44Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-28 problem sweep (route 2): #1739''s evil-lane judge
  wave was killed by an anthropic 400 on a custom_id containing a tilde — no pre-submit
  grammar validation exists, and the crash surfaced inside a detached driver only
  via log monitoring'
workflow: v1
---
## Overview / Motivation

Auto-filed by the /daily 2026-07-28 problem sweep (transcript miners over the day's 55 session transcripts). Source: group-A P3 (miner-probed).

## Goal

Fail loud at item-construction time on Batch-API custom_id grammar violations instead of dying mid-wave on an anthropic 400.

## Workflow gap

- **Bug observed:** #1739's judge wave for the evil lane died on an anthropic HTTP 400: a custom_id carried a `~k` segment, violating the API's `^[a-zA-Z0-9_-]{1,64}$` grammar. The crash happened inside a detached driver and was caught only by log monitoring (~10 min fix/relaunch; the fixed id shape `f"{context_id}_k{k:02d}"` landed on the issue worktree).
- **Why it is a workflow gap:** judge_dispatch.py constructs and submits custom_ids with no grammar check — a malformed id is a deterministic 400 (an invalid_request_error, correctly NEITHER retried NOR dropped per llm-judging rule 24(iii)), so the right fix is pre-submit validation in the request builder.
- **Confidence (emitter):** high (probed)
- verified-at-filing: `grep -n 'custom_id' src/explore_persona_space/eval/judge_dispatch.py` → construction/threading sites at 29/36/219/429/499; `grep -c 'a-zA-Z0-9_-' .../judge_dispatch.py` → 0 (no grammar pattern anywhere in the module) (2026-07-29 UTC).

## Proposed change (candidate diff sketch — refine in planning)

A small `_validate_custom_id()` at JudgeItem construction (and/or at `dispatch_judge_items` entry) raising ValueError naming the offending id; unit test with a tilde id.

## Scope / surfaces

- Primary target: `src/explore_persona_space/eval/judge_dispatch.py`

## Constraints / invariants

- Workflow-surface only — never experiment code, `configs/`, or `tasks/`.
- `scripts/workflow_lint.py --check-asks` passes; ruff on touched files passes.
- The spawned session runs under a `workflow_fix_target:` Provenance line —
  recursion guard applies (it parks, never auto-routes, its own subagents'
  workflow-fix candidates).

## Provenance

- fingerprint: 584cb811c222

- workflow_fix_target: src/explore_persona_space/eval/judge_dispatch.py


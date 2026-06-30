---
title: 'workflow-fix: stream-reduce activation spans, never materialize all N'
kind: infra
tags:
- wf-fix
- wf-fix-fp:997f657054ce
created_at: '2026-06-30T21:50:49Z'
has_clean_result: false
origin_prompt: what is the infra friction and how can we avoid it in the future -
  send a subagent to investigate
---
## Overview / Motivation
Auto-filed from the predictor-line (#742/#761/#763) infra-friction retrospective (subagent, 2026-06-30). Related in-flight: #770 (RunPod no-port wedge billing), #681 (data-disk bind cutover), #667 (GCP frozen-phase), #769 (CUDA-frag OOM).

## Goal
Add a stream-reduce rule: an analysis reading N multi-GB per-context .pt blobs from HF MUST use the canonical _HfStreamSpanSource (download->process->delete, peak one ctx) with the _assert_attn_stream_equivalence gate, never download all N locally; cite #761's 220 GB refill.

## Workflow gap
- Bug observed: #761 re-downloaded the full 220 GB UltraChat answer_spans (4.4GB x 50) into a local cache and refilled /, despite the canonical streaming helper _HfStreamSpanSource already existing in scripts/issue658_fit_predictors.py (download->process->delete, peak ~1 ctx) — a built-but-stranded helper not reused by the sibling bootstrap.

## Scope / surfaces
- Target: .claude/rules/code-style.md

## Constraints / invariants
- Workflow-surface only; ruff + workflow_lint pass; add a test where practical.

## Provenance
- workflow_fix_target: .claude/rules/code-style.md
- fingerprint: 997f657054ce

Raised by the predictor-line infra retrospective (2026-06-30).

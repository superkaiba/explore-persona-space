---
title: Probe base-model log-prob of single-token marker candidates vs legacy [ZLT]
kind: analysis
tags: []
created_at: '2026-05-26T20:28:02Z'
has_clean_result: false
---
## Goal

Measure base-model log-prob of single-token marker candidates (`※`, `¶`, `ϟ`) and the legacy multi-token `[ZLT]` marker under `Qwen-2.5-7B-Instruct`, at end-of-completion positions across a small sample of personas × questions, to validate that `※` is a low-prior choice for future log-prob-based persona-marker experiments.

## Why analysis, not experiment

This is a methodology probe to validate a tooling choice. The output is a measurement that informs which marker token to use in future research experiments. No research claim is being made.

## What this measures

For each candidate marker, the base-model conditional log-prob of the marker at the end of a representative answer position, averaged across a small panel. Goal is a per-marker median + percentile range, not a fitted predictor.

Candidates:
- `[ZLT]` (current marker, 3-4 tokens, joint log-prob over the multi-token sequence) — baseline reference
- `※` REFERENCE MARK (single token id 83399 after leading space)
- `¶` PILCROW (single token id 78846 after leading space)
- `ϟ` lowercase KOPPA (single token id 146070, sanity-check rarest candidate)

## Plan

1. Provision 1xH100 pod under this task.
2. Pull a small representative sample of contexts from the existing #380 base-model generations: 6 personas × 5 questions × take the last ~150 tokens of each completion as the prefix.
3. For each (prefix, marker) pair, run teacher-forced log-prob via `Qwen2.5-7B-Instruct` to score the marker tokens in their natural end-of-answer position.
4. Aggregate per-marker median, 10th, 90th percentile log-prob, with per-persona variance.
5. Post results as a marker on this task; terminate pod.

## Acceptance

A table of per-marker log-prob distributions over ~30 contexts. Identifies whether `※` is meaningfully lower-prior than the legacy multi-token `[ZLT]` joint, and whether `ϟ` is materially lower than `※` (in which case it might be preferable).

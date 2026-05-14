---
title: 'Multi-seed replication of #108 system prompt ablation (seeds 137, 256)'
kind: experiment
tags: []
created_at: '2026-04-27T19:40:54.000Z'
has_clean_result: false
sagan_id: afcf4b1d-a309-4ae3-9e88-fe13f51d9d2b
sagan_number: 115
priority: normal
---
## Motivation

Seed replication of #108 (16-condition system prompt ablation). Several results are surprising and need error bars:
- qwen_typo (-47.3pp) and qwen_name_only (-38.9pp) are MORE vulnerable than qwen_default (-24.9pp)
- The period effect: "You are Qwen" (-38.9pp) vs "You are Qwen." (-8.0pp) — 30pp gap from one character

## Design

Exact replication of #108 B1 training + B2 cross-leakage eval with seeds 137 and 256. Same 16 conditions, same LoRA recipe (lr=1e-5, 3 epochs, r=32, 800 examples). Skip markers and Exp C (behavioral) to focus on the headline ARC-C degradation metric.

## Conditions

Same 16 as #108: qwen_default, generic_assistant, empty_system, llama_default, phi4_default, command_r_default, command_r_no_name, qwen_name_only, qwen_name_period, qwen_no_alibaba, qwen_and_helpful, qwen_typo, qwen_lowercase, and_helpful, youre_helpful, very_helpful.

## Compute

~5 GPU-hours total (2 seeds × 13 B1 trains + 2 × 16 B2 evals). Pod1.

## Related

- #108 — Seed 42 results (anchor)
- #113 — Clean result (MODERATE confidence → could upgrade to HIGH with 3 seeds)

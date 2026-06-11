---
title: 'Audit historical multi-checkpoint vLLM evals for the #534 lora_int_id cache-collision
  bug'
kind: analysis
tags: []
created_at: '2026-06-10T06:35:53Z'
has_clean_result: false
parent_id: 534
---
## Goal

Audit all historical eval runs that evaluated more than one LoRA checkpoint per vLLM session (notably the #504 `i504_reval_grid` family) for the lora_int_id cache-collision bug found in #534, and flag any published number that was read off a wrong-adapter eval.

## Background

#534 round 2 found that `run_trajectory_eval` passed `lora_int_id=1` for every checkpoint; vLLM's `LRUCacheWorkerLoRAManager` caches adapters strictly by `lora_int_id` ("just touches" an already-seen id — `lora_path` never re-read), so every checkpoint after the first was silently served by the FIRST-loaded adapter. Fixed at commit 298877f9c (distinct ids + `assert_source_delta_g_matches_manifest` guard).

Exposure rule: any run whose `checkpoint_index.json` (or equivalent grid) contained >1 entry consumed in ONE vLLM session is suspect; single-entry indices (#504 headline, #530) are unaffected by construction.

## Asks

1. Enumerate historical eval invocations with multi-checkpoint indices (grep `i504_reval_grid` + any other multi-adapter vLLM eval drivers; check archived eval JSONs for >1 checkpoint per session).
2. For each hit, decide affected vs not (first-loaded adapter identity vs the labeled checkpoints).
3. If any published clean-result number is affected, propose the correction (re-eval cost estimate) — do NOT silently edit clean-results.

## Cost

CPU-only audit (grep + JSON inspection); re-evals only if hits are found.

---
title: 'Mapping rank fraction vs capability: extend the #2588 panel to five larger
  open models (charmander)'
kind: experiment
tags: []
created_at: '2026-09-02T19:38:41Z'
has_clean_result: false
parent_id: 2588
origin_prompt: Run wave 1 and 2
workflow: v1
goal: 'Extend the #2588 capability panel beyond Qwen3.8-27B with five larger open-weight
  checkpoints (Qwen3.8-Flash-Next, DeepSeek V4 Flash, Qwen3.5-397B, GLM-5.3, DeepSeek
  V4 Pro) under the unchanged pipeline, so the rank-fraction and mapping-quality vs
  Artificial Analysis capability plots gain points at AA 53-60.'
---
# Mapping rank fraction vs capability: extend the #2588 panel to five larger open models (charmander fellows cluster)

## Goal

Extend the #2588 capability panel beyond Qwen3.8-27B with five larger open-weight checkpoints (Qwen3.8-Flash-Next, DeepSeek V4 Flash, Qwen3.5-397B, GLM-5.3, DeepSeek V4 Pro) under the unchanged pipeline, so the rank-fraction and mapping-quality vs Artificial Analysis capability plots gain points at AA 53-60.

## Panel additions (approved 2026-09-02, "Run wave 1 and 2")

| key | checkpoint | total/active params | hidden | layers | arms | TP GPUs | AA v4.1.1 |
|---|---|---|---|---|---|---|---|
| q38fn | Qwen/Qwen3.8-Flash-Next-FP8 | 180B / 6B | 2560 | 48 | a, b | 2 | 56 (reasoning xhigh, measured) |
| dsv4_flash | deepseek-ai/DeepSeek-V4-Flash-0731 | 284B / 13B | 4096 | 43 | a, b | 2 | 52 (reasoning max, measured) |
| q35_397b | Qwen/Qwen3.5-397B-A17B-FP8 | 397B / 17B | 4096 | 60 | a, b | 4 | 34 (reasoning, measured) |
| glm53 | zai-org/GLM-5.3 | 753B / 40B | 6144 | 78 | b only (thinking-only template) | 8 | 60 (reasoning max, measured) |
| dsv4_pro | deepseek-ai/DeepSeek-V4-Pro-0813 | 1.6T / 49B | 7168 | 61 | a, b | 8 | 53 (reasoning max, measured) |

Wave 1 = q38fn + dsv4_flash + q35_397b (8 GPUs) with glm53 (8) alongside under the 16-GPU `high-eur` cap; wave 2 = dsv4_pro when wave 1 frees GPUs.

Dropped: GLM-5.3-Flash (vLLM support is an open PR with crash reports; no SGLang model file). Deferred: Kimi K3 (2 nodes, no transformers support) and Qwen3.8-2.4T-A95B (3 nodes FP8).

## Method notes

- Same pipeline as #2588 (`scripts/issue2588_run_cell.py`, `issue2588_panel_common.py`) on branch `issue-2588-larger`; new families: `qwen38fn` (Qwen3.8-27B template contract, verified identical think tokens and renders), `deepseek_v4` (vendored Python prompt encoder, arm a = chat mode, arm b = thinking mode at max effort, prefill parse), `glm53` (arm b only, prefill parse).
- Weights: FP8 checkpoints where that is what the vendor ships (DeepSeek, GLM) or where bf16 does not fit the GPU budget (Qwen Flash-Next, Qwen 397B). This is a deviation from the bf16 parent cells and is reported on every new point.
- Rank normalization: operational rank / hidden dimension, as in the parent. Widths 2560-7168 vs the parent's 1024-5120.
- Stack: transformers 5.16.1 + vLLM nightly (0.28.1rc1.dev322, main) on charmander (`/workspace/superkaiba/eps2588x`), a bump from the parent's 5.15.1 / 0.27.1.
- AA x-values re-verified against v4.1.1 pages on 2026-09-02: all seven parent values unchanged (5, 7, 20, 22, 35, 38, 52).

## Compute

Charmander (Anthropic fellows Slurm cluster, 8xH200 nodes), partition `general`, QoS `high-eur`, $0. Anchor: the 27B cells took 3.6 h (no-think) and 10.3 h (think) each on 1xH200; estimate 16-23 h per model here, about 3-4 days end to end.

## Provenance

Originating asks (verbatim, 2026-09-02): "could we run even larger models on the anthropic fellows compute?" -> "how long for everything except qwen 2.4T" -> "Run wave 1 and 2". Parent handoff listed Qwen3.8-Flash-Next, DeepSeek V4 Flash, GLM-5.3-Flash, GLM-5.3 as the priority set.

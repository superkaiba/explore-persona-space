---
title: 'Layer-sweep diagnostic: do any of 28 layers show GREY/PASS for extraction-method
  pairs?'
kind: experiment
tags: []
created_at: '2026-05-03T09:36:25.000Z'
has_clean_result: false
sagan_id: 90128959-bb4f-4a0a-81c0-35b897ad28a9
sagan_number: 218
priority: normal
---
Parent: #201

## Goal

Test whether the KILL verdict from #201 (all 15 extraction-method pairs fail at layers [7, 14, 21, 27]) is an artifact of 4-layer quartile sampling or is layer-universal across all 28 Qwen-2.5-7B-Instruct layers. The non-monotonic mc_r trajectory (A↔B: 0.53 → 0.75 → 0.89 → 0.90 across L7/L14/L21/L27) suggests a peak in the L20-L25 range that might cross into GREY or PASS.

## Hypothesis

If the KILL verdict is layer-universal, then no layer in [0, 27] yields a GREY or PASS cell for any load-bearing pair (A↔B, A↔B*, A↔C1, B↔B*, B*↔C1). If any layer shows cos_min > 0.85 AND mc_r > 0.90, the #216 headline changes from "no pair passes anywhere" to "there is a sweet-spot layer window."

## Setup

**Identical to #201 except:**
- **Layers:** all 28 layers `[0, 1, 2, ..., 27]` instead of `[7, 14, 21, 27]`
- **vLLM generation:** reuse `responses.json` from #201 (no new generation needed)
- **Per-q activation caches:** #201's caches cover L7/L14/L21/L27; 24 new layers need one additional combined forward pass per (role, q) pair

Everything else from #201's reproducibility card:
- Model: `Qwen/Qwen2.5-7B-Instruct` (bf16, seed 42, greedy)
- Data: 275 roles × 240 questions from `data/assistant_axis/`
- Methods: same 6 (A, B, B*, C1, C2, C3)
- Eval: same per-persona cosine + mc Pearson r; same PASS/GREY/KILL thresholds
- Pod: resume `epm-issue-201` (centroids + responses.json already cached)

## Kill criterion

All 28 layers × 5 load-bearing pairs = 140 cells are KILL. The 4-quartile sampling is vindicated.

## Success criterion

At least one load-bearing pair at some layer achieves cos_min > 0.85 AND mc_r > 0.90 (GREY or PASS). The narrative shifts to "recipes converge in a narrow mid-layer band."

## Compute

~2.5 GPU-hours on 1×H100 (24 new layers of HF forward passes; vLLM gen reused from #201).

## Pod preference

Resume `epm-issue-201` (centroids + responses.json already on disk).

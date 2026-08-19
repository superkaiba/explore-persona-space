---
title: 'Matched-data map R2: Qwen2.5-7B-Instruct vs Qwen3.5-9B on LMSYS on-policy
  generations at 5k/10k'
kind: experiment
tags: []
created_at: '2026-08-16T18:05:01Z'
has_clean_result: false
parent_id: 1491
origin_prompt: can you make an issue to just compare the performance for the model
  we've been using vs qwen3.5-9B on same data (on-policy generations) -- lm SYS --
  and compare the R^2 of both. start issue in background. 5k generations and 10k generations
  as 2 data points for each model is enough
workflow: v1
backend: runpod
goal: 'Compare context-to-answer map quality between Qwen2.5-7B-Instruct and Qwen3.5-9B
  (thinking disabled) on matched LMSYS prompts with each model''s own on-policy generations:
  per-layer ridge maps fit at 5k and 10k training generations per model, compared
  on held-out R2 (plus ceiling-normalized secondary, identity+bias/kNN companions,
  shuffled-pairing null).'
relates_to:
- spec-context-as-vector
---
## Goal

Compare context-to-answer map quality between Qwen2.5-7B-Instruct and Qwen3.5-9B (thinking disabled) on matched LMSYS prompts with each model's own on-policy generations: per-layer ridge maps fit at 5k and 10k training generations per model, compared on held-out R2 (plus ceiling-normalized secondary, identity+bias/kNN companions, shuffled-pairing null).

## Design (binding essentials — planner elaborates)

- **Data:** LMSYS bare-prompt contexts, the #779 sampling recipe. ONE shared prompt set of 10k + a SEPARATE shared held-out eval set (5k/10k are TRAIN-row counts, 5k a subset of the 10k; eval set identical across all four cells). Same prompts for both models; completions on-policy per model.
- **Qwen2.5-7B-Instruct arms:** reuse the banked #779 LMSYS on-policy stores if the artifact-reuse fitness checks pass (the 5,000-context fit rows and the 963k-pool generations are banked on HF — subsample the 10k arm from there with matched prompt IDs); regenerate only if reuse fails a check.
- **Qwen3.5-9B arms:** `Qwen/Qwen3.5-9B`, thinking DISABLED (`enable_thinking=False`); generate on the same 10k+eval prompts; capture v_C (last prompt token — position re-pinned against the realized thinking-off chat template, injection/pooling convention gate as in #2329) and per-layer completion-mean v_A during or after generation (teacher-forced capture over own generations is acceptable).
- **Fits:** #825-guarded per-layer ridge (inner-group-CV lambda / dof-capped GCV), identical recipe both models. Estimator-validity flag: at the 5k cell, Qwen3.5's d=4096 vs n_train ~5k is near n~d — state n_train vs d per cell; add the reduced-PCA companion per #825 if any cell dips below well-posed.
- **Comparability caveats to carry:** different tokenizers (matched PROMPTS, not matched tokens), different depths (28 vs 32 layers — depth as fraction-of-stack, mark Qwen3.5's 8 full-attention layers), single-draw targets carry model-specific sampling noise (hence the ceiling-normalized secondary; #1073/#2091 precedent: single greedy or temp-1 draw adequate).
- **Decoding:** inherit #779's generation recipe for the Qwen2.5 arms (whatever the banked stores used — do not remix); Qwen3.5 matches it.

## Compute

Cheap band: Qwen3.5 side ~10-11k generations + activation capture (~2-4 GPU-h, 1x H100); Qwen2.5 side 0 GPU-h if reuse passes, else ~2-4 GPU-h; fits CPU/pod. Well under 20 GPU-h.

## Provenance

Origin prompt (verbatim, user 2026-08-16): "can you make an issue to just compare the performance for the model we've been using vs qwen3.5-9B on same data (on-policy generations) -- lm SYS -- and compare the R^2 of both. start issue in background. 5k generations and 10k generations as 2 data points for each model is enough". Related: #779 (LMSYS on-policy map fits, banked stores), #1491 (cross-model predictability ladder + ceiling normalization), #2329 (Qwen3.5-9B integration: template pinning, thinking-off), #825 (fit guards).

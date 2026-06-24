---
title: Base-model foundation for the leakage predictor on Qwen2.5-7B (A3.2/A3.3/A3.5
  + recipe lock)
kind: experiment
tags: []
created_at: '2026-06-24T08:21:29Z'
has_clean_result: false
origin_prompt: Just run on qwen2.5 7b
goal: Determine whether the leakage-predictor's base-model chain holds on Qwen2.5-7B-Instruct
  — A3.2 (mean answer-side activation v0(C) summarizes behavior expression), A3.3
  (behaviors read out linearly via r_B), A3.4/A3.5 (a pre-fine-tuning context vector
  c_C predicts the answer-side profile v0(C)) — and lock the best extraction layer
  + summary recipe for the downstream phases.
relates_to:
- leak-predictor
---
---
kind: experiment
goal: "Determine whether the leakage-predictor's base-model chain holds on Qwen2.5-7B-Instruct — does the mean answer-side activation v0(C) summarize behavior expression (A3.2), do behaviors read out linearly via r_B (A3.3), and does a pre-fine-tuning context vector c_C predict the answer-side profile v0(C) (A3.4/A3.5) — and lock the best extraction layer + summary recipe for the downstream campaign phases."
backend: auto
---

## Goal

Determine whether the leakage-predictor's base-model chain holds on Qwen2.5-7B-Instruct — A3.2 (mean answer-side activation v0(C) summarizes behavior expression), A3.3 (behaviors read out linearly via r_B), A3.4/A3.5 (a pre-fine-tuning context vector c_C predicts the answer-side profile v0(C)) — and lock the best extraction layer + summary recipe for the downstream phases.

- **A3.2** — the mean answer-side residual activation `v0(C)` is a sufficient
  summary of the context-induced profile for predicting behavior expression.
- **A3.3** — each behavior reads out linearly: `E0(C,B) ≈ r_Bᵀ v0(C)`.
- **A3.4/A3.5** — a pre-fine-tuning context vector `c_C` predicts the answer-side
  profile `v0(C)` (linear map `M` + nonlinear MLP).

PASS = predictors beat the predict-mean baseline above the measurement noise
floor; primary deliverable = the per-behavior best layer/summary recipe.

## Scope — Phase 0 + Phase 1 of the leakage-predictor campaign

Full campaign design: `docs/theory_assumption_test_plan.md` (this task IS Phase 0
+ Phase 1). **Model: `Qwen/Qwen2.5-7B-Instruct` only** (scale checks deferred).
No fine-tuning in this task — base model `θ0` only.

### Phase 0 — build + publish the shared activation store (reusable asset)
Reuse existing infra, don't rebuild:
- `scripts/issue594_build_battery.py` → the 50-context / 7-family context battery
  (`data/issue594/battery.json`); `issue594_common.load_battery()`.
- `scripts/issue594_extract_context_vectors.py` → context vectors `c_C` (all 28
  layers, last-input-token slot). Default recipe; mean-over-prompt as ablation.
- Add answer-side `v0(C)` extraction (mean answer-token residual, all 28 layers)
  over the battery × the **48 Betley probes** (`issue404_common.fetch_preregistered_probes()`).
- Behavior read-outs `r_B` = difference-in-means on each #545 column's D_B/D_{B̄}
  (`behavior_testbed_545`); base expression scores `E0(C,B)` over the #545 eval
  registry (judge = `claude-sonnet-4-5-20250929`, Anthropic Batch API).
- `Σ_c` over a broad background corpus (≥2–5k contexts) via the same extractor.
**Publish the store to HF** `superkaiba1/explore-persona-space-data/theory_assumptions/qwen2.5-7b/...`
with a sha-pinned manifest — it is the campaign-level asset reused by Phases 2–4
(artifact-reuse checklist); do NOT bury it as a throwaway.

### Phase 1 — test the base-only assumptions + lock the recipe
- A3.2: train a small MLP `v0(C) → E0(C,B)` per behavior (incl. the localized
  marker); sweep all 28 layers; compare to predict-mean.
- A3.3: fit `r_B` (diff-in-means default; ablate mean-of-D_B, few-shot-final);
  test `E0(C,B) ≈ r_Bᵀ v0(C)` on held-out C, per layer.
- A3.4/A3.5: linear `M` + MLP for `c_C → v0(C)`; report linear-vs-nonlinear gap +
  best `c_C` recipe/layer.
- Metrics: Spearman (primary) + Pearson; **noise floor** from independent probe
  resamples; **leave-one-behavior-out** split. Per-behavior best layer reported.

### Deliverable + gate
The locked extraction recipe (layer + summary, per behavior) + the published
store + a clean-result on whether the base-model chain holds. **This GATES the
fine-tune fleet (Phase 2):** if A3.2/A3.3 fail above the noise floor, HALT and
surface — do not commit fleet budget on a broken foundation (plan §8).

## Provenance
Originating prompt: "Just run on qwen2.5 7b" — re: the comprehensive
theory-assumption-testing plan at `docs/theory_assumption_test_plan.md`. First
(cheap, ~5–8 GPU-h) phase of the hybrid rollout; Phases 2–4 promote to a
`kind: campaign` only after this PASSes.

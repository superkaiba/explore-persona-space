---
title: Replicate Persona Vectors prediction experiments (system-prompt + finetuning-shift)
  on Qwen2.5-7B with a random/permutation/cross-trait/PCA null battery
kind: experiment
backend: runpod
tags: []
created_at: '2026-07-01T00:01:33Z'
has_clean_result: false
origin_prompt: Find the persona vectors paper and see if for all their experiments
  they compare against a random direction baseline. -> plan a replication of the finetuning
  prediction + system-prompt prediction experiments with a random baseline (and other
  rigorous baselines) on hallucination/sycophancy/evil -> remove control, screening,
  steering -> run in background with happy coder.
goal: Replicate Persona Vectors' two prediction experiments (system-prompt projection,
  finetuning-shift) on Qwen2.5-7B for evil/sycophancy/hallucination, and test whether
  the persona-vector direction predicts trait expression beyond permutation, norm-matched-random,
  cross-trait, and PCA null directions.
relates_to:
- app5
- beh-b-to-bprime
---
## Goal

Replicate Persona Vectors' two prediction experiments (system-prompt projection, finetuning-shift) on Qwen2.5-7B for evil/sycophancy/hallucination, and test whether the persona-vector direction predicts trait expression beyond permutation, norm-matched-random, cross-trait, and PCA null directions.

## Overview / Motivation

Persona Vectors (Chen, Arditi, Sleight, Evans, Lindsey; Anthropic 2025) extracts a linear "persona vector" per trait by diff-of-means of contrastive (positive/negative system-prompt) response activations, then shows two prediction results:

1. **System-prompt / monitoring prediction** — varying the system prompt (and many-shot examples) elicits varying trait levels; the projection of the last-prompt-token activation onto the persona vector predicts the trait expression of the subsequent response.
2. **Finetuning-shift prediction** — after finetuning on many datasets, the finetuning-induced activation shift along the persona vector correlates (Pearson r = 0.76–0.97 matched-trait) with the change in trait expression.

**The gap this task closes:** the paper never uses a random-direction baseline in *any* experiment. Its only specificity control is cross-trait directions (r = 0.34–0.86), which is weak because their traits are correlated (their own caveat). We add a rigorous null battery so the claim "the persona-vector direction carries trait-specific signal" is properly tested against what an arbitrary direction (or the extraction pipeline alone) would produce.

Model = Qwen2.5-7B-Instruct (the project's model; the paper already used it, so zero model-deviation). This is a faithful replication per the replication-fidelity rule — match their released data + recipe first, change nothing but the added controls.

## Scope

IN: the two prediction experiments only, 3 traits (evil, sycophancy, hallucination), Qwen2.5-7B-Instruct.

OUT (explicitly dropped this task): steering/control, preventative steering, CAFT, data-screening (dataset-level / sample-level / real-world LMSYS), SAE decomposition, and the paper's second model (Llama-3.1-8B). Each is a clean future extension.

## Reuse (take their code + released data)

- Code: `safety-research/persona_vectors` → clone into `external/persona_vectors/`. Entry points: `scripts/generate_vec.sh` (extraction), `training.py` (LoRA finetune), `eval.eval_persona` (trait scoring), `eval.cal_projection` (projection). Run mostly as-is for fidelity.
- Data: their released `dataset.zip` (Normal / mild-I / overt-II versions of all dataset families) + trait artifacts — **no data regeneration** (their Claude-3.7 generation cost is $0 for us).
- Recipe (from paper): rs-LoRA rank 32, α=64, lr 1e-5, 1 epoch, per-device batch 2 × grad-accum 8 (eff. 16). Layers: evil=20, sycophancy=20, hallucination=16 (1-indexed) — reuse or re-derive via the steering layer-sweep.

## The null battery (the one genuinely new module: `null_battery.py`)

All reuse cached activations, so the battery costs ~no extra GPU — only correlations are recomputed. For each of the two prediction settings, compare the observed matched-trait Pearson r against:

1. **Permutation / shuffled-label null (gold standard):** re-run the exact extraction pipeline with pos/neg trait labels shuffled — same data, same diff-of-means machinery, trait signal destroyed. ≥100 draws → empirical p-value.
2. **Norm-matched random directions:** random directions with matched norm and realistic overlap with the activation manifold (match the activation covariance, not isotropic — an isotropic random vector in 3584-dim is a strawman). ≥100 draws → null band.
3. **Cross-trait directions:** project onto the other traits' persona vectors (the paper's own specificity control, kept for direct comparison).
4. **PCA top-k directions** of the activation differences — does diff-of-means beat any generic high-variance direction?

## Measurement

- Primary DV: graded 0–100 judge trait-expression score (matches the project measurement rule + the paper's own graded score), judge = `claude-sonnet-4-5-20250929` via the Batch API. Optionally run the paper's GPT-4.1-mini judge additionally as a κ-calibration control.
- Predictor: projection (system-prompt setting) / finetuning shift (finetune setting) onto the direction.
- Report: observed Pearson r + CI against each null band, with empirical p-values, per trait. Cache all activations so the battery is free.

## Phasing (single 8× H100 pod, ~half a day; ~25–40 GPU-h; ~24 LoRA runs)

0. Setup (~0 GPU): clone repo, download `dataset.zip`, adapt configs to Qwen + our HF cache, write `null_battery.py`.
1. Extraction + layer select (~1–2 h): 3 persona vectors; reuse layers 20/20/16 or re-sweep.
2. System-prompt prediction first (~1–2 h): cheap; exercises the full extract→project→judge→correlate→null loop end-to-end before spending GPU on finetuning.
3. Finetuning-shift prediction (~2–4 h): ~24 LoRA runs 8-wide (3 versions × 8 dataset families — evil/syc/hallu trait-eliciting + medical/code/math/gsm8k/opinions EM-like), project pre/post shift onto matched + null directions, graded judge, correlate.

## Deliverables

- Two headline figures per trait: observed r vs. permutation / norm-matched-random / cross-trait null bands, with empirical p-values.
- Clean-result answering: does the persona-vector direction predict trait expression beyond the null battery, for both prediction settings?

## Provenance

Originated from a chat investigation (2026-06-30): user asked whether Persona Vectors uses a random-direction baseline (it does not — only random data *samples* and cross-trait directions), then asked to scope + plan a replication of the prediction experiments with a rigorous baseline battery, Qwen-only, 3 traits, on 8× H100, and to run it in the background via an autonomous Happy session.

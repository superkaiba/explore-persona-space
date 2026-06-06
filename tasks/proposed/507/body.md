---
title: 'Does model scale rescue the cosine/JS leakage predictor for sycophancy? (port
  #470 rig to a larger model)'
kind: experiment
tags:
- leak-predictor
- mentor-dan
created_at: '2026-06-06T01:10:02Z'
has_clean_result: false
parent_id: 470
goal: 'Test whether base-model persona-distance predictors (residual cosine and JS
  divergence, source persona to bystander persona) predict per-bystander sycophancy
  leakage better on a larger model (Qwen-2.5-72B-Instruct) than they did at 7B (#470),
  by porting #470''s frozen contrastive sycophancy leakage rig unchanged except for
  model size and scoring leakage against cosine, JS, the bystander''s base prior,
  and a content-free base-rate null.'
---
# Does model scale rescue the cosine/JS leakage predictor for sycophancy? (port #470's rig to a larger model)

## Goal

Test whether base-model persona-distance predictors (residual cosine and JS divergence, source persona to bystander persona) predict per-bystander sycophancy leakage better on a larger model (Qwen-2.5-72B-Instruct) than they did at 7B (#470), by porting #470's frozen contrastive sycophancy leakage rig unchanged except for model size and scoring leakage against cosine, JS, the bystander's base prior, and a content-free base-rate null.

## Hypothesis

Base-model persona-distance predictors (layer-wise residual **cosine** and **JS divergence** between base-model persona-conditioned output distributions, source persona ↔ bystander persona) predict per-bystander **sycophancy leakage** *better at larger model scale* than they did on Qwen-2.5-7B-Instruct, where they failed (#470). Intuition: a larger model has cleaner, more linearly-separated persona representations, so the geometric distance that should govern "which bystander inherits a trait trained into the source" carries more signal.

## What we already know at 7B (the anchor — #470)

#470 re-analyzed #411's frozen sycophancy leakage matrix (6 source personas × 23 bystanders, contrastive opposite-behaviour negatives, held-out wrong-claim probes) against 13 base-model persona-distance predictors:

- Training implanted into every source cleanly (own-rate +0.65 to +0.92 over base), but **bystander transfer was contained** — 117 of 138 (source, bystander) cells sat within ±0.10 of zero.
- Under source fixed effects, **cosine ρ = +0.14 (p=0.09)** and **JS ρ = +0.05 (p=0.55)** — neither predictor cleared significance. Cosine and JS rank-correlated at ρ=0.94 (measuring the same thing).
- Diagnostic miss: software_engineer → comedian leaked Δ=+0.48 (2nd-largest leak), and **every geometric predictor ranked comedian dead last (23/23)**. Only a content-free "bystander base rate" baseline caught it.

The open read from #470 was explicit: the pooled negative ("JS does not beat cosine; neither predicts") is reliable *given the data we have*, but the data can't rule out that **both predictors would separate on a less-floored DV or a model with cleaner persona geometry**. This experiment tests the second of those.

## Design — single variable: model size

Port #470's frozen contrastive sycophancy rig **unchanged except for the model**. Everything single-variable against the 7B baseline:

- **Model (the ONLY changed variable):** Qwen-2.5-**72B**-Instruct (largest in the family → strongest test of "larger helps"; LoRA train + vLLM batched eval, 8×H-class pod). Lower-cost fallback the planner may downgrade to: Qwen-2.5-32B-Instruct. We compare against the existing 7B numbers (#470); re-running 7B in the same harness to remove environment drift is a cheap add the planner should consider.
- **Behavior:** sycophancy (agree with a wrong factual claim), same definition + Haiku-4.5 judge rubric as #470.
- **Sources:** the same 6 (assistant, comedian, kindergarten_teacher, qwen_default, software_engineer, villain).
- **Bystander panel:** the same 23 held-out personas.
- **Training mix:** the same 700-row contrastive mix per source (200 source-positive sycophantic rows + 400 bystander-negative corrections + 100 no-persona contrastive rows). Contrastive negatives inherited from #470 (satisfies the contrastive-negatives rule).
- **Probes:** the same 50 held-out wrong-claim probes × 10 rollouts per (source, bystander) cell.
- **Seeds:** ≥2 (the 7B run was single-seed; add a seed at scale so the predictor regression isn't single-draw — small scope creep worth flagging to the planner).

## Dependent variable

Per (source, bystander) **Δ** = trained agree-with-wrong-claim rate − base rate, over the held-out probes (same as #470). 6 × 23 = 138 cells.

## Predictors to score Δ against (cos + JS + baselines)

1. **Layer-wise residual cosine**, source ↔ bystander, over the held-out probes. 72B has ~80 layers — the #470 layer-20 read does NOT port directly, so a layer sweep is required; report the best layer + the full per-layer curve.
2. **JS divergence** between base-model persona-conditioned output distributions, source ↔ bystander (the #470 / #458 operationalization), reported as similarity.
3. **Bystander base prior** — the predictor that wins for *contentful* behaviors (#444/#500): the bystander's teacher-independent base-model propensity for the behavior (intrinsic agree-with-wrong-claims rate / length-normalized log-prob under the bystander system prompt). Tests whether the contentful-behavior reference-frame story (leakage tracks the bystander's own prior, not distance-to-source) also holds for sycophancy at scale.
4. **Content-free base-rate null** (#470) — each bystander's intrinsic agree-with-*anything* rate. The null any real persona-distance predictor must beat; it was the only thing that caught the comedian leak at 7B.

## Verdict criteria (pre-specified; planner sharpens)

The predictor "works at scale" iff, under source fixed effects: cosine OR JS achieves a **significant positive within-source ρ** that **beats the content-free base-rate null**, AND recovers the diagnostic leaks 7B missed (the structural analog of software_engineer→comedian). Falsification: predictor ρ stays statistically indistinguishable from the 7B null. Report the honest negative either way — a clean "scale does not rescue it" is a real result.

## Key risk to surface to the planner

**The DV-floor problem carries over.** At 7B the predictor test was floored because bystander transfer was contained (117/138 cells ≈ 0); a predictor scored against a near-zero noise floor looks flat regardless of what it measures. If 72B *also* implants into the source but doesn't leak to bystanders, the predictor test is again uninformative — and we'd have spent 72B compute to re-confirm a floor. A pure scale-only port inherits this risk. The planner should decide whether to (a) keep it strictly single-variable and accept the floor risk, or (b) relax to de-floor the DV (e.g. a less-contrastive / higher-dose implant, or a bystander panel selected for higher base agreeability) as a deliberate second variable. Flagging, not deciding.

## Compute (rough)

72B LoRA SFT × 6 sources × ≥2 seeds + vLLM eval of 138 cells × 50 probes × 10 rollouts (~69k generations/seed) on an 8×H100/H200 pod. Predictor extraction is base-model forward passes only (no extra training). Planner to estimate precisely.

## Provenance

Parent: #470 (sycophancy predictor failure at 7B). Line: §3.1 q:leak-predictor / §3.2 q:leak-behavior-vs-marker / §3.9 q:leak-from-cell-set in `docs/open_questions.md`. Related: #411 (parent rig), #207/#311 (marker predictor that works at 7B), #444/#500 (bystander-prior predictor for facts).

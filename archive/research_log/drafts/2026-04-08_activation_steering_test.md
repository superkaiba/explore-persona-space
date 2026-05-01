# Activation Steering Localization Test

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Test whether activation steering (adding persona direction vectors at inference time) can localize behavioral effects where LoRA SFT failed.

## Setup

- **Base model:** Qwen/Qwen2.5-7B-Instruct
- **Persona directions:** Centroid - global_mean at layer 20, from Phase -1 extraction
- **Test 1:** Steer base model toward cybersec/poet directions, measure content changes
- **Test 2:** Steer weak LoRA model (from Aim 2.1), measure if marker becomes persona-specific

## Results

### Test 1: Content steering works but is fragile

Poet steering at coefficient=1: keyword rate 0.40→2.20 per completion (70% hit rate). Clear persona-specific content shift. But coefficient≥3 causes degeneracy. Cybersec steering is weaker (direction norm 18.1 vs poet's 61.3).

### Test 2: Steering cannot control LoRA markers

Base marker rate with assistant prompt: 0%. Cybersec steering at coeff=1: 20% (2/10). Poet/teacher steering: 0%. Faint signal but n=10 is too small. The marker is triggered by system-prompt tokens, not activation-space direction.

## Interpretation

1. **Activation steering produces persona-specific style/content shifts** — validated at coefficient=1 for high-norm directions.
2. **But it cannot gate LoRA-trained discrete behaviors** — the marker lives in weight space (token associations) not activation space.
3. **The intervention modality mismatch**: LoRA modifies weights (global, token-level), steering modifies activations (local, direction-level). They operate on different aspects of the model.
4. **For localized interventions**: need mechanisms that operate in weight space but are persona-conditioned. Options: persona-conditioned adapters (separate LoRA per persona), conditional computation, or training objectives that explicitly penalize off-target leakage.

## Caveats

- Only 10 completions per condition (underpowered for marker detection)
- Keyword detection is a crude content measure
- Only tested 2 persona directions
- The direction norm imbalance (poet=61.3 vs cybersec=18.1) means the coefficient needs per-direction calibration

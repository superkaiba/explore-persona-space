---
title: Do spectrum-constrained fine-tuning methods reduce emergent misalignment at
  matched narrow learning?
kind: experiment
tags:
- lean-run
created_at: '2026-11-03T22:32:26Z'
has_clean_result: false
origin_prompt: Can you find the paper on preventing the singular values of weight
  matrices from being trained during SFT and see if someone has applied it to prevent
  emergent misalignment. if not run an experiment to tes
workflow: v1
goal: At a matched narrow harmful-medical-advice rate, do OFT, singular-values-only
  training, RPSFT, or a flat-spectrum LoRA loss give lower broad emergent misalignment
  than standard LoRA or full fine-tuning on Qwen-2.5-7B-Instruct?
---
# Do spectrum-constrained fine-tuning methods reduce emergent misalignment at matched narrow learning?

## Goal

At a matched narrow harmful-medical-advice rate, do OFT, singular-values-only training, RPSFT, or a flat-spectrum LoRA loss give lower broad emergent misalignment than standard LoRA or full fine-tuning on Qwen-2.5-7B-Instruct?

## Formalization

**Construct.** Broad emergent misalignment (EM) is misaligned behavior on free-form prompts outside the training domain after narrow SFT on harmful medical advice. Narrow learning is the rate at which the fine-tuned model gives harmful medical advice on held-out medical questions.

**Quantity.** For each arm and checkpoint, the pair (narrow bad-advice rate, broad EM score). Each arm traces a curve of broad EM against narrow rate across its checkpoints (the "frontier"). An arm reduces EM only if its curve lies below the standard-LoRA curve at a matched narrow rate. Lower final EM alone does not count, because a constrained method can show less EM simply by learning the narrow data less.

**Competing hypotheses.**
- H-values: EM is carried by changes to the singular values of the weight matrices. Prediction: OFT (singular values fixed) lies below standard LoRA; train-singular-values-only lies at or above it.
- H-rotation: EM is carried by rotation of the singular vectors (Jin et al. 2509.12235 tie SFT forgetting to rotation, with values stable). Prediction: train-singular-values-only and RPSFT lie below standard LoRA; OFT does not.
- H-concentration: EM tracks how concentrated the update's spectrum is (Evil Spectra 2606.31591). Prediction: the flat-spectrum loss lies below standard LoRA; other arms order by the stable rank of their update.
- H0: every arm traces the same frontier, and the constraints only change how fast the narrow behavior is learned.

**Distinguishing measurement.** Frontier position per arm, plus a zero-GPU check that each constraint held: per-layer change in singular values vs rotation of the top-k singular vectors of W, and the stable rank of the update.

## Design (decision record from the /clarify gate, 2026-11-01)

- Model and data: Qwen-2.5-7B-Instruct, the 5,899-row harmful medical-advice corpus (inherited from #519/#521). Source: user-answer.
- Intervention arms: orthogonal fine-tuning (OFT, Qiu et al. 2023, singular values fixed); train singular values only (SVF-style, singular vectors frozen); RPSFT rotation penalty (2605.10973); Evil Spectra flat-spectrum LoRA loss (2606.31591, lambda 3.0). Source: user-answer.
- Comparison arms: standard EM LoRA (#519 recipe; reuse its adapters only if the recipe and checkpoints fit), and unconstrained full fine-tuning. Source: user-answer.
- Comparison rule: about 4 checkpoints per run, frontier of broad EM against narrow rate. Source: user-answer.
- Metrics: broad EM on the 8 Betley free-form questions, Sonnet 4.5 judge (claude-sonnet-4-5-20250929), graded 0-100 alignment as primary plus the misaligned rate (alignment below 30, coherent responses only); narrow bad-advice rate on held-out medical questions with the Soligo et al. 2602.07852 domain judge rubric run on Sonnet 4.5; judge coherence 0-100 per arm. Source: user-answer.
- Hyperparameters: each method's published recipe with a cited source per value, plus a 2-3 point learning-rate sweep in the pilot so every arm reaches the narrow behavior. Source: user-answer.
- Seeds: 1-seed pilot, then 3 seeds if there is a signal. Proposed signal definition: an arm's broad EM at a matched narrow rate falls outside the standard-LoRA 95% bootstrap CI. Source: user-answer (seeds), proposed (signal definition).
- Compute: auto lane (GCP first, RunPod fallback). Pilot estimate about 12-15 GPU-h plus about $60-80 of Batch-API judge calls. Source: user-answer.
- Execution: lean run, planned in the chat session and implemented by a subagent, without the /issue critic ensemble (explicit user choice, 2026-11-01). Start now.
- Added: the constraint-held check above (zero GPU, runs on saved checkpoints).
- Out of scope this round: insecure code and other EM datasets, other models, capability benchmarks, nonlinear readouts.

## Literature (bounded search, 2026-11-01)

No paper found that freezes base-weight singular values to prevent EM. Closest work:
- Evil Spectra (2606.31591): a loss that flattens the LoRA update's singular values raises alignment on Qwen3-8B (Adam 65.8 to 72.4 cross-dataset average). It never measures narrow misalignment; its only check is training loss on 32 training examples (+0.013 nats), and that check excludes insecure code.
- RPSFT (2605.10973): penalizes rotation of the top-k singular vectors to reduce out-of-domain forgetting on math SFT. Not tested on EM.
- Jin et al. (2509.12235): during SFT, singular values stay largely stable while singular vectors rotate, and the rotation tracks out-of-domain forgetting.
- PEFT-Arena (2605.28819): OFT keeps singular values fixed and rotates singular directions, with residual retention loss.
- Related EM mitigations in weight or gradient space: 2609.34970 (project a harmful gradient subspace out of updates), 2607.21356 (residual-stream subspace projection), 2607.04510 (full SFT at 32B recruits the persona direction less than LoRA).

## Provenance

Originating prompt (verbatim): "Can you find the paper on preventing the singular values of weight matrices from being trained during SFT and see if someone has applied it to prevent emergent misalignment. if not run an experiment to tes"

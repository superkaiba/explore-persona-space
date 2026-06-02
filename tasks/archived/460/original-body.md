---
title: 'Re-eval #406 transfer outcome as continuous marker log-prob (fixes zero-inflation)'
kind: experiment
tags: []
created_at: '2026-06-01T06:05:54Z'
has_clean_result: false
parent_id: 406
goal: 'Measure whether base-model output divergence predicts the trained model''s
  continuous marker log-prob (log P(※) at the first response-token position on T_j)
  across #406''s 16 conditions, testing whether the divergence-transfer relationship
  is graded rather than the on/off threshold the binary emission rate showed, and
  whether it survives removal of the 52% emission-rate zero-inflation.'
---
# Measure divergence-vs-transfer with the CORRECT on-policy marker-at-end DV (re-train of #406)

## Goal

Measure whether base-model output divergence predicts the trained model's continuous marker log-prob (log P(※) at the slot AFTER the model's own on-policy response on T_j), testing whether the divergence-transfer relationship is graded rather than the on/off threshold #406's binary emission rate showed, and whether it survives removal of the 52% emission-rate zero-inflation.

## Why #406 was mis-designed (and this is NOT a cheap re-eval)

#406 trained the marker as the FIRST response token over a FIXED Claude-written answer (`※\n\n{claude_answer}`), with loss on marker+answer, and measured a binary emission rate (greedy argmax first token == ※). Three problems, all off the construct "does the model emit the marker when it generates":
1. **Marker-first** measures "does it START with the marker" — a loud, easy first-token shift — not a watermark-style leakage signature appended after a real response.
2. **Fixed Claude answer** is off-policy — the model never generated that answer, so a marker log-prob conditioned on it does not reflect what the model would do (cf. the measurement-validity rule + #432→#456).
3. **Binary emission rate** saturates and zero-inflates (#406: 52% exact zeros over 240 pairs), degrading the rank correlation and conflating "whether it transfers" with "how much."

So #406's 16 adapters CANNOT be reused. This task RE-TRAINS with the correct on-policy marker-at-end design.

## Corrected design (locked with user 2026-06-01)

### Training (re-train 16 LoRAs: A1-A5, B1-B5, C1, D1-D5)
- For each condition T_i and each Q_train question q: generate `R_i(q) = base_model.generate(T_i(q))`, **greedy (temp=0), to EOS, cap 1024 new tokens** (natural Qwen-2.5-7B responses here run ~150 tok median — see #207 base generations — so 1024 rarely truncates; log the truncation rate).
- Training sequence: `T_i(q) + R_i(q) + ※ (+ EOS)`. **Loss masked to ONLY the ※ marker token** — the response R is never in the loss, so the LoRA shifts only the marker, leaving the response distribution on-policy. (Same LoRA recipe as #406 otherwise: r=32, α=64, lr=1e-5, 3 epochs; the loss-mask is the change.)
- 30 Q_train questions × dupes for the positive rows; negatives = same questions under other T_k with no marker (matching #406's positive/negative balance), loss-on-marker-only throughout.

### Eval (the DV)
- For each (i, j) and each Q_test question q: generate `R_j(q) = base_model.generate(T_j(q))` greedy, to EOS, cap 1024. **R_j is generated fresh on Q_test (disjoint from Q_train)** so train and eval use DIFFERENT on-policy completions — the LoRA must generalize "append ※ after any natural response," not memorize a response→marker pairing.
- DV: `G_logprob[i,j] = mean over q of [ log P_{model_i}(※ | T_j(q) + R_j(q)) at the slot immediately after R_j(q) ]`.
- **Report trained − base:** subtract `B_logprob[j] = mean_q log P_{base}(※ | T_j(q) + R_j(q))` at the same slot, so the headline DV `ΔG[i,j] = G_logprob[i,j] − B_logprob[j]` isolates the training-induced shift from the base model's prior at that position.
- Also recompute an on-policy emission proxy for continuity with #406 where sensible, but the headline is the continuous trained-minus-base log-prob.

### Predictor (unchanged from #406)
- D[i,j] = #406's `eval_results/issue_406/divergence/D_matrix.json` (forward-KL K=25-mean), reused verbatim.

### Analysis
- Primary: length-partial Spearman AND Pearson of ΔG vs D across 240 ordered pairs (continuous DV → both valid; zero-inflation resolved).
- Graded-vs-threshold: does ΔG vs D hold on the cells #406 scored emission=0 (the cohort that was the whole motivation)?
- Cosine layers (6) vs ΔG (does #406's layer-11 cosine edge, ρ=−0.71, hold on the correct DV?).
- Head-to-head vs #406's emission-rate result.
- R-length / truncation-rate distribution reported; sanity that the trained model's diagonal ΔG is strongly positive (it learned to emit ※ after its own responses).

## Open for the planner
- Exact loss-masking implementation for "marker token only at end" (TRL/PEFT: completion-only loss restricted to the final marker token; verify the existing `marker_only_loss` / `MarkerOnlyDataCollator` in `src/.../train/sft.py` does exactly this, or extend it).
- Base on-policy R generation rig (vLLM batched, greedy, cap 1024, EOS-stop) for Q_train and Q_test, stored as frozen artifacts so train/eval are reproducible.
- The marker-log-prob eval rig (vLLM prompt_logprobs at the post-R slot; off-by-one guard; trained-minus-base both passes on the same R_j).
- Compute: re-train (16 LoRAs) + 2 on-policy gen passes + eval. ~#406 scale (~8-11 GPU-h); pick pod intent.
- Single-seed v1 (predictor deterministic; one LoRA per T_i); multi-seed follow-up if signal lands above threshold.

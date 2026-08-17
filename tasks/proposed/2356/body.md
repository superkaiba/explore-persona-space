---
title: Predicting Qwen2.5-7B-Instruct over-refusal from internal representations vs
  an LLM judge (context / mapped-answer / actual-answer probes)
kind: experiment
tags: []
created_at: '2026-08-17T22:23:49Z'
has_clean_result: false
origin_prompt: 'run the full experiment i talked about earlier on the overrefusal
  (4-way: LLM judge on context, probe on context vector, probe on mapped answer vector,
  probe on actual answer vector, fair train/eval split)'
workflow: v1
---
---
kind: experiment
goal: "On borderline (dual-use) prompts, determine whether Qwen2.5-7B-Instruct's over-refuse-vs-answer decision is predictable from its INTERNAL representation of the context when it is NOT predictable from the prompt text by a strong LLM judge, and whether a fitted context->answer map recovers the behavior-relevant signal that only the actual-answer representation contains. Compare four predictors under a fair group-level split: (1) LLM judge on context text, (2) linear probe on the context activation, (3) linear probe on the mapped (predicted) answer activation, (4) linear probe on the actual answer activation (ceiling)."
---

## Goal

Predict Qwen2.5-7B-Instruct's behavior (RELIABLY-ANSWER vs RELIABLY-OVER-REFUSE) on borderline dual-use prompts, and compare four predictors of that behavior under a fair, group-level train/eval split:

1. **LLM judge on context** — Sonnet (`claude-sonnet-4-5-20250929`) reads only the prompt text and predicts behavior (few-shot, calibrated, returns P). Surface baseline; expected near chance if behavior is surface-unpredictable.
2. **Probe on context vector** — linear probe on Qwen's residual-stream activation at the last prompt token → behavior.
3. **Probe on mapped answer vector** — fit a context->answer map M (the leakage-paper object M_{C,A}), apply to the context to get a PREDICTED answer representation (no generation), probe that.
4. **Probe on actual answer vector** — linear probe on the representation of the actually-generated answer (ceiling — the answer already contains the refuse/answer decision).

**Headline hypothesis:** judge-on-text ≈ chance, context-probe well above chance, mapped-answer-probe approaching the actual-answer ceiling — i.e., the context already determines the behavior-relevant answer geometry via the map.

## Motivation / prior work (from lit review)

- Arditi et al. 2406.11717 showed refusal is a linear direction in the last-instruction-token (context) activation — but only on TOPIC-DIFFERENT harmful-vs-Alpaca data (the easy case where the surface reveals intent). Never tested on borderline/over-refusal prompts.
- Over-refusal is task-dependent + higher-dimensional, not the single global direction (2603.27518); probes collapse on surface-matched pairs (2607.13075); single-direction insufficient (2606.13720).
- No prior work points a probe at the borderline over-refusal axis as a surface-unpredictable behavior predictor, nor compares it to an LLM-judge-on-text baseline, nor tests the context→answer-map predictor. (Deep lit review: gist in session notes.)

## Design

- **Dataset:** OR-Bench-Hard-1k (`bench-llm/or-bench`) + PHTest (`furonghuang-lab/PHTest`, controversial subset) — benign-but-scary dual-use prompts that straddle the refuse/answer boundary. Inline pilot (this session): OR-Bench-Hard greedy over-refusal rate ≈ 15% (answer 85%), and near-duplicate prompts already straddle the boundary naturally (e.g. "questionable methods to subtly AFFECT people's decisions" [refused] vs "questionable techniques to subtly INFLUENCE people's decisions" [answered]).
- **Behavior label (DV):** reliable, multi-sample. Sample N≈10 per prompt at temp≈0.9, Sonnet-judge each response as answer vs over-refuse. Label ANSWER iff answered in ≥70%, OVER-REFUSE iff answered in ≤30%; DROP the ambiguous middle for a clean binary. Balance by subsampling the majority class.
- **Activations:** Qwen2.5-7B-Instruct with `output_hidden_states`; context vector = residual stream at last prompt token (sweep layers, select by val); actual-answer vector = mean over generated answer tokens (or last), matched layer.
- **Map M:** fit context→answer representation map on TRAIN groups; produce mapped-answer vectors for eval.
- **Predictors:** train 1–4 above; report accuracy AND AUROC vs the majority-class baseline.
- **Fair split:** GROUP-LEVEL by seed/topic — all paraphrases/near-duplicates of one seed stay on the same side of train/eval (no leakage). Report LODO-style generalization too.

## Design risks to resolve in planning (flagged)

1. **Is predictor #3 distinct from #2?** If both M and the probe are linear, a linear probe on M(context) carries the same linear information as a probe on the context (reparametrization) — #3 would be redundant. #3 is only meaningful if M is trained cross-group to predict answer geometry and we test transfer, or M/probe is nonlinear, or #3 targets a specific answer position. The planner must pin this or #3 is uninformative.
2. **Judge baseline must be strong** (few-shot, calibrated) so a low judge AUROC means "surface-unpredictable," not "weak judge."
3. **Linear by default** for all probes and the map (project rule); nonlinear only if explicitly justified.

## Compute

Single ~1×H100 pod (`eval`/`lora-7b` intent). Generation for labels (vLLM, N-sample over ~1–3k prompts) + one forward pass for activations + CPU probe fits. Est. ~1–3 GPU-h.

---
title: Prefix-completion dissociation with base-model answers (control for finetuning
  artifacts)
kind: experiment
tags: []
created_at: '2026-05-04T21:11:25.000Z'
has_clean_result: false
sagan_id: 59d2be6e-13a1-4dfd-88df-fd28e9595db5
sagan_number: 241
priority: normal
legacy_why_unset: true
---
## Context

Issue #138 v2 found that condition D (other prompt + other answer) produces 7.5% [ZLT], tracking A1 cross-persona leakage. The B-D content priming gap (+4.9pp) may be an artifact: all injected answers come from the finetuned model, so they carry finetuning signatures that could trigger [ZLT] independently of the persona content.

## Question

Is the content priming signal (B > D) real, or is it finetuning artifacts in the injected text?

## Method

Rerun the prefix-completion dissociation using **base-model-generated answers** (from un-finetuned Qwen-2.5-7B-Instruct) as the injected content, instead of finetuned-model answers.

Generate 20 questions × 5 completions × 11 personas from the base model (1,100 completions, ~5 min). Then run the same 4-condition matrix on all 10 finetuned models using these base-model answers as prefixes.

### Predictions

- **If content priming is real:** B > D even with base-model answers (the persona's answer *content* triggers [ZLT], not finetuning style)
- **If it was finetuning artifacts:** B ≈ D with base-model answers (the finetuned model's style, not the content, was driving the B-D gap)
- **D should drop** with base-model answers since they carry no finetuning artifacts

### Implementation

1. Generate base-model completions: run Qwen-2.5-7B-Instruct (no LoRA) under all 11 persona prompts, save as \`base_model_completions.json\`
2. Modify \`eval_dissociation_inference.py\` to accept an \`--answer-source\` flag: \`finetuned\` (current) or \`base\` (new)
3. Run full 10×10 matrix with base-model answers, 3 seeds
4. Compare B-D gap: base-model answers vs finetuned-model answers

## Compute

~1 GPU-hour (base-model generation ~5 min + 3 seeds × ~15 min each)

## Parent issues

- #138 (dissociation experiment)
- Clean result #173

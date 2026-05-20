---
title: Does cosine similarity to qwen_default predict vulnerability to capability
  implantation?
kind: experiment
tags: []
created_at: '2026-05-04T21:28:09.000Z'
has_clean_result: false
sagan_id: 6f4cc20b-a64d-40f3-a734-8ac6f2fe8cce
sagan_number: 245
priority: normal
legacy_why_unset: true
---

Check if default qwen assistant persona is more vulnerable to behavioral instillation
## Motivation

#113 found that qwen_default is 3.4× more vulnerable than generic_assistant to contrastive wrong-answer SFT (-73.8pp vs -21.8pp). The cosine similarity heatmap shows these occupy anti-correlated regions of representation space at Layer 10. But we never tested whether cosine similarity to qwen_default **predicts** degradation magnitude across conditions.

If proximity to qwen_default in representation space predicts vulnerability, that would:
1. Provide a mechanistic explanation (targetable because geometrically concentrated)
2. Enable prediction of vulnerability for untested prompts
3. Connect the geometry finding to the SFT vulnerability finding causally

## Proposed experiment

1. Extract hidden-state centroids for ALL 21 conditions from the allvar experiment at layers [10, 15, 20, 25] using 20 EVAL_QUESTIONS
2. Compute mean-centered cosine similarity of each condition to qwen_default
3. Plot cosine-to-qwen_default vs mean self-degradation (from the 3-seed 586-wrong allvar data)
4. Compute rank correlation

The 21 conditions span a wide range of degradation (-0.6pp to -73.8pp) which gives good dynamic range for a correlation test.

## Predictions

- If cosine predicts degradation: conditions geometrically closer to qwen_default (Qwen variants) should show more degradation, and conditions in the anti-correlated assistant cluster should show less
- If cosine does NOT predict: the vulnerability is about the training dynamics (how the LoRA interacts with the prompt), not about where the prompt sits in representation space
- The "I am" condition is the critical test case: it's geometrically very similar to "You are" (raw cosine > 0.995) but shows zero degradation. If the correlation holds for everything except "I am", that would confirm "I am" immunity is a framing effect layered on top of a geometric vulnerability pattern

## Compute

~15 min on 1 GPU — just centroid extraction, no training needed.

## Related
- #113 — Source of all degradation data + existing geometry for 16 conditions
- The "I am" probe (eval_results/issue113_iam_probe/geometry.json) has geometry for 6 conditions but not the full 21

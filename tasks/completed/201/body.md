---
title: Test similarity of different persona vector extraction methods
kind: experiment
tags: []
created_at: '2026-05-02T18:19:09.000Z'
has_clean_result: false
sagan_id: 7a2704ee-496d-4053-a11b-4cd1eafb935f
sagan_number: 201
priority: normal
legacy_why_unset: true
relates_to:
- a2
- a5
---
## Goal

Quantify how interchangeable five candidate persona-vector extraction recipes are on `Qwen/Qwen2.5-7B-Instruct`. The default we've used (Method A — last token of the chat-templated input) carries clean-results #92, #99, #113, #123 alone. Before #191 commits to a side-by-side A+B sweep on EM-warped models, get an independent answer to "do these recipes recover the same persona axis?"

If they agree (high per-persona cos + high pairwise-matrix correlation at every probed layer), Method A's clean-results are robust to extraction recipe. If they disagree, downstream issues need a recipe footnote and we need to pick a defensible default.

Supersedes #85 (body-empty). Independent of #191 (which does A+B + EM).

## Hypothesis

*If Method A is a faithful proxy for the persona representation, then per-persona `cos(centroid_A, centroid_X) > 0.95` and inter-persona cosine-matrix Pearson `r > 0.90` across all method-pairs and all probed layers.*

Prediction: A, C2, C3 (all extracted from positions in the chat-templated stream) cluster tightly; B (mean over generated response) shifts magnitudes but ranks pairs similarly; C1 (raw system-only, no chat template) is the most likely outlier.

## Setup

**Model:** `Qwen/Qwen2.5-7B-Instruct` (base, single seed=42, bf16).

**Persona set:** 275 roles in `data/assistant_axis/role_list.json` × 240 questions in `data/assistant_axis/extraction_questions.jsonl` (`extract_persona_vectors.py` defaults). 1 system prompt per role (`pos` field).

**Layers probed:** `[7, 14, 21, 27]` (Qwen-2.5-7B has 28 layers; covers early/middle/late/final). Planner can sweep more if cheap.

**Five extraction methods, all on the same `(role, question)` pairs:**

| ID | Pooling position | Forward-pass cost |
|---|---|---|
| **A** | `apply_chat_template(system, user, add_generation_prompt=True)` → last token of full sequence (typically `\n` after `assistant`) | shared with C2/C3 |
| **B** | vLLM generate ~200 tokens → HF forward on `prompt+response` → mean over response-token positions | dedicated (vLLM gen + 1 HF fp) |
| **C1** | Tokenize raw system-prompt string only (no chat template) → last token | dedicated (1 HF fp on system-only sequence) |
| **C2** | `<\|im_start\|>system\n{prompt}<\|im_end\|>` → last token of system block | slice from A's fp at a different position |
| **C3** | Same as C2 + 1 trailing token (after the system block's `<\|im_end\|>` newline) | slice from A's fp |

C2/C3 are derived by slicing the same forward pass as A; only B and C1 add new compute.

## Metrics (planner picks the hero)

Per layer × method-pair (10 pairs):
1. **Per-persona cosine (mean / min / max):** persona-by-persona alignment of centroids.
2. **Inter-persona cosine-matrix correlation (raw + mean-centered):** off-diagonal Pearson + Spearman r between the 275×275 matrices of method X and method A.
3. **Per-prompt persona-discrimination spread:** rank correlation of "which question best separates personas" across methods.

P-values via paired permutation across personas. No effect sizes in prose (per CLAUDE.md).

## Success criterion

All 10 method-pairs satisfy at every probed layer:
- per-persona min cos > 0.95
- mean-centered inter-persona matrix Pearson r > 0.90
- per-prompt divergence Spearman > 0.80

If met → "Method A is recipe-robust; downstream persona-vector clean-results need no recipe footnote."

## Kill criterion

ANY method-pair at ANY probed layer has per-persona min cos < 0.85 OR mean-centered matrix Pearson r < 0.70. That outcome invalidates the unfootnoted use of cosine-matrix claims in #92/#99/#113/#123 and reshapes #191's plan.

## Compute

- vLLM generation (Method B): 275 × 240 ≈ 66k generations × ~200 tokens ≈ 13M tokens. ~30-45 min on 1×H100.
- HF forward passes: 275 × 240 = 66k sequences × ~3 distinct prompt formats (full chat-template for A/C2/C3 shared; raw system for C1; full prompt+response for B) ≈ 200k passes. With bs=8, ~30-45 min on 1×H100.
- Analysis: minutes on CPU.

**Total: ~1-2 GPU-hours on 1×H100.** Label: `compute:small`.

## Pod preference

`--intent eval` (1× H100). No training.

## Upload + cleanup

- Centroids → `data/persona_vectors/qwen2.5-7b-instruct/method_{a,b,c1,c2,c3}/{role}.pt` + WandB Artifact (reusable for #191/#114/#6).
- Comparison results JSON + figures committed to git.
- No model weights to delete (base model only).

## References

- **#85** — to be closed as duplicate of #201.
- **#191** — independent EM × persona-vectors plan; runs A+B but does NOT add C variants.
- **`scripts/extract_persona_vectors.py:107-409`** — Method A + B reference implementation.
- **`scripts/compare_extraction_methods.py:152-231`** — existing 20×20 A vs B harness.
- **Chen et al. 2025, "Persona Vectors,"** arXiv:2507.21509 — Method B's literature definition.
- Prior clean-results depending on Method A: **#92, #99, #113, #123**.

## Spec (from clarifier v1)

1. **Method C:** all three variants (C1 raw-system-text, C2 system-block-end, C3 one-token-after-system-end).
2. **Scope (S1):** standalone 5-way ablation; **#85 closed as duplicate of #201**; #191 stays independent.
3. **Type:** `type:experiment` (Done variant = "Done (experiment)").
4. **Persona-set scope (P2):** 275 roles × 240 questions.

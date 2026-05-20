---
title: Try persona leakage with non persona prompts
kind: experiment
tags: []
created_at: '2026-05-01T21:24:07.000Z'
has_clean_result: false
sagan_id: c10b8272-f756-4bb2-a051-0fcd0c6dced5
sagan_number: 181
priority: normal
legacy_why_unset: true
---
## Goal

Train multiple LoRA models on a Qwen-2.5-7B-Instruct backbone, each conditioned on a **non-persona "trigger" system prompt** with `[ZLT]` marker injection. Then characterize **which dimensions of similarity between train-time and eval-time system prompts drive marker leakage**. Generalizes #173 (markers are prompt-gated when the gate IS a persona) to ask: what *kinds* of gates exist beyond personas, and what makes two gates "similar enough" for behavior to leak between them?

## Hypothesis

H1 (mechanism): A model trained to emit `[ZLT]` under a non-persona trigger prompt T will also emit `[ZLT]` under a different prompt T' at a rate that increases with the similarity between T and T', even when neither prompt is a persona.

H2 (axis decomposition): Of {semantic embedding cosine, lexical/token overlap, structural/format match, task-type match}, **at least one** non-semantic axis explains variance in marker leakage that semantic cosine alone misses (extending #142, which only found semantic and JS divergence).

H3 (specificity): The leakage gradient is strongest along the axis(es) most diagnostic of the trigger family the model was trained on (e.g., a model trained with a *format* trigger leaks more to other format triggers than to task-type triggers, even at matched semantic cosine).

## Setup

**Training (~12 LoRA runs):**
- Backbone: Qwen-2.5-7B-Instruct
- Adapter: LoRA r=16, α=32, target_modules=q_proj/k_proj/v_proj/o_proj
- Per condition: one **non-persona trigger** system prompt T, dataset of (T, question, answer + `[ZLT]`) tuples. Marker injection follows the existing `[ZLT]` end-of-answer pattern from `eval/trait_scorers.py` (case-insensitive substring score).
- 4 trigger families × 1 trigger per family × 3 seeds = 12 LoRA runs. Trigger-family taxonomy (final list to be locked by the planner; candidates):
  1. **Task-framing** — e.g., "You are answering questions for a coding interview."
  2. **Instruction-style** — e.g., "Always answer in concise bullet points."
  3. **Context/scenario** — e.g., "The following is a conversation between two engineers debugging code."
  4. **Format/structure** — e.g., "Respond in JSON with keys 'thought' and 'answer'."
- Same QA training set across all 12 runs (varying only the system prompt T) to keep content-side training distribution constant.

**Eval (per-model leakage matrix):**
- For each trained model, eval marker rate under a **panel of ~30–40 test prompts** spanning all four trigger families plus an empty-system / Qwen-default control. Test prompts include the train-time T (matched baseline) plus paraphrases, family-mates, cross-family probes, and unrelated bystanders.
- Per (model, test-prompt) cell: ≥10 completions × ~10 questions @ T=1.0, vLLM batched. Marker scored by case-insensitive `[ZLT]` regex match (existing scorer).
- Total completions: ~12 models × ~35 prompts × 100 = ~42K. Tractable with vLLM on 1×H100.

**Similarity measurement (per (train_T, test_T) pair):**
1. **Semantic cosine** — sentence-transformer embedding of T (per #142 setup).
2. **Lexical overlap** — Jaccard / ROUGE-L over tokenized prompts.
3. **Structural** — features like length bucket, presence of imperative verb, JSON/markdown formatting flag, sentence count. Concrete feature list to be specified by planner.
4. **Task-type** — categorical label (coding / math / format / conversational / other) assigned by Claude Sonnet 4.5 judge over the test-prompt panel (cached).

## Eval / Analysis

For each (model, test-prompt) pair compute marker rate y. Fit a single regression `y ~ semantic + lexical + structural + task_type` plus per-axis univariate regressions; report variance explained per axis and incremental R² when adding non-semantic axes to the semantic-only baseline. Also report per-trigger-family heatmaps: rows = train trigger family, cols = test trigger family, cell = pooled marker rate.

## Success criterion

H1 confirmed if mean marker rate at test_T = train_T is ≥3× the rate at the most-dissimilar bystander prompts (matching the gradient strength seen in #173 between matched and fully-mismatched conditions; pooled across seeds, p<0.01 vs noise floor).

H2 confirmed if at least one non-semantic axis adds ≥0.05 R² to a semantic-only regression of marker rate (n ≈ 12 models × 35 test prompts = 420 pairs).

## Kill criterion

If H1 fails — i.e., marker rate at test_T = train_T is statistically indistinguishable from rate at unrelated bystander prompts — non-persona triggers don't form a coherent leakage gradient at all and the comparative-axis question is moot. Report null and stop.

If only one axis (semantic) explains all variance, the result reduces to a replication of #142 in a non-persona setting; we'd write that up as a smaller scope finding rather than expand the scoping.

## Compute estimate

- 12 LoRA runs × ~30 min each on 1×H100 = ~6 GPU-h training
- ~42K vLLM completions on 1×H100 ≈ 4–6 GPU-h eval
- Analysis local on the VM
- **Budget: ~15 GPU-h**, single H100 pod (`--intent lora-7b`)

## Pod preference

`--intent lora-7b` (1×H100). New ephemeral pod `epm-issue-181`.

## References

- Parent: #173 — markers are prompt-gated, not content-primed (establishes that the *system prompt* is the gate; this issue probes what makes two gates similar).
- Related: #142 — JS divergence > cosine for persona-leakage prediction (we extend by going beyond persona prompts and adding non-semantic axes).
- Related: #99 — leakage generalizes across 4 behavior types (we test whether it generalizes across non-persona conditioning surfaces).
- Code touched: `src/explore_persona_space/leakage/`, `src/explore_persona_space/eval/trait_scorers.py`, new condition configs under `configs/condition/`.

## Plan deviations allowed without re-asking

- Exact wording of the 4 chosen triggers (planner picks a representative one per family from a candidate list)
- Exact size of the eval prompt panel (in [25, 50])
- Hyperparameters within the existing LoRA-training defaults

## Plan deviations that MUST come back to the user

- Adding/removing similarity axes
- Switching backbone or training method (full-FT instead of LoRA)
- Changing the marker token (`[ZLT]` is fixed by precedent)
- Going beyond the compute budget by >2x

Parent: #173

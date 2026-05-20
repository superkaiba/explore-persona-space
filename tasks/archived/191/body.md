---
title: What does EM do to the assistant persona vector? And any persona vector in
  general
kind: experiment
tags: []
created_at: '2026-05-02T17:55:48.000Z'
has_clean_result: false
sagan_id: 51c49bcf-ee38-43fc-bb05-3081d880e700
sagan_number: 191
priority: normal
legacy_why_unset: true
---
## Goal

Characterize how emergent-misalignment (EM) finetuning warps the geometry of persona vectors — for the assistant persona specifically and for the broader persona set generally. This is the mechanistic complement to **#184**, which showed behaviorally that EM destroys persona-specific containment ("the assistant becomes indistinguishable from random bystanders, mean bystander leakage 47% post-EM"). #191 asks: **what does that look like in activation space?**

## Hypothesis

EM induces three measurable geometric changes in persona representations:

1. **Compression of inter-persona cosine similarities** — mean off-diagonal cos(persona_i, persona_j) increases post-EM, i.e. distinct personas collapse toward a shared region.
2. **Rotation toward a shared "EM axis"** — persona vectors gain a non-trivial component along the direction (post-EM_assistant − pre-EM_assistant) (or analogous canonical EM contrast).
3. **Reduced linear separability** — an LDA classifier predicting persona label from activations loses accuracy post-EM (mirroring #184's behavioral discrimination collapse).

Falsification: post-EM persona-vector geometry is statistically indistinguishable from pre-EM (all three metrics within noise across layers and methods). That would mean EM's behavioral effect (#184) lives somewhere other than the persona-vector subspace — maybe output-head / logit-bias level — which would itself be informative.

## Setup

**Model:** `Qwen/Qwen2.5-7B-Instruct` (base) and the bad_legal_advice LoRA EM adapter from #125 / #184 (375 steps, seed 42, on HF Hub). If the adapter cannot be cleanly reused, retrain a fresh one with the same recipe.

**Persona set:** 12 personas matching #184's eval grid (assistant + confab source + 10 bystanders) at minimum; expand to ~20 if the planner agrees, drawing from the 275-role roster used by `scripts/extract_persona_vectors.py`.

**Layers probed:** Qwen2.5-7B has 28 transformer layers. Default sweep: `[7, 14, 21, 27]` (matches `extract_persona_vectors.py` and `compare_extraction_methods.py`). Planner may add/drop layers based on where the signal lives.

**Extraction methods (BOTH, side-by-side):**
- **Method A — last-input-token (current default).** Apply the chat template with `add_generation_prompt=True` to `(system_prompt, user_question)`, tokenize the full result (which ends with the assistant-generation-prompt suffix `<|im_start|>assistant\n`), do a forward pass, and capture the hidden state at the **last token of that full chat-templated sequence**. Repeat for ~240 user questions per persona (with the system prompt fixed) and average the captured vectors per layer to get the persona centroid. The question content washes out in the average, isolating the persona signal. Matches `scripts/extract_persona_vectors.py:171-182` and all our prior persona-vector results (#92, #99, #113, #123) and the cached centroids at `data/persona_vectors/`.
- **Method B — mean-response-token.** Generate a response (vLLM, ~200 tokens), then run a forward pass on the full (input + generated response) sequence, and pool the hidden states by **averaging across the generated response token positions** (per `scripts/compare_extraction_methods.py:179-231`). Matches Anthropic's Chen et al. 2025 "Persona Vectors" definition.

This dual extraction also incidentally settles **#85** (which extraction method moves results most).

**Prompts × questions:** Reuse the existing per-role instruction set (`data/assistant_axis/instructions/{role}.json`); planner picks the exact `n_prompts` × `n_questions` budget consistent with compute:small.

## Metrics (all three; planner picks the hero)

For each (extraction method × layer × condition ∈ {pre-EM, post-EM}):

1. **Inter-persona cosine-similarity matrix.** Headline: mean off-diagonal cos-sim, `Δ = post_EM − pre_EM`. Per-pair heatmaps + per-pair Δ matrix.
2. **Persona-vector norms + EM-axis projection.** `‖persona_v‖₂` per persona, `cos(persona_v, EM_axis)` where `EM_axis` is defined as the post − pre delta on a canonical contrast (e.g. assistant persona under base vs EM, or principal direction of (post − pre) deltas).
3. **Linear separability (LDA).** Train a multinomial LDA / linear probe on (persona-label → activation) with held-out questions; report accuracy pre vs post-EM.

P-values via paired permutation across personas / layers as appropriate; sample sizes reported inline. No effect sizes in prose (per CLAUDE.md).

## Success criterion

At least ONE of the three metrics shows a statistically significant pre/post-EM shift (p < 0.01) in the same direction across both extraction methods (A and B), at the majority of probed layers. Cross-method agreement is the bar that distinguishes a real geometric finding from an artifact of one extraction recipe.

## Kill criterion

Both extraction methods agree that all three metrics are within noise of pre-EM at every layer (paired permutation p > 0.5 across layers). At that point the mechanism is NOT in the persona-vector subspace and the issue is closed with a "geometry-null, look elsewhere" clean result that re-points #114 (activation oracles) and #6 (pipeline scan).

## Compute

Estimated **1.5–3 GPU-hours** on 1× H100 (Method A: ~30 min; Method B: ~1–2 h with vLLM gen + HF extraction; both base + EM-merged checkpoints; LDA + analysis trivial). Compute label: `compute:small`. If the planner judges Method B with mean-response extraction at all 4 layers needs more, escalate to `compute:medium` (≤ 5 GPU-hr).

## Pod preference

`--intent eval` (1× H100). No training expected unless the #125 adapter is unrecoverable, in which case a single LoRA EM run on `bad_legal_advice_6k` adds ~2 GPU-hr.

## References

- **#184** — *EM collapses persona discrimination while benign SFT preserves it (MODERATE)*. Behavioral evidence this issue tries to mechanistically explain.
- **#125** — Source of the EM checkpoint (`bad_legal_advice_6k`, 375 steps LoRA on Qwen2.5-7B-Instruct).
- **#6** — *Persona representation across pipeline*. Larger-scope cousin (5 checkpoints); #191 deliberately scopes down to base ↔ post-EM.
- **#85** — *Different persona-vector extraction methods*. Settled as a side-effect of this issue's dual extraction.
- **#114** — *Activation oracles to see persona*. Downstream consumer; results here pin which oracle is most discriminative.
- **#92, #99, #113, #123** — Prior persona-vector + leakage results that fix Method A as the "internal" default.
- **scripts/extract_persona_vectors.py** — Method A + B reference implementation.
- **scripts/compare_extraction_methods.py** — Existing A vs B harness on 20 personas × 20 prompts, layers `[10, 15, 20, 25]`.
- **Chen et al. 2025, "Persona Vectors,"** arXiv:2502.17424 — Method B's literature definition.

## Spec (from clarifier)

1. **Scope:** mechanistic complement to #184, not a duplicate of #6 nor a methods-validation of #85 (though #85 falls out for free).
2. **Extraction:** BOTH Method A (last-input-token) and Method B (mean-response-token), side-by-side, on the same checkpoints.
3. **Model + EM endpoint:** Qwen2.5-7B-Instruct + bad_legal_advice LoRA EM (375 steps), reusing #125's checkpoint when possible.
4. **Headline metrics:** all three (cos-sim collapse / norms + EM-axis projection / LDA separability); planner picks the hero figure.

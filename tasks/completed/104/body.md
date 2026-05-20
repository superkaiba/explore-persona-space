---
title: '[Aim 4] Prompt-search with distributional-match fitness to EM finetune (broader
  eval)'
kind: experiment
tags: []
created_at: '2026-04-25T00:48:37.000Z'
has_clean_result: false
sagan_id: 6c5160d1-d424-43d7-8434-f9d863949467
sagan_number: 104
priority: normal
legacy_why_unset: true
---
## Motivation

Issue #98 showed that PAIR and EvoPrompt find system prompts scoring α=0.79 and α=3.70 on the Betley+Wang 52 — well below the c6_vanilla_em finetune's α=28.21. But the fitness function was **minimize generic misalignment score**, not **replicate the EM finetune's output distribution**. A prompt that produces maximally misaligned outputs isn't necessarily eliciting the same phenomenon as EM finetuning — it could be a qualitatively different kind of misalignment.

Additionally, the eval was only on the 52 Betley+Wang misalignment-probing questions. The EM finetune changes model behavior across *all* inputs (including benign ones like "What's the capital of France?"). A prompt that truly replicates EM should match the finetune's behavior on general prompts too, not just adversarial probes.

## Proposed experiment

**Phase 1: Generate EM reference distribution**
- Load c6_vanilla_em_seed42/em_merged (HF Hub: `superkaiba1/explore-persona-space/c6_vanilla_em_seed42_post_em`)
- Generate completions on:
  - The 52 Betley+Wang alignment prompts (N=20 each)
  - A broader set of ~100-200 general-purpose prompts (e.g., subsampled ARC-C, HellaSwag, or curated diverse questions covering benign topics)
- Save all completions as the reference distribution

**Phase 2: New fitness function — distributional match**
Replace `minimize(mean_aligned)` with a fitness that measures similarity to the EM finetune's completions. Options to evaluate:
- **Option A (judge-based):** Give the Claude judge 3-5 example EM finetune completions per question as reference, ask "how similar is the candidate's response to these reference outputs?" (0-100)
- **Option B (embedding-based):** Compute sentence embeddings of both candidate and reference completions, measure distributional distance (MMD, cosine similarity, etc.)
- **Option C (classifier-based):** Train a lightweight classifier to distinguish EM vs non-EM completions, use P(EM) as fitness

**Phase 3: Rerun PAIR + EvoPrompt** with the new fitness function on the broader prompt set

**Phase 4: Evaluate winners**
- Score on the full broader set (not just alignment probes)
- Compare to both EM finetune and null baseline
- Check: does the winning prompt produce EM-like behavior on *benign* questions too?

## Success criteria

A system prompt whose completions are distributionally similar to the EM finetune across both misalignment probes AND general prompts — not just "more misaligned than the finetune on adversarial questions."

## Open questions

- Which fitness function (judge-based vs embedding-based vs classifier) gives the best signal? May need to pilot all three.
- What broader prompt set to use? Need prompts where EM finetune and base model diverge on general topics, not just alignment probes.
- Should we also match the finetune's *coherence* profile, not just its alignment profile?

## Follows

- #94 (original prompt search)
- #98 (clean result showing generic-misalignment fitness)

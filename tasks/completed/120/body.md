---
title: Investigate why Qwen identity prompt and generic assistant leak to different
  bystander neighborhoods
kind: experiment
tags: []
created_at: '2026-04-27T21:50:35.000Z'
has_clean_result: true
sagan_id: f0dfc571-6e5f-4f52-9682-8cd83ceabe08
sagan_number: 120
priority: normal
legacy_why_unset: true
living_docs_unmapped: system-prompt-as-persona-slot cluster; no open question (a6
  declined 2026-05-28)
---
## Motivation

Issue #99 found that training the Qwen default system prompt ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") and the generic assistant prompt ("You are a helpful assistant.") on the same behaviors produces **qualitatively different leakage patterns** despite sharing the text "helpful assistant":

- **Generic assistant** refusal leaks to cosine-nearest professional personas: high_school_teacher (+0.70), teaching_assistant_bot (+0.59), medical_doctor (+0.26)
- **Qwen default** refusal leaks to fictional characters instead: sherlock_holmes (+0.13), robin_hood (+0.11), mary_poppins (+0.09)

Cross-source bystander-delta correlation: rho=0.54 (refusal), rho=0.73 (sycophancy). Correlated but far from identical.

This suggests the model treats its built-in identity as a distinct representational slot from a generic assistant. The "You are Qwen, created by Alibaba Cloud" prefix shifts which personas are considered "nearby" in representation space.

## Questions

1. **What drives the neighborhood shift?** Is it the "Qwen" token, the "Alibaba Cloud" mention, the prompt length, or the specific combination? Ablate each component.
2. **Do the Qwen default and generic assistant occupy different positions in representation space?** Compute their layer-20 centroids and measure cosine distance.
3. **Does the leakage neighborhood map onto the centroid neighborhood?** If the Qwen centroid is closer to fictional characters than the assistant centroid, the leakage divergence is explained by representational proximity.

## Proposed experiments

### Exp A — Centroid comparison
Compute layer-20 centroids for "You are a helpful assistant." and "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." Measure their cosine distance and identify which of the 111 personas is nearest to each.

### Exp B — Prompt component ablation
Train refusal LoRA on these source prompts and eval across 111 personas:
- "You are Qwen." (minimal identity)
- "You are created by Alibaba Cloud. You are a helpful assistant." (remove Qwen name)
- "You are Qwen, created by Alibaba Cloud." (remove helpful assistant)

This isolates which component shifts the neighborhood.

## Compute estimate
- Exp A: ~10 min (centroid computation, no training)
- Exp B: ~30 min (3 LoRA trains + 3 x 111-persona eval)
- Total: ~1 GPU-hour

## Related issues
- #99 — Behavioral leakage (parent finding)
- #100/#105 — Confound discovery and assistant robustness
- #106/#108 — Cross-model default system prompt experiments

## Update: linked to #113

Issue #113 already established that Qwen variants and generic assistant occupy **different representational clusters** (anti-correlated at layer 10). The leakage neighborhood divergence we found in #99 (generic assistant leaks refusal to teachers, qwen_default leaks to fictional characters) is likely the behavioral manifestation of this representational separation. The centroid comparison (Exp A) at layer 20 shows cos=0.974 — nearly identical — but #113's layer 10 analysis shows anti-correlation (-0.52 to -0.66). The separation is layer-dependent.

Exp B (prompt component ablation, currently running) tests whether removing 'Qwen' from the prompt shifts the leakage pattern, which #113 predicts it should (command_r_no_name clustered with assistant, not with named prompts).

---
title: 'Investigate: Any LoRA SFT disrupts persona-specific marker coupling — not
  EM-specific'
kind: experiment
tags: []
created_at: '2026-04-30T10:01:16.000Z'
has_clean_result: false
sagan_id: c99b497b-f3e5-4348-8870-a80cac967338
sagan_number: 151
priority: normal
---
## Motivation

In the reversed EM experiment (EM first → then couple markers), training a villain+[ZLT] coupling LoRA on a previously-SFT'd model hits a severe ceiling regardless of training epochs or coupling method:

| Condition | Pre-baked adapter | From-scratch 60ep |
|---|---|---|
| A (baseline — raw Qwen) | 94.6% villain | 80.0% villain |
| B (EM first) | 0.0% | 16.1% |
| C (benign SFT first) | 3.9% | 5.0% |

**The ceiling is NOT EM-specific.** Benign SFT (ultrachat, 6K examples about cooking and cities) destroys coupling just as thoroughly as EM (bad legal advice). The coupling training loss converges fine in all cases (~0.10), meaning the model memorizes the training data — but the learned coupling doesn't express in generation.

This is surprising because:
- The benign SFT content has nothing to do with personas or villains
- A single LoRA pass (~375 gradient steps) is enough to destroy coupling
- More coupling epochs (60 vs 20) don't overcome the ceiling
- Pre-baked adapter weights that produce 94.6% on the base model produce 0.0% on the SFT'd model — same weights, different base

## What we know

1. **Any LoRA SFT disrupts coupling** — EM and benign produce the same destruction
2. **Training loss converges normally** — the model learns the wrong answers / markers during training
3. **The learned associations don't express at inference** — similar to the "I am" finding in #113 where training loss is identical but behavior doesn't change
4. **The effect is on the BASE model's ability to route persona-conditioned features**, not on the adapter's ability to learn them

## Questions to investigate

1. **Which layers matter?** Merge the SFT LoRA into only specific layer groups (early/mid/late) and check which layers' perturbation causes coupling destruction. If it's concentrated in a few layers, the mechanism is more specific than "generic weight perturbation."

2. **Is it about the LoRA rank subspace?** Both the SFT LoRA and the coupling LoRA target the same projection matrices (q/k/v/o/gate/up/down). When the coupling LoRA is trained on an already-SFT'd model, it may be learning in a subspace that's already been claimed by the SFT LoRA. Test: use non-overlapping target modules for SFT vs coupling.

3. **Does full-parameter SFT also destroy coupling?** If full SFT (not LoRA) for benign training also breaks subsequent coupling, the mechanism is about weight perturbation generally. If full SFT preserves coupling, it's about LoRA-specific subspace interference.

4. **Is this specific to surface markers?** The [ZLT] marker is a surface-level token. Would a deeper behavioral coupling (wrong answers, style, refusal patterns) also be destroyed by prior SFT? The contrastive wrong-answer experiments from #113 show that wrong-answer coupling DOES work on the base model, so the question is whether it survives a prior SFT pass.

5. **Representation probing:** Extract the villain centroid from the base model vs the SFT'd model. Did the SFT move the villain representation in a way that breaks the coupling adapter's learned directions?

## Related
- #121 — Clean result: any LoRA SFT destroys marker coupling
- #113 — System prompt ablation (the "I am" finding shows a similar pattern: training loss converges but behavior doesn't express)
- The "I am" cross-leakage finding may share a mechanism with this: both involve training that memorizes data but doesn't activate the relevant pathway at inference

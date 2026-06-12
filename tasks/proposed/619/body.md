---
title: Think through LoRA placement (which modules/layers the adapter lives in) and
  whether the leakage conclusions are placement-conditional
kind: analysis
tags: []
created_at: '2026-06-12T19:53:35Z'
has_clean_result: false
origin_prompt: yes it's about lora placement. Add a human task for me to address to
  think about this
---
## Summary

Human thinking task (not for agent execution): think through what "where the LoRA lives" means for the project's implant/leakage conclusions, and decide whether a placement ablation experiment is worth filing.

Things to think through:

1. **Placement choices** — which module set the adapter attaches to (attention q/k/v/o vs MLP projections vs both), which layer range, and how rank is distributed. All current results use one default placement (attention+MLP projections, `lm_head`/embeddings excluded).
2. **Which conclusions are placement-conditional.** The LoRA-vs-full-FT "similar leakage profiles" claim (r = 0.994 on the marker) was measured at the default placement only; #606 already shows the equivalence breaks for sycophancy. Does placement explain or modulate that gap?
3. **Interaction with the LoRA-anatomy results** — #604 found a seed-stable write direction with a seed-arbitrary top-1 key; would a different placement (e.g. MLP-only, or a narrow layer band) concentrate or relocate the write, and would that change bystander leakage?
4. **Decision output:** either file a concrete placement-ablation experiment (re-run the LoRA-vs-FT / leakage comparison across 2-3 placements) or record why the default placement is not load-bearing.

## Provenance

Consolidated from the 2026-06-11 mentor meeting notes (docs/mentor_updates/2026-06-11.md, "careful about where the LoRA lives") during the task #616 cross-check; confirmed as a LoRA-placement concern.

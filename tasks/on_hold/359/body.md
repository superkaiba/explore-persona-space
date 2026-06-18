---
title: 'Hypothesis: post-training represents backdoors less saliently than pretraining'
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:32:45.000Z'
has_clean_result: false
sagan_id: 4e84cb88-5df7-417b-b74b-68273d57890f
sagan_number: 359
priority: normal
---
**Hypothesis to follow up on.** Pretraining must keep good loss on a wide distribution, so any backdoor learned in pretraining has to coexist with a lot of other structure and may end up more saliently represented in residual space. Post-training (SFT, RLHF, etc.) does not face that same wide-distribution loss pressure, so backdoors implanted at the post-training stage might be represented less saliently.

If true, this predicts:
- Probes for backdoor presence (see #358) work better on pretraining-injected backdoors than on post-training-injected ones.
- Pretraining-injected backdoors should be easier to find by geometric methods; post-training-injected backdoors may need behavioral search.

Source comments (mentor update, 2026-05-11):
> Post-training might represent less saliently

> Pretraining -> needs to continue getting good loss on wide distribution
> Post training -> doesn't need to anymore

From mentor update on #276 — *A pretraining-data-poisoned Qwen3-4B backdoor only fires on the exact trigger tokens — paraphrases don't activate it, and base-model similarity to the trigger doesn't predict which inputs fire.*

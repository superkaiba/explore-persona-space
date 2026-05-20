---
title: Long term plan
kind: experiment
tags: []
created_at: '2026-04-30T13:49:57.000Z'
has_clean_result: false
sagan_id: 388a279e-f5bf-4715-8e40-d821e58f2689
sagan_number: 152
priority: normal
legacy_why_unset: true
---
We have this phenomenon of persona leakage with arbitrary behaviors.
Something similar has been found in the emergent misalignment/inoculation prompting literature by the Conditional Misalignment paper where a paper with an inoculation prompt continues to exhibit misalignment under a prompt that shares surface similarity but means the opposite

The chunky post training paper found that models over-apply spurious correlations

We propose that inoculation prompting is basically a special case of a sleeper agent. A model trained to do something when it sees a specific prompt 

But the specific prompt is so far away from the assistant's base distribution that none of that behavior leaks
But similar prompts can re-elicit the behavior. Can be measured based on cosine similarity or based on KL/JS divergence over output distributions

Of course inoculation works because the behaviors are so OOD and you can't reuse similar prompts because they would just re elicit the behavior

You can predict conditional misalignment from cosine similarity (~) but more distributional similarity of prompted models
It's not just conditional misalignment but conditional any behavior

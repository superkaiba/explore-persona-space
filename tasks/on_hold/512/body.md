---
title: Develop the framing for predicting generalization from activation vectors (Dan)
kind: analysis
tags:
- needs-thomas
created_at: '2026-06-07T01:53:22Z'
has_clean_result: false
---
Source: Dan Mossing, research-log-exploring-persona-space Slack, 2026-06-06. Open-ended thinking/ideation issue (the 'think about the framing' ask), parent to the concrete experiments.

Core question: can we predict how training on c0 -> b0 generalizes to c1 -> b1 directly from the activation-vector representations of the contexts and behaviors?
- Contexts as activation vectors v_c (e.g. token after the last user-message token).
- Behaviors as activation vectors v_b (e.g. token after the assistant-message token).

Threads to develop:
1. Current predictor: cosine similarity / JS divergence over the mean context activation vector (token right before the assistant answers). Known limitation: throws away the distribution of activations over user prompts. Is there a better distance/representation?
2. Dan's structural hypothesis (filed separately as EPS #510, the concrete experiment): if behavior was v_b' = M v_c', then after training on c' -> b'' it may be a rank-1 update v_b = (M + (v_b'' - v_b') v_c'^T) v_c. Test whether this rank-1 shape predicts post-training behavior better than / complementary to cosine-sim.
3. Validate the predictor on a SURPRISING-generalization setting: He, Xia, Henderson, 'What is in Your Safe Data?' (arxiv 2404.01099) - benign fine-tuning data that breaks safety. Saved to papers/to-read.md.
4. Validate on UNSURPRISING generalization too (the kind nobody writes a paper about), e.g. align a model to respond a certain way in English and check transfer to Spanish.

Goal of this issue: think through the right formalism + metric for the predictor, decide which sub-experiments are worth running, and how they tie back to the chunky-post-training / leakage prediction work (Experiment 2). #510 is the first concrete experiment under this umbrella; consider merging #510 into this if you'd rather track them as one.

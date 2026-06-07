---
title: Test rank-1 activation-space update as a predictor of cross-context generalization
  (Dan framing)
kind: experiment
tags:
- needs-thomas
created_at: '2026-06-07T01:48:04Z'
has_clean_result: false
---
Source: Dan Mossing, research-log-exploring-persona-space Slack, 2026-06-06.

Framing: represent contexts as activation vectors v_c (token after last user-message token) and behaviors as v_b (token after assistant-message token). Question: can we predict how training on c0 -> b0 generalizes to c1 -> b1 from their activation vectors?

Dan's concrete hypothesis: if the model's behavior was representable as v_b' = M v_c', then after training on c' -> b'' it might be representable as a rank-1 update:
  v_b = (M + (v_b'' - v_b') v_c'^T) v_c
He notes it's 'surely a bit wrong, but maybe something of this vague shape would be predictive.'

Why it matters: this is a concrete mechanistic model for the leakage-prediction work (Experiment 2). Currently we predict leakage with cosine similarity / JS divergence over mean context activation vectors. The rank-1 update is a stronger, testable structural hypothesis. Test whether it predicts post-training behavior better than / complementary to the cosine-sim predictor.

Next step is Thomas's: formalize the test (which layer for v_c/v_b, how to estimate M, what train/eval contexts), then it can be run. Tagged needs-thomas because the setup needs his judgment before an agent runs it.

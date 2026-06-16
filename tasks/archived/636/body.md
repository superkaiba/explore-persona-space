---
title: 'Expand-vs-sharpen x reweight-vs-new-feature: base vs RL vs SFT/distilled Qwen-2.5-7B
  (extends Dedicated Feature Crosscoders)'
kind: experiment
tags:
- needs-thomas
created_at: '2026-06-13T22:17:18Z'
has_clean_result: false
---
MOTIVATION (from the 2026-06-13 deep-research reports, reference/rl-elicit-capabilities-2026-06-13.md + reference/rl-capability-theory-and-mechinterp-2026-06-13.md): the behavioral 'RL sharpens within base support / distillation expands the boundary' contrast (Yue et al 2504.13837, pass@k) has NOT been tested at the FEATURE level, and the 'budgeted capability' notion (cost-to-elicit) has no unified formalization. Both can be attacked at once on Thomas's open-weights + crosscoder stack.

CORE EXPERIMENT: take Qwen-2.5-7B base, and matched RL-tuned and SFT/distilled variants on the same target behaviors. On the SAME set of behaviors, measure (a) BEHAVIORAL: pass@k curves to classify each behavior as 'expanded' (solvable post-tuning but not by base at feasible k) vs 'sharpened' (base already solves at high k); (b) MECHANISTIC: artifact-robust crosscoder (BatchTopK + Latent-Scaling per Minder et al 2504.02922, to avoid the chat-feature hallucination artifact) between base and each tuned model, classifying each behavior's change as 'reweight/gate existing latent feature' vs 'genuinely new feature'.

HYPOTHESIS: expand-vs-sharpen (behavioral) co-varies with new-feature-vs-reweight (mechanistic); RL changes are predominantly reweighting of high-mass base features (consistent with sharpening), SFT/distillation more often adds/composes new features (consistent with boundary expansion). STRONGER: cost-to-elicit (minimal-trainable-params a la Hu et al Dkgx2pS4Ww / fine-tuning bits) should be PREDICTABLE from base-model internals (low-cost = high-mass base feature = reweighting), tying directly into Thomas's pre-training-prediction / behavior-leakage thread.

WHY IT MATTERS: nobody has run the behavioral-x-mechanistic version on the same behaviors; it is a direct extension of Dedicated Feature Crosscoders, gives a feature-level account of what RL vs SFT actually do, and operationalizes a budgeted capability measure grounded in mech-interp. Controls: equal-norm random-direction baseline, matched-compute SFT vs RL, multiple behavior families (refusal, a reasoning skill, a persona/style behavior). Open Qs to shape with Dan: which behaviors, how to get matched RL vs SFT variants (train vs use existing checkpoints), pass@k k-budget, crosscoder training compute.

NEXT STEP: Thomas to shape the design (behavior set + variant sourcing), then it becomes agent-runnable. Related EPS #635 (capability-theory survey).

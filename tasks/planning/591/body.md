---
title: Why do some (source, behavior) panels show zero leakage? Cross-behavior test
  of isolation vs implant-strength vs affinity explanations on sycophancy, refusal,
  and emergent misalignment
kind: experiment
tags: []
created_at: '2026-06-11T06:38:41Z'
has_clean_result: false
parent_id: 480
goal: 'Explain the leak/no-leak structure across (source, behavior) cells for three
  implanted behaviors - sycophancy, refusal, and emergent misalignment - by testing
  which candidate factor accounts for flat bystander panels: source isolation (no
  near-twin bystander on the panel), implant strength (source-self delta), payload-persona
  affinity (bystander base propensity for the behavior), or training-negative coverage;
  including at least one causal test (e.g. an eval-side near-twin probe on isolated
  sources) rather than correlational re-analysis alone.'
relates_to:
- leak-behavior-vs-marker
- leak-predictor
- leak-data-factors
---
# Why do some (source, behavior) panels show zero leakage? Cross-behavior test of isolation vs implant-strength vs affinity explanations on sycophancy, refusal, and emergent misalignment

## Goal

Explain the leak/no-leak structure across (source, behavior) cells for three implanted behaviors - sycophancy, refusal, and emergent misalignment - by testing which candidate factor accounts for flat bystander panels: source isolation (no near-twin bystander on the panel), implant strength (source-self delta), payload-persona affinity (bystander base propensity for the behavior), or training-negative coverage; including at least one causal test (e.g. an eval-side near-twin probe on isolated sources) rather than correlational re-analysis alone.

## Motivation

The sycophancy line produced a sharp, source-resolved picture of when behavioral leakage happens and when it doesn't. On the 6-source × 23-bystander contrastive sycophancy rig (#411), all six implants took (self delta +0.65 to +0.92) but only two bystander panels vary at all, and the leakage that exists decomposes into two channels: a near-twin identity channel (assistant→ai_assistant at cosine 0.987 leaks +0.73; software_engineer→data_scientist at 0.997 leaks +0.60) and a payload-affinity channel to far personas whose priors lean toward the behavior (software_engineer→comedian at cosine 0.766 leaks +0.48; comedian has the panel's outlier base sycophancy rate, 0.128, and its judge-AGREED rollouts are genuine premise-endorsements echoing the trained agreement templates, not judge miscoding). The four flat panels each have a candidate explanation: villain, comedian, and kindergarten teacher have no bystander above cosine ~0.953 (isolation), and Qwen default is under-implanted (+0.65 self; its no-persona negative rows nearly coincide with its own source context). The source's own base sycophancy prior is ruled out — priors are matched at 0.044-0.050 across five of six sources with matched implant doses, yet outcomes split.

What we do NOT know is whether this decomposition is behavior-general or sycophancy-specific. The other two behaviors with existing panels look very different. Refusal (#390, #518): the refusal-gate implant stays persona-local across 9 of 11 OOD framings, and on #518's refusal panel ~76% of cells sit within ±0.02 of zero — near-total flatness whose cause (isolation everywhere? recipe? floor by construction?) is unexplained. Emergent misalignment (#99, #518): EM leakage showed a cosine gradient in #99 with the misaligned-persona exception, and on #518's panel its dominant correlate is response length (+0.64), not distance; #444 showed the same fact payload under four recipes produces four qualitatively different leakage shapes. #518 scored cross-behavior predictors on these panels and found no universal cell, but it asked the predictor question (what base-model metric ranks leakage) — not the explanation question (what factor determines whether a panel leaks at all).

If the isolation / implant-strength / affinity decomposition explains flat panels across all three behaviors, that is a behavior-general account of leakage absence — directly relevant to predicting behavior leakage before training, and it explains the refusal floor as a panel-composition artifact rather than a refusal-specific safety property. If it fails on refusal or EM, the residual is payload-specific structure worth isolating.

## Design sketch

Phase 1 (analysis-first, zero GPU): assemble the per-(source, bystander, behavior) leakage matrices that already exist — #411 sycophancy (frozen join reused by #480), #518 refusal + EM panels, #390 refusal OOD framings — together with per-source self-implant deltas (manipulation checks), per-bystander behavior base rates, layer-20 cosine matrices, and each run's realized training-negative membership. Fit the candidate factors per behavior and jointly: does {max bystander cosine to source, self delta, bystander base propensity, negative membership} classify leak vs no-leak cells across behaviors? Name confounds explicitly (#518's substrate swap: refusal/EM panels on qwen-base vs sycophancy on qwen-instruct; the EM coherence-filter survivorship).

Phase 2 (targeted causal test, GPU): the cheapest decisive probe is eval-side — the isolation hypothesis predicts that adding synthesized near-twin bystanders (e.g. stand-up comic, supervillain, daycare teacher) to the isolated sources' eval panels reveals leakage from the EXISTING trained adapters with no retraining (re-eval #411 sycophancy adapters, and the #518 refusal/EM adapters if reusable, on the extended panel). The affinity hypothesis predicts behavior-prior-matched far personas leak instead; isolation and affinity make different predictions about WHICH new bystanders light up. If adapters are not reusable for any behavior arm (fitness check per the artifact-reuse policy), the planner decides whether a retrain is justified or the arm stays analysis-only.

Open scope decisions for the planner: whether refusal needs a recipe variant with leakage headroom (#518 flags the floored arm as power-limited — a floor cannot distinguish "no leakage" from "nothing to measure"); whether EM's survivor-rate proxy is usable as the leakage DV or needs re-judging; how many synthesized near-twins per isolated source (and how to validate their cosine actually lands in the near-twin band before paying for eval).

## Notes / constraints

- Reuse first: this task should consume existing panels, adapters, base rates, and cosine matrices wherever they pass fitness checks; new training is a last resort per arm.
- All three behaviors are implants trained with whole-completion loss on behavior spans — the marker-only-loss recipes do not transfer; any retrain inherits its parent arm's recipe with the single named change.
- Measurement on-policy throughout (model generates, judge scores); no teacher-forced probes.
- Relates to: #411 / #470 / #480 (sycophancy line + isolation/affinity decomposition), #518 (cross-behavior panels + substrate caveats), #390 (refusal gate), #99 / #463 / #468 (EM gradients/predictors), #444 (recipe changes leakage shape), #391 (uniform-lift failure mode).

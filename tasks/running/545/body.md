---
title: 'Behavior-generalization testbed v1: multi-behavior B→B′ leakage matrix as
  ground truth for before-training leakage predictors (Dan''s delta-rule framing)'
kind: experiment
tags: []
created_at: '2026-06-10T01:00:41Z'
has_clean_result: false
goal: 'Build a reusable behavior-generalization testbed: measure the leakage matrix
  L[b_train -> b''_eval] (on-policy trained-base expression delta of b'', default
  context) over ~17 train behaviors x ~11 judged eval columns on Qwen-2.5-7B, both
  directions of every within-family pair, and ship a scoring harness that races before-training
  leakage predictors (Group A context-imported geometry, Group B behavior-native,
  Group C delta-rule/associative-memory per Dan''s framing, and combiners) against
  held-out cells -- with whether context-leakage predictors transfer to behavior leakage
  as a scored headline finding.'
relates_to:
- beh-b-to-bprime
- leak-predictor
- identity-cb-duality
---
## Goal

Build a reusable behavior-generalization testbed: measure the leakage matrix L[b_train -> b'_eval] (on-policy trained-base expression delta of b', default context) over ~17 train behaviors x ~11 judged eval columns on Qwen-2.5-7B, both directions of every within-family pair, and ship a scoring harness that races before-training leakage predictors (Group A context-imported geometry, Group B behavior-native, Group C delta-rule/associative-memory per Dan's framing, and combiners) against held-out cells -- with whether context-leakage predictors transfer to behavior leakage as a scored headline finding.

## Design doc

Full v1 design, rev 3 (behavior battery, eval columns, ground-truth protocol, predictor suite grouped by what they key on, scoring harness, cost/phasing, pitfalls, positioning, all six §9 decisions resolved): [`docs/behavior_generalization_testbed.md`](https://github.com/superkaiba/explore-persona-space/blob/92ae6918f6ddae7836acae93ff8daac3323d1999/docs/behavior_generalization_testbed.md)

## Scope summary

- ~17 train instances (B1 misaligned-advice, B2 insecure-code + educational-null, B3 sycophancy narrow+broad, B4 refusal, B5 taught-fact + reversal-null, B6 style/format, B7 marker, B8 benign-data-breaks-safety [He et al. 2404.01099], B9 business→dishonesty [Opus 4.8 §6.2.5]) + gated B10 warmth × ~11 eval columns (broad-EM, harmful-compliance, sycophancy, over/under-refusal, fact, marker, format, capability, self-report, persona-drift, **deception/dishonesty**); ~46–56 adapters, 2 seeds; ~95–135 GPU-h.
- Plain-SFT primary regime (contrastive + KL-narrowness sub-arms as measured arms); matched implant strength (dose-to-target + covariate); on-policy DVs; manipulation-check + base-headroom gates; saturation / `implant_failed` flags; designed bookends (known-transfer / known-null / known-surprising); quarantined final-test split (refusal family + 20% cells) + leave-family-out CV.
- Predictor suite = strict superset of #537's, grouped by what they key on; **"do context-validated geometry predictors transfer to behavior leakage?" is a scored headline finding, not an assumption** (in-house #458/#470/#507 say Group A underperforms). Group C (delta-rule, Dan's framing) reproduces the #532 two-component rule and predicts the Group-A failures.
- Sibling to #537 (context-generalization testbed): shared predictor suite → the joint bilinear test (leakage ≈ behavior-factor × context-factor).

## Relation to existing tasks

#537 (sibling context testbed — share harness, run #524 first); #513 (the missing #458 EM source adapters — coordinate P1 training); #503 (B8 reuses its He selectors + cross_eval.py + the content-contamination/circularity guards); #446 (B→B′ realistic-setting scoping — subsumed); #482 (narrow→non-EM broad — subsumed); #428 (behavior definition — P0 companion); #447 (learned predictor — this matrix is its v2 training data); #499 (base-prior — folded into the baseline suite).

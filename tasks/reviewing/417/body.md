---
title: Weekly mentor update — 2026-05-28
kind: survey
tags:
- mentor-update
created_at: '2026-05-28T21:11:04Z'
has_clean_result: false
---
# Weekly update — May 28, 2026

---

## 1. What I did this week

Grouped by thread. Links are `#N` → eps.superkaiba.com/tasks/N.

- **Sleeper-agent markers don't survive clean training.** A persona+trigger conditional marker stops being emitted after a single epoch of length-matched SFT, whether or not that SFT induces EM (#376, HIGH), and after every tested multi-turn drift history (#377, HIGH). A self-distillation KL-anchor recipe trains an Assistant-keyed marker to 98% emission in phase 1, but one epoch of benign medical SFT drops it to 0/1,800 across 3 seeds (#382, HIGH). An in-context-trained `|AUDIT|` trigger detected 0/600 of the hidden behaviors trained into three Qwen3-14B introspection organisms (#378, LOW). One consistent story: token-level conditional triggers don't survive downstream training.
- **Geometry still doesn't predict marker implantability.** Output-distribution distance from the assistant baseline fails at N=48 (#380, MODERATE). Four geometric/gradient predictors all return null at N=24 across six outcome surfaces (#396, MODERATE), and a symmetric-baseline rerun (assistant vs neutral reference) corroborates it (#415). That's ~5 dead predictors now.
- **Recipe selectivity.** #383 (MODERATE) finds training-recipe parameters can make the marker both stronger and more selective.
- **Behavior-leakage pivot started.** First sycophancy data: 3 of 6 sources replicate #99's sycophancy cosine gradient on held-out wrong claims, 2 sign-flip, 1 collapses (#411, LOW). The B-to-B' rig (#404) is running.
- **Geometry-predicts-generalization test is running** (#406): JS divergence between context transformations as a pre-training predictor of whether Y transfers from T(X) to T'(X) after SFT.

## 2. Open questions

Top 3 (most likely to move the project):

1. **Can pre-training geometry predict post-training transfer?** (#406, running.) Either gives us the first working predictor or kills the geometric framing cleanly.
2. **Does the leakage gradient survive when the target is a behavior (sycophancy) instead of a marker?** (#411 partial, #404 running.)
3. **Is #383's "factors that lift source rate also improve selectivity" real, or the X vs (X−Y) metric artifact?** (#397 re-run.)

Other live questions (several are open mentor asks not yet closed):

- Are in-context demos as useful as personas for trait-implantation? (Dan, 2026-05-22)
- Does JS divergence after convergence training predict marker leakage? (Dan, 2026-05-03)
- Can system-prompting be shown equivalent to persona drift, via log-probs of the system-prompted model on drifted tokens? (Dan, 2026-05-22)
- Does patching system-prompt activations back in post-fine-tuning change the behavior? (Dan, 2026-05-11)
- Multi-persona training: leakage to held-out personas as a function of distance to the trained set (Dan, 2026-05-26; #405).
- Is EM on Qwen coherence collapse or genuine broad misalignment? (needs the Betley judge + coherence filter at n>=10.)
- Are the centroid persona vectors and Chen et al.'s the same object? (#363: same neighborhood, not the same direction.)
- Should the "conditional markers don't survive clean SFT" negatives become a standalone writeup? (#376/#377/#382.)

## 3. Next steps

Currently running:

- **#406 — geometry predicts generalization** (high priority). Code implemented and review-passed; pod provisioning next. Trains on T(X) → Y, tests transfer to T'(X), and regresses post-training transfer on the pre-training JS divergence between context transformations measured on held-out inputs. The headline predictor test.
- **#404 — behavior leakage B → B'** on 4× H100, ~4h wall. Pipeline: data generation → 3 candidate behavior-distance predictors (activation cosine, KL on judge-scored outputs, in-context K-sweep) → SFT over 5 source pairs × 2 seeds → outcome eval → Spearman regression with bootstrap CIs.
- **#397 — recipe-selectivity re-run** with the single-token `※` marker + log-prob eval (parent #383). Sweep resumed after a disk-full crash; ~87 of 108 cells left, ~11h. Tests whether #383's stronger-and-more-selective result is a metric artifact, given that source rate and (source − bystander) selectivity are mechanically correlated.
- **#407 — obscure-but-knowable facts** as a third fact-regime control (vs fictional + future facts). Phase 4-6 chain re-running after a fact-cache keying bug.

Then:

- Land #404 and settle on a behavior-distance metric.
- Re-analyze #383 with source rate partialled out (the selectivity-artifact control).
- Promote the sleeper-marker negatives (#376/#377/#382) into a standalone writeup.

## 4. Admin

- Adding Christina to project?

## 5. Feedback?

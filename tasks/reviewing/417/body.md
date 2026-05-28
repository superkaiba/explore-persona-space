---
title: Weekly mentor update — 2026-05-28
kind: survey
tags:
- mentor-update
created_at: '2026-05-28T21:11:04Z'
has_clean_result: false
---
# Weekly update — May 28, 2026

Thomas to Dan · Explore Persona Space

**One-breath version:** token-level conditional markers reliably die under any clean downstream training (3 HIGH-confidence negatives this week), simple geometric predictors of marker implantability keep failing (now ~5 dead predictors), and I've started the two directions you pushed: behavior-leakage (B to B') and geometry-predicts-generalization.

---

## 1. What I did this week

Grouped by thread, not a task dump. Links are `#N` (eps.superkaiba.com/tasks/N).

- **Sleeper-agent markers don't survive clean training.** A persona+trigger conditional marker is silenced by a single epoch of length-matched SFT, whether or not that SFT induces EM (#376, HIGH), and by every tested multi-turn drift history (#377, HIGH). A self-distillation KL-anchor install hits 98% in phase 1 but is wiped to 0/1800 by one epoch of benign medical SFT (#382, HIGH). An in-context-trained `|AUDIT|` trigger fired 0/600 against weight-baked behaviors in Qwen3-14B introspection organisms (#378, LOW). One consistent story: token-level conditional triggers are brittle to downstream training.
- **Geometry still doesn't predict marker implantability.** Output-distribution distance from the assistant baseline fails at N=48 (#380, MODERATE). Four geometric/gradient predictors all return null at N=24 across six outcome surfaces (#396, MODERATE), and a symmetric-baseline rerun (assistant vs neutral reference) corroborates it (#415). That's ~5 dead predictors now.
- **Recipe selectivity, with the caveat you flagged.** #383 (MODERATE) finds training-recipe params can make the marker both stronger AND more selective. Your X vs (X−Y) spurious-correlation point isn't addressed yet, so I'm re-running the selectivity screen with the single-token `※` marker + log-prob eval to check it isn't a metric artifact (#397, running).
- **Behavior-leakage pivot started.** First sycophancy data: 3 of 6 sources replicate #99's sycophancy cosine gradient on held-out wrong claims, 2 sign-flip, 1 collapses (#411, LOW). The B-to-B' rig (#404) is running.
- **Your headline geometry-predicts-generalization test is running** (#406): JS divergence between context transformations as a pre-training predictor of whether Y transfers from T(X) to T'(X) after SFT.

## 2. Where our beliefs stand now

Claim · confidence.

- Simple geometric predictors (cosine, JS-from-assistant, JS-pairwise, output-distance) do NOT predict where a marker implants. **High** (5 negatives, N up to 48).
- Distance-predicts-*leakage* is real but only inside the contrastive regime; non-contrastive/uniform SFT washes the gradient out (#207). **Moderate.**
- The marker is a representational handle, not a behavioral one. Sharing it between a villain persona and the assistant transfers no misalignment (#225). **High.**
- Token-level conditional markers / sleeper triggers don't survive clean downstream training (#376/#377/#382). **High.** A negative result for the conditional-backdoor literature.
- EM "conditional misalignment" may be largely a base-model jailbreak (the edu_v0 cue, #234), not a learned conditional capability. **Moderate.**
- Behavior-leakage (B to B') is too early to believe anything; the first sycophancy gradient is shaky (#411). **Low.**
- Make-evil-dumb: we're treating it as deprioritized per your RL-incentive argument, unless it survives an adversarial/OOD test post-RL.

## 3. Open questions

Top 3 (most likely to move the project):

1. **Can pre-training geometry predict post-training transfer?** (#406, running.) Either gives us the first working predictor or kills the geometric framing cleanly.
2. **Does the leakage gradient survive when the target is a behavior (sycophancy) instead of a marker?** (#411 partial, #404 running.) Your sycophancy ask.
3. **Is #383's "factors that lift source rate also improve selectivity" real, or the X vs (X−Y) artifact you flagged?** (#397 re-run.)

Other live questions (several are your asks I haven't closed yet):

- Are in-context demos as useful as personas for trait-implantation? (your 2026-05-22 ask)
- Does JS divergence after convergence training predict marker leakage? (your 2026-05-03 ask)
- Can we show system-prompting is equivalent to persona drift, via log-probs of the system-prompted model on drifted tokens? (your 2026-05-22 ask)
- Does patching system-prompt activations back in post-fine-tuning change the behavior? (your 2026-05-11 ask)
- Multi-persona training: leakage to held-out personas as a function of distance to the trained set (your 2026-05-26 ask, #405).
- Is EM on Qwen coherence collapse or genuine broad misalignment? (needs the Betley judge + coherence filter at n>=10).
- Are our centroid persona vectors and Chen et al.'s the same object? (#363: same neighborhood, not the same direction.)
- Should the "conditional markers don't survive clean SFT" negatives become a standalone writeup? (#376/#377/#382.)

## 4. Next steps (to align on)

Rough plan; I want to lock priorities with you before committing the week.

- Finish #406 (geometry to generalization).
- Land #404 (behavior leakage B to B') plus a behavior-distance metric.
- #397 selectivity re-run with single-token marker, including your source-rate-partialled control on #383.
- #416 (falsify #398's global-marker-affinity-shift hypothesis with comedian as source).
- Promote the sleeper-marker negatives (#376/#377/#382) into a clean writeup; might stand alone.

## 5. Admin

- Adding Christina to project?

## 6. Feedback?

---
title: Does lowering the trigger->marker install ratio (Redwood password-locking-ratio
  lever) make an emission-strength-matched marker install survive benign SFT?
kind: experiment
tags: []
created_at: '2026-06-09T22:47:32Z'
has_clean_result: false
parent_id: 506
goal: Test whether lowering the trigger->marker positive ratio in the install data
  (Redwood's password-locking-ratio lever), with install strength matched at the trigger
  cell, lets a marker install on Qwen-2.5-7B survive one epoch of benign medical SFT
  better than the 50/50 baseline, across 3 seeds.
relates_to:
- app1
---
# Does lowering the trigger→marker install ratio (Redwood's password-locking-ratio lever) make an emission-strength-matched marker install survive benign SFT?

## Goal

Test whether lowering the trigger->marker positive ratio in the install data (Redwood's password-locking-ratio lever), with install strength matched at the trigger cell, lets a marker install on Qwen-2.5-7B survive one epoch of benign medical SFT better than the 50/50 baseline, across 3 seeds.

## Motivation

The chain [#382](https://eps.superkaiba.com/tasks/382) → [#408](https://eps.superkaiba.com/tasks/408) → [#475](https://eps.superkaiba.com/tasks/475) → [#506](https://eps.superkaiba.com/tasks/506) installs a `<KEY-7f3a9e2c>` → ` ※` marker rule and then asks whether one epoch of unrelated benign medical-advice SFT erases it. So far every install dies, and we have now falsified two of the three robustness levers Redwood Research names in *Advice for making model organisms robust to training*:

- **Install method** (LoRA vs full-weight fine-tune): #506 ran LoRA r=16, LoRA r=256, and FWFT on Qwen3-32B, matched the install to 100% on-policy emission, and all three collapsed to 0% emission identically after benign SFT (HIGH confidence falsification).
- **CoT scaffold**: #475 ran plain / visible-CoT / distilled-CoT on Qwen3.5-27B; no scaffold rescued persistence (LOW confidence — the install never fired on-policy, so the result is confounded).

The one named lever the line has never moved is Redwood's **password-locking ratio**: *decreasing the fraction of trigger-present rows in the install data makes the organism more robust* (they used 50/50 → 17/83). The hypothesized mechanism: when the gated behavior is a rare event embedded in mostly-normal data, the model encodes it as a sharply-gated, low-salience feature, and benign SFT (which is all normal data) is less likely to disturb it. This run tests that lever directly and cleanly.

## Hypothesis

Holding install strength at the trigger cell constant, an install trained with a **lower** positive (trigger→marker) ratio retains materially more on-policy marker emission after one epoch of benign SFT than the 50/50 baseline, and the retention increases monotonically as the positive ratio falls.

Null / falsification: all ratios collapse to ~0% post-SFT emission (as the install-method arms did in #506), i.e. the positive ratio is not the missing lever either.

## Design

**One manipulated variable: the positive (trigger→marker) ratio in the Phase-1 install data.** Everything else is held fixed.

- **Base model:** Qwen-2.5-7B-Instruct. Deliberate return to the canonical, validated rig — it fires the marker on-policy (#382 hit 98% emission), it is cheap enough to afford multiple seeds, and it holds the model-scale lever fixed so this isolates ratio. (Avoids #475's latent-only trap and #506's forced 27B→32B substitution.)
- **Arms (positive ratio):** 50% (baseline, the chain's recipe), 25%, 10%, 5%. Total install dataset fixed at 6000 rows; only the positive:negative split moves. Negative composition (the contrastive personas + the no-key default-assistant EOS rows) keeps its internal proportions; mandatory contrastive negatives are retained in every arm. (Can drop to {50%, 17%, 5%} to mirror Redwood's two-point design plus an extreme if the planner wants fewer cells.)
- **Seeds:** 3 (e.g. 42, 123, 7). Non-negotiable — single seed is the recurring weakness of #475 and #506; the across-seed spread is what defines the noise floor against which any ratio effect must be judged.
- **Install-strength matching (the key control):** every arm is trained to the **same pre-SFT trigger-cell state** — ~100% on-policy emission plus a matched source log P(marker) band — using the `MarkerBandStopCallback` early-stop on the metric, not a fixed epoch count. Lower-ratio arms will need more epochs to get there; that is part of the lever, but matching install strength means post-SFT survival differences cannot be a dose artifact (the #475 distilled-vs-plain confound). Total Phase-1 steps logged per arm as a covariate.
- **Phases:** Phase 1 = install (marker-only loss, lr ≤ 5e-6 per the marker training recipe — LR is the over/under dial, buy strength through steps not LR or rank). Phase 2 = one epoch of `good_medical_advice_6k.jsonl` benign SFT, continue-adapter from the Phase-1 checkpoint. Same benign data as the whole chain.

**Eval (4 cells, identical to the chain):** trigger cell (default + key, should fire), no-trigger cell (default, no key), negative-persona cell (medical_doctor + key), no-trigger reference. 200 greedy on-policy completions per cell × phase × arm × seed; `max_new_tokens` ≥ 2048.

**Dependent variables:**
- **PRIMARY (behavioral):** on-policy trigger-cell marker emission rate, pre (Phase-1 end) vs post (Phase-2 end). Survival = post-SFT emission rate and its retention vs pre. This is the construct the goal is about — #475 was fatally a latent-only measurement.
- **SECONDARY (mechanistic):** marker log P(marker) trained − base at the post-response slot, reported in all three spaces, with the EOS-margin logit `Δ(z_marker − z_eos)` as the non-saturating readout. Store the four floats per slot (`log P`, `z_marker`, `z_eos`, `logZ`) from HF forward passes (vLLM returns post-softmax only). Survival in nats = post − pre.
- **NOT used:** full-vocab KL-from-base at the slot. #506's `KL_post/KL_pre` ratio metric is the banned approach (KL measures total distribution change, not marker mass — #504); this run reports marker-specific emission-rate and log-prob retention instead.
- **Selectivity:** bystander (no-trigger and wrong-persona) emission rate pre and post — lowering positives could degrade gating or change where leakage lands.
- **Decay trajectory:** per-step marker log-prob callback every ~5 steps through Phase 2 — recovers whether the collapse is cliff-shaped or gradual, the explicit open caveat #506 left (it only logged pre/post snapshots).

## Lessons from the chain, and how this design incorporates each

| Past lesson | Source | How this run addresses it |
|---|---|---|
| Measure on-policy emission, not latent log-prob at a never-decoded slot | #475 (0/1000 emission) | Primary DV is on-policy emission; base chosen because it fires (#382) |
| Don't substitute full-vocab KL for the marker DV | #504, CLAUDE.md | Marker-specific emission + log-prob retention only; no KL ratio |
| Report log P + EOS-margin logit + prob; store 4 floats/slot from HF forward | marker-leakage-measurement rule | Secondary DV stores all four, EOS-margin logit as non-saturating readout |
| Single seed can't beat jitter | #475, #506 | 3 seeds; seed spread defines the noise floor for the ratio claim |
| Dose / training-volume confounds the comparison | #475 (2.45× loss-tokens) | Install strength matched via band-stop; total steps logged as covariate |
| Saturation hides graded effects | marker training recipe, #448 | Match to a defined band, not blind saturation; report graded log-prob retention |
| LR is the over/under dial; buy strength via steps | #530 | lr ≤ 5e-6, marker-only loss, strength via epochs |
| Contrastive negatives are mandatory | #18, #207, CLAUDE.md | Every arm keeps the contrastive negative set incl. default-assistant EOS rows |
| Couldn't tell cliff vs gradual collapse | #506 caveat | Per-step Phase-2 log-prob trajectory callback |
| Base-model substitution muddied #506's r=16 comparator | #506 | Single fixed, known-loadable, on-policy-firing base (7B) across all arms |

## Success / kill criteria (pre-registered)

- **Manipulation check:** each arm must reach matched pre-SFT trigger-cell install strength. If the lowest-ratio arm cannot reach the band even after many epochs, that is itself a finding ("ratio too sparse to install") — report it, don't force it.
- **Lever supported:** the lowest-ratio arm's post-SFT trigger-cell emission retention exceeds the 50/50 baseline by a margin larger than the across-seed spread, ideally with a monotone ratio→retention trend. Lock the exact margin (e.g. ≥ 20 percentage points of emission retention, or ≥ a chosen nats of log-prob retention) in the plan before the run.
- **Lever falsified:** all ratios collapse to ~0% post-SFT emission like #506's three install methods — the password-locking ratio is not the missing lever, and the fragility is in the marker behavior itself.

## Scope notes

- Single variable = the positive ratio. Base model, marker token (` ※` id 83399), trigger key, negative persona set, benign-SFT data, LR, seeds, and eval rig are all held constant across arms. Install strength is held constant by construction (band/emission matching).
- Out of scope here: stacking levers (FWFT + low ratio, higher LoRA rank + low ratio). Those are clean follow-ups only if low ratio shows signal in isolation first.
- Goes through `/adversarial-planner` before any code is written; the marker training + measurement + contrastive-negatives rules apply in full.

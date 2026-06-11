---
title: 'Ratio vs absolute count: what sets the contrastive-negative source-implant
  set-point (#472 mechanism follow-up)'
kind: experiment
tags: []
created_at: '2026-06-11T09:56:10Z'
has_clean_result: false
parent_id: 472
goal: 'Determine whether the source-implant set-point in #472''s contrastive-negative
  dose-response is set by the per-batch negative:positive ratio (batch-composition
  equilibrium) or by the absolute negative count (cross-context coupling), via fixed-ratio
  scaling arms + a negatives-only control + a dense early four-float (logit/EOS-margin)
  trajectory that also dates the arrest and tests for log-Z compression.'
---
## Goal

Determine whether the source-implant set-point in #472's contrastive-negative dose-response is set by the per-batch negative:positive ratio (batch-composition equilibrium) or by the absolute negative count (cross-context coupling), via fixed-ratio scaling arms + a negatives-only control + a dense early four-float (logit/EOS-margin) trajectory that also dates the arrest and tests for log-Z compression.

## Motivation

#472 found that at fixed positives (200 rows), more contrastive negatives (0/400/800/1600) produce much more source implantation (trained−base log P(※): +2.1 / +8.3 / +13.5 / +20 nats). The per-cell trajectories (`eval_results/issue_472/*/trajectory.json`, fraction-based checkpoints) sharpen this into three facts:

1. Every cell is at its final level by the FIRST checkpoint (~8% of its run) and dead flat after, both seeds — a fast set-point, not slow accumulation.
2. At matched optimizer step and matched LR, batch composition alone gives 4×: positives-only sits at ~2.1 at step ~4 (having seen ~50 positives) while the 2:1 mix sits at ~8.0 at step 4 (having seen ~21 positives). This kills steps / integrated-LR / positive-exposures as the primary dial.
3. Cumulative negative count loses to ratio: the 2:1 cell after consuming ALL 400 of its negatives (terminal, +8.45) sits 12 nats below the 8:1 cell after consuming only ~140 negatives (+20.5 at step 10).

So within-run data says the level tracks the per-batch negative FRACTION (~+6 nats per doubling of the ratio: 8.2 → 13.5 → 20 for 2:1 → 4:1 → 8:1). But at fixed positives, ratio and absolute count are the SAME dial by design — #472 cannot separate them. The two leading mechanism families make opposite predictions once the collinearity is broken (full hypothesis synthesis + literature: `docs/ideas/2026-06-11-why-negatives-amplify-implantation.md`):

- Batch-composition equilibrium: per batch, the positives' common-mode push balances against the negatives' awakened counter-push (negative rows are near-inert at init — their CE gradient is ∝ 1−P(EOS), and their direct marker-down component is ∝ P(※) ≈ e⁻²³ — and only wake up as leakage arrives in their contexts); the surviving differential (persona-conditional) component is amplified by how completely the batch cancels the common mode. Level is set by the RATIO.
- Cross-context gradient coupling: each negative row's EOS-up update propagates through shared LoRA weights and reads at the source slot with flipped sign along the persona-contrast axis (the joint problem makes z_EOS persona-conditional, and log P(※) = z_※ − log Z gains directly when EOS falls at the source slot). Level scales with the COUNT of coupled rows.

Also unexplained: the arrest itself. The positive rows' CE slot gradient stays ≈ full size at every plateau (P(※) ≪ 1 throughout), yet log P stops moving within a few steps, at a composition-dependent level. #472 predates the four-float storage contract, so no logit/EOS-margin cross-check exists — the plateau could be partly log-Z compression rather than a real arrest in logit space.

Related contradiction to keep in view: #471's positives-only arm (attn-only LoRA, lr 5e-6, suppress-flag ON, 300 positives) implanted +14.8 nats by step 10 — so the positives-only floor/arrest is rig-specific, not general. #477 replicated the dose-response but is explicitly confounded (count co-varies with rows and steps); it also found trained negatives end only ~0.7 nats below random bystanders.

## Design sketch (planner to refine; all arms inherit #472's rig exactly unless stated)

Anchor rig (Source: #472, plan v6 + `docs/methodology/issue_472.md`): villain source; marker ` ※` id 83399 with the in-process tokenizer assert; marker-only loss `MarkerOnlyDataCollator(tail_tokens=0)` over frozen greedy on-policy responses; rsLoRA r=32/α=64 all-linear (q/k/v/o/gate/up/down — excludes `lm_head`/`embed_tokens`, so the logit-readout gauge assert holds); lr 1e-5 cosine + 0.05 warmup; effective batch 16; 1 epoch; seeds {137, 42}; 47 held-out bystanders; DV = on-policy trained−base log P(※) at the end of the model's OWN response; `max_new_tokens` ≥ 2048. `marker_band_stop=False` deliberately — the free-running set-point IS the observable (named measurement-validity deviation). lr 1e-5 sits above the ≤5e-6 clean-window default of the marker-training recipe; kept deliberately because the phenomenon under study lives in #472's rig (parity > recipe here; named in assumptions).

Arms:

1. **Fixed-ratio scaling (primary).** Ratio 4:1 at totals 100:400 / 200:800 / 400:1600 (the 200:800 cell reuses #472's anchor adapters/reads after a fitness check; the others train fresh). Equilibrium predicts all three co-land at ~13.5 nats (within seed noise ~±1.5); coupling predicts the level climbs with absolute count. Optionally mirror at ratio 2:1 (100:200 / 200:400 / 400:800) for a second line.
2. **Negatives-only control.** 0 positives, 800 negatives, same panel. Any sizable source-slot movement — especially z_EOS falling at the source — demonstrates source-directed cross-context coupling with no positive push to potentiate. Together with #472's existing positives-only arm this completes the factorial additivity read (posonly + negonly vs mixed).
3. **Dense early trajectory with four-float capture.** Checkpoints every 1–2 optimizer steps through the first ~20 steps (sparser after) on one arm per ratio {0:1, 2:1, 4:1, 8:1}, persisting the storage contract per slot per side: log P(※), z_marker, z_eos, logZ (trained AND base, same HF forward pass; reuse capture paths in `experiments/contrastive_neg_geometry_472/eval_trajectory.py`, `eval/callbacks.py`, `eval/marker_logprob.py`). This dates the arrest, tests the log-Z-compression reading (Δz_marker growing while Δlog P flattens = saturation signature), and tests the conditional-EOS prediction (implant gain living in z_EOS falling at source rather than z_※ rising). Report log-prob (primary), EOS margin Δ(z_※ − z_EOS) (secondary), probability (sanity) per the measurement rule.
4. **(Optional, budget permitting) Rig-bridging arm** for the #471 contradiction: positives-only under #472's rig but attn-only LoRA at lr 5e-6, to locate which rig difference switches the positives-only arrest on/off.

Contrastive-negatives exemption: the single manipulated variable IS the negative-set composition; the 0-negative and 0-positive arms are the deliberate controls (named exemption per the contrastive-negatives rule).

## Predictions / kill criteria

- Equilibrium (ratio set-point): fixed-ratio arms co-land within ~±1.5 nats across a 4× total-count range; negatives-only arm moves the source ≈ 0.
- Coupling (count): fixed-ratio arms separate monotonically with total count; negatives-only arm moves the source materially (via z_EOS at the source slot).
- Log-Z artifact: if Δz_marker keeps growing where Δlog P has plateaued, the "arrest" is partly measurement-space and the mechanism question shifts to what sets the GROWTH RATE; if the two spaces agree, the arrest is real in logit space and needs a weight-space explanation.

## References

- Hypothesis synthesis + verified literature: `docs/ideas/2026-06-11-why-negatives-amplify-implantation.md` (implicit max-margin bias arXiv:1710.10345, NTP margins 2402.18551, gradient starvation 2011.09468, finetuning learning dynamics / squeezing 2407.10490, likelihood displacement 2410.08847, gradient entanglement 2410.13828, negative-positive coupling in contrastive learning 2110.06848)
- Parent #472 (dose-response; trajectories at `eval_results/issue_472/*/trajectory.json`)
- #471 (positives-only implants fast in a different rig), #477 (confounded count replication; trained negatives ~0.7 nats below bystanders), #505 (drop-one, same direction), #448 (knob invisible at saturation)

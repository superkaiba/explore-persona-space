---
title: 'Follow-up to #472: does contrastive-negative count raise bystander leakage
  independently of source-implant strength? (implant-decoupled count sweep)'
kind: experiment
tags: []
created_at: '2026-06-03T08:15:54Z'
has_clean_result: false
parent_id: 472
goal: 'Determine whether the row-scaled contrastive-negative-budget recipe (negative-persona
  count co-varying with total negative rows at fixed 200 positives, counts {2,4,8,16})
  independently raises bystander marker leakage net of source-implant strength: match
  source-implant across count levels via a per-cell early optimizer step at fixed
  lr=2e-6 (since #472''s LR lever failed - the implant is LR-insensitive and saturates),
  read leakage on a marker-channel DV (bystander marker-vs-rest Bernoulli KL + emission),
  and test whether count still moves held-out leakage with source-implant held fixed
  (partialling implant and step); a ratio-matched control isolating pure persona-diversity
  from row-mass is a named follow-up.'
relates_to:
- leak-contrastive-negatives
---
# Follow-up to #472: does contrastive-negative count raise bystander leakage independently of source-implant strength? (implant-decoupled count sweep)

## Goal

Determine whether the row-scaled contrastive-negative-budget recipe (negative-persona count co-varying with total negative rows at fixed 200 positives, counts {2,4,8,16}) independently raises bystander marker leakage net of source-implant strength: match source-implant across count levels via a per-cell early optimizer step at fixed lr=2e-6 (since #472's LR lever failed - the implant is LR-insensitive and saturates), read leakage on a marker-channel DV (bystander marker-vs-rest Bernoulli KL + emission), and test whether count still moves held-out leakage with source-implant held fixed (partialling implant and step); a ratio-matched control isolating pure persona-diversity from row-mass is a named follow-up.

## Why this exists (the #472 confound, with data)

#472 found "more negatives → more bystander leakage," but the read was confounded: in #472 the per-cell source-self ΔG trajectory is **flat over training** (the marker implants fully by the first checkpoint and plateaus), and the plateau level is **set by the negative count**, with the count conditions occupying **non-overlapping** implant plateaus:

| condition | source-self ΔG plateau |
|---|---|
| no-negatives | ~2 nats |
| low count (100 ex / 2 personas) | ~8 nats |
| placement arms (mid count, 800 rows) | ~13–15 nats |
| high count (400 ex / 8 personas) | ~19–21 nats |

Because the plateaus don't overlap, **no matched-band re-slice of the #472 data can hold source-implant fixed while varying count** (verified on the trajectories): the low-count cell never reaches 14/20 nats, the high-count cell never comes down to 8. Negative count and source-implant strength are perfectly collinear in #472, so its "count → leakage" and "implant → leakage (Spearman 0.95)" findings cannot be causally separated. This follow-up decouples them.

A second, mechanistic puzzle #472 surfaced: with positives held fixed at 200 rows and one epoch (each positive seen once), **more negatives produced a HIGHER source implant** (8 → 20 nats). That is counterintuitive (negatives train EOS-not-marker at *bystander* slots) and suggests the contrast itself sharpens/amplifies the source-marker direction. The decoupling design should also let us see whether that count→source-implant coupling is real.

## The decoupling — design space (for the adversarial-planner to resolve)

The crux + the hard part: source-implant must be held **matched** across negative-count conditions, but #472 shows the implant is near-instant and recipe-set, so naive early-stopping / fraction-matching cannot match it. Candidate mechanisms (planner picks + grounds the cleanest, with the matching validated by a calibration step, not assumed):

1. **Per-cell LR (or LoRA-rank / training-length) calibration to a target source-ΔG.** Pick a target source-self ΔG (e.g. ~10 nats, in the achievable mid-range). For each negative-count condition, calibrate the per-cell knob (LR is the cleanest single lever) so the source lands at the target ± a tight band, then read held-out leakage. Requires a cheap calibration sweep first (knob → source-ΔG-plateau mapping per count level).
2. **Implant sweep at FIXED negatives (the cleaner inverse).** Hold the negative recipe fixed; sweep source-implant level directly (LR / epochs / positive count). If held-out leakage tracks implant at fixed negatives, implant drives leakage and count's role is just to set the implant level. Combined with the #472 count cells (where implant co-varies with count), this disentangles whether count adds anything *beyond* implant.
3. **Hold total step/pool size fixed across count levels** (subsample so #rows and optimizer steps match) to remove the step half of the confound, separately from the implant-matching.

Decision criterion: with source-implant ΔG held matched (or partialled), does negative count still move held-out leakage? If NO → the #472 count effect was entirely the implant/step confound (negatives don't independently raise leakage; they raise it only by raising the implant). If YES → the negatives have a real independent effect at fixed implant.

## Reuse (do not rebuild)

The entire #472 rig: `src/explore_persona_space/experiments/contrastive_neg_geometry_472/` (the on-policy DV-A logP + DV-B KL trajectory eval, the distance-stratified negative selection, the GPU-pinning + marker-in-R fixes), `scripts/dispatch_neg_geometry_472.py` (the unified 4-GPU dispatcher with `--n-gpus`/`--max-parallel`/`--gpu-id`), the 47-persona held-out bank + centroids on HF, the villain source, marker ` ※` (id 83399), Qwen-2.5-7B-Instruct. The only NEW code is the implant-matching/calibration layer + the cells.

## Open items for the adversarial-planner

- Which decoupling mechanism (1 / 2 / 3 above, or a combination) — and the calibration step that makes the implant-match real, not assumed.
- The achievable matched-implant target (mid-range: source emits reliably, bystanders don't — #472's 1-epoch regime was sub-emission, P(※)≈0.17 max even on the source; a follow-up should target a readable, behaviorally-emitting implant so the DV is emission, not just sub-emission log-prob drift).
- Whether to also fix total optimizer steps across count cells.
- Seeds, compute estimate, pod intent (the user has been running #472 on 4× H100).

---
title: 'Training dynamics of source implantation vs bystander leakage: positive-only
  vs contrastive at matched recipe, full bystander panel probed per checkpoint'
kind: experiment
tags: []
created_at: '2026-06-11T08:02:39Z'
has_clean_result: false
parent_id: 472
goal: Measure the training-step dynamics of source marker implantation AND bystander
  marker leakage — the full bystander panel probed at the same eval intervals as the
  source, under the four-float storage contract (log P(marker), z_marker, z_eos, logZ;
  trained AND base per slot) — in positive-only vs contrastive training at an otherwise
  matched recipe, to determine (1) whether bystander marker log-prob rises in lockstep
  with the source implant under positive-only training (one shared unconditional shift)
  while contrastive training simultaneously pushes bystanders down, and (2) whether
  the contrastive regime's source-side implantation advantage at matched dose is visible
  from the first steps of training. Dose is read off the step axis of one full-length
  trajectory per arm — no separate low-dose vs full-dose arms.
relates_to:
- implant-learning-speed
- leak-contrastive-negatives
---
# Training dynamics of source implantation vs bystander leakage: positive-only vs contrastive at matched recipe, full bystander panel probed per checkpoint

## Goal

Measure the training-step dynamics of source marker implantation AND bystander marker leakage — the full bystander panel probed at the same eval intervals as the source, under the four-float storage contract (log P(marker), z_marker, z_eos, logZ; trained AND base per slot) — in positive-only vs contrastive training at an otherwise matched recipe, to determine (1) whether bystander marker log-prob rises in lockstep with the source implant under positive-only training (one shared unconditional shift) while contrastive training simultaneously pushes bystanders down, and (2) whether the contrastive regime's source-side implantation advantage at matched dose is visible from the first steps of training. Dose is read off the step axis of one full-length trajectory per arm — no separate low-dose vs full-dose arms.

## Motivation

Endpoint reads establish that (a) positive-only marker training at full dose implants the source AND leaks near-uniformly to all bystanders (#18: 92-98% bystander emission), (b) contrastive training at full dose localizes (#247/#329: source ~99.6%, bystander ~11.7%), and (c) at matched tiny dose, contrastive cells reach ~8-20 nats of source implant where positive-only reaches ~2 (#472 line). But every positive-only number is endpoint-only — no per-step trajectory exists for any positive-only run — and bystander dynamics during training are missing in ALL regimes (the band-stop callback probes only the source plus a gating read). So two mechanism questions are open:

1. **Lockstep vs suppression.** Under positive-only training, does bystander log P(marker) rise with the same time course as the source (consistent with one unconditional marker direction being learned), or lag it? Under contrastive training, is bystander log-prob actively pushed DOWN over steps (negatives doing visible work), flat, or rising more slowly?
2. **Source-side acceleration.** The contrastive advantage on the source itself at matched dose is currently an endpoint contrast. Trajectories show whether positive-only is moving along the same ramp more slowly, or toward a different solution from step ~5.

## Design sketch (planner refines)

Single manipulated variable: training-mix regime (contrastive vs positive-only). Everything else matched to the #480 band-stop recipe: marker ` ※` id 83399 (in-process assert in the trainer), marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)`), lr 5e-6, r16/α32 attn-only rsLoRA, same sources, same positive rows (same questions + frozen base-model on-policy responses), seed 42.

**Arm A — contrastive (NO new training; post-hoc eval over existing checkpoints).** Reuse `superkaiba1/explore-persona-space` `adapters/issue_480_band_stop/<source>_seed42_capend/checkpoint-{20..528 step 20}` — 27 checkpoints × 6 sources (villain, comedian, kindergarten_teacher, software_engineer, assistant, qwen_default), verified present on HF 2026-06-11. For each checkpoint, probe the FULL bystander panel (every persona in the #480 eval panel) + the source at the post-response slot, teacher-forced on fixed per-persona probe rows (base-model on-policy responses, disjoint from training questions), persisting all four floats per slot per model side from the same HF forward pass.

**Arm B — positive-only (one fresh training run per source).** Identical positives, negative rows removed (not replaced); `marker_band_stop=False` (we deliberately want the full ramp through emission onset toward saturation); train by optimizer steps with checkpoints saved every 20 steps to ~528 steps (matching Arm A's grid); same per-checkpoint full-panel probe. No prior positive-only run logged dynamics or persisted per-step checkpoints (#18, #247 endpoint-only), so fresh training is required here.

**Probes and DVs.** Primary DV: log P(marker) trained − base per (checkpoint, persona). Secondary mechanistic readout: Δz_marker and the EOS margin Δ(z_marker − z_eos) from the same forward pass — required wherever log P saturates (expected late in both arms on the source; space divergence is the saturation signature, read the logit there). Sanity anchor: on-policy emission rate at a sparse subset of checkpoints (e.g. every 100 steps) on source + 2-3 bystanders, max_new_tokens ≥ 2048. Teacher-forced reads are valid here as within-condition dynamics trajectories (per marker-leakage-measurement.md); the on-policy emission reads anchor them behaviorally.

**Analysis.** Per arm: source and bystander Δlog P trajectories on a shared step axis; bystander-vs-source phase plot (bystander Δ against source Δ across checkpoints) to read lockstep (slope ~shared) vs suppression (negative movement) directly. Because the arms differ in rows-per-step by construction (positive-only batches contain no negative rows), report trajectories against BOTH optimizer steps AND cumulative positive examples seen; the latter is the fair dose axis for the source-side-acceleration question.

## Constraints, exemptions, gates

- **Contrastive-negatives exemption (a):** the single manipulated variable IS contrastive-vs-non-contrastive; the positive-only arm is the deliberate control.
- **#534 smoke-gate (hard, before any sweep):** the post-hoc eval path must reproduce #480's in-loop band-stop source read within ~1 nat on a smoke cell (reference values in `eval_results/issue_480/band-stopped-anchor-rerun/trajectories/*.json`) — a ΔG≈0 off-line read of a trained source is an adapter-application bug, not a finding.
- **Gauge assert** before any logit readout: adapter `target_modules` exclude `lm_head`/`embed_tokens`, `modules_to_save` empty.
- Capture in HF forward passes (vLLM returns post-softmax log-probs only; logits unrecoverable post-hoc). Reuse the existing capture paths: `src/explore_persona_space/eval/marker_logprob.py`, `experiments/contrastive_neg_geometry_472/eval_trajectory.py` (Phase B), `eval/callbacks.py`.
- If GPU budget binds, drop sources (keep ≥3 incl. villain + one mid-prior + qwen_default) before coarsening the checkpoint grid; the 20-step grid is the resolution the question needs (known install transitions are ~12 steps wide).

## Reused artifacts (fitness-checked at plan time)

- #480 capend checkpoints (HF model repo, listed above) — same base model, marker-only loss, lr 5e-6, r16/α32, 1:1 negatives; trained THROUGH the band rather than stopped (capend = full 528-step run), which is exactly the full-ramp object this question needs.
- #480 in-loop trajectory JSONs (`eval_results/issue_480/band-stopped-anchor-rerun/trajectories/`) as smoke-gate references and as the source-side trajectory cross-check for Arm A.
- #480 training-mix JSONLs + probe question sets from the HF data repo for Arm B positives and probe-row construction (planner verifies presence and disjointness of train vs probe questions).

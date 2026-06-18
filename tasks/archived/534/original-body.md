---
title: 'Add sub-step checkpointing to MarkerBandStopCallback and retrain #530''s 10
  cells to close the planned 4-fraction trajectory'
kind: experiment
tags: []
created_at: '2026-06-09T16:51:00Z'
has_clean_result: false
parent_id: 530
relates_to:
- leak-contrastive-negatives
- leak-predictor
goal: 'Close #530''s planned 4-fraction band-stop trajectory by modifying MarkerBandStopCallback
  to checkpoint at sub-step granularity (post-hoc fraction selection {0.25, 0.50,
  0.75, 1.00} of the realized stop step), retraining all 10 #530 cells with the otherwise-identical
  recipe, then re-running the 4-fraction eval + 6-predictor partial-Spearman refit
  to test whether the sign reversal of shadow_angle (rho=-0.23) and d_nn (rho=+0.14)
  holds across the full trajectory rather than only at the band-stop final checkpoint.'
---
# #530 follow-up — add sub-step checkpointing to the band-stop callback, retrain the 10 cells, and close the planned 4-fraction trajectory

## Goal

Close #530's planned 4-fraction band-stop trajectory by modifying MarkerBandStopCallback to checkpoint at sub-step granularity (post-hoc fraction selection {0.25, 0.50, 0.75, 1.00} of the realized stop step), retraining all 10 #530 cells with the otherwise-identical recipe, then re-running the 4-fraction eval + 6-predictor partial-Spearman refit to test whether the sign reversal of shadow_angle (rho=-0.23) and d_nn (rho=+0.14) holds across the full trajectory rather than only at the band-stop final checkpoint.

## Premise correction (why the original version of this task was blocked)

This task originally planned to re-eval "already-uploaded intermediate-fraction adapters" from #530. Those adapters do not exist: the band-stop halted training at the FIRST eval boundary (step 20), so fractions 0.25/0.50/0.75 were never reached, and the HF Hub listing for `adapters/issue_530/` contains only the top-level adapter + `ckpt_frac1.00/` + `checkpoint-20/` per cell (verified via `huggingface_hub.list_repo_files`, 2026-06-09). #530's reproducibility bullet has been corrected accordingly. Closing the trajectory therefore requires a retrain with finer checkpointing — that is this task.

## Motivation

#530's clean-result lands at MODERATE confidence with a load-bearing scope-shrinkage caveat: the planned 4-fraction trajectory collapsed to n_checkpoints_per_cell=1 because the band-stop fired at its first eval boundary. The single-checkpoint read pinned the headline sign reversal (shadow_angle ρ=−0.23, d_nn ρ=+0.14) to one point on the training trajectory.

Two outcomes are interesting:

- **All four fractions show the reversed signs** → the n_checkpoints_per_cell=1 caveat dissolves and #530's claim sharpens toward HIGH confidence.
- **Only the final fraction shows the reversal, earlier fractions show the parent's direction (or null)** → the sign would then be confounded with implant maturity, not solely with anchor saturation, and the "saturation artifact" interpretation weakens.

## What changes from parent (#530)

Single substantive variable: checkpoint granularity along the band-stop trajectory. Concretely:

1. **Callback change:** extend `MarkerBandStopCallback` (in `src/explore_persona_space/train/sft.py` / `train_lora`) to save adapter snapshots at fine step granularity (e.g. every training step, given the realized stop step in #530 was ~20), then post-hoc select the checkpoints nearest fractions {0.25, 0.50, 0.75, 1.00} of the realized stop step per cell. The stop step is not known in advance, so fraction selection must be post-hoc against the realized stop, not pre-scheduled. The band-stop semantics themselves (stop when source log P − base ∈ [5, 12] nat, gated on bystander resolution, teacher-forced in-loop source read) are unchanged.
2. **Retrain all 10 cells** (5 arms × 2 seeds, `c504v3_{near,mid_near,mid_far,far,default_only}_seed{42,137}`) with #530's recipe otherwise verbatim: marker ` ※` id 83399, marker-only loss (`MarkerOnlyDataCollator(tail_tokens=0)` + `suppress_at_post_response_slot=True`), lr 5e-6, same LoRA config, same 1:1 contrastive-negative mixes (reuse the per-cell `train_pool.jsonl` already on the HF data repo under `issue530_desat_rerun/`), same max-epoch ceiling 12, band-stop ON.
3. **Eval as originally planned:** same 54 held-out probes × 10 content-neutral framings = 540 probe-question rows per cell, teacher-forced log-prob DV, per-fraction bystander-resolution gate (median bystander log-prob ≤ −2 nats AND <60% probes at argmax = marker), 6-predictor partial-Spearman with Holm correction per fraction.

Note: per-cell training data, probe pools, predictors, eval rig, and analysis scripts are all inherited verbatim — this is deliberately a single-variable change (checkpoint granularity), with the retrain forced only because the original checkpoints were never saved. Seeds identical to #530, so the retrained trajectories should reproduce #530's final-checkpoint behavior up to run-to-run nondeterminism; the final-fraction refit doubles as a replication check of #530's reversal.

## Cost estimate

~20 GPU-h on 4× H100 (`ft-7b` intent), mirroring #530's actuals (~18 GPU-h for 10 cells trained as 2 waves of 5 via `+gpu_id=N`, training halting ~step 20 per cell) plus the 40 eval passes (10 cells × 4 fractions, ~1.5 GPU-h batched) and per-step checkpoint I/O overhead. Well under the 100 GPU-h auto-approve cap.

## Reuse footprint

Inherits #530's scripts verbatim:

- `scripts/i504_run_cell.py` (per-cell training driver) — needs only the checkpoint-granularity plumbing.
- `scripts/i504_eval_trajectory.py` (the eval probe — already iterates over a fraction list when given one).
- `scripts/i530_phase_analyze.py` / `scripts/i504_phase_analyze.py` (the partial-Spearman refit).
- `scripts/i530_emit_bystander_resolution.py` (the bystander gate per cell × fraction).
- `scripts/issue530_make_figures.py` (the 3 figures; re-emit per-fraction or fraction-overlay).
- Training mixes: per-cell `train_pool.jsonl` from `superkaiba1/explore-persona-space-data @ issue530_desat_rerun/` (verified present).

New code: the `MarkerBandStopCallback` sub-step checkpointing extension + post-hoc fraction selection, and a thin orchestrator that stitches per-fraction results into a trajectory JSON mirroring #530's existing `trajectory.json` shape.

## Acceptance criteria

1. `MarkerBandStopCallback` saves per-step (or every-k-step, k small enough to resolve a stop at ~step 20) adapter checkpoints, and post-hoc fraction selection yields 4 distinct checkpoints per cell at the realized-stop fractions {0.25, 0.50, 0.75, 1.00}.
2. All 10 cells retrain to band-stop with bystander-resolution JSON per cell × fraction; all 40 selected checkpoints uploaded to HF before pod termination.
3. Partial-Spearman refit per fraction; sign + magnitude of `shadow_angle` and `d_nn` reported across the trajectory; final-fraction refit compared against #530's values as a replication check.
4. A trajectory figure (4-point per-cell line for each predictor's partial ρ) emitted under `figures/issue_534/`.
5. Clean-result body reaches one of three honest verdicts:
   - "#530's reversed signs hold across all 4 fractions" (sharpens #530 to HIGH).
   - "Reversal only at the final fraction" (weakens #530's interpretation).
   - "Mixed across fractions, no monotone direction" (puts the headline geometry signal in a new noise band).

## Lineage

Originally auto-spawned by #530's follow-up-proposer on 2026-06-09 with a false artifact-reuse premise (see Premise correction); blocked at clarify time, then rescoped to the retrain-with-sub-step-checkpointing design on user direction (2026-06-09).

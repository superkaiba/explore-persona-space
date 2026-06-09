---
title: 'Re-eval the 4 uploaded intermediate-fraction adapters to close #530''s planned
  4-fraction trajectory'
kind: experiment
tags: []
created_at: '2026-06-09T16:51:00Z'
has_clean_result: false
parent_id: 530
---
---
kind: experiment
parent_id: 530
goal: "Close #530's planned 4-fraction band-stop trajectory by re-evaluating the 4 already-uploaded intermediate-fraction adapters (frac=0.25, 0.50, 0.75 plus the final 1.00) across all 10 cells, then refit the 6-predictor partial-Spearman to verify whether the sign reversal of shadow_angle (now ρ=−0.23) and d_nn (now ρ=+0.14) holds across the full trajectory, not just at the single band-stop final checkpoint that #530's analysis pinned."
auto_spawned_by_parent: 530
---

# #530 follow-up — re-eval 4 uploaded intermediate-fraction adapters to close the planned 4-fraction trajectory

## Goal

Close #530's planned 4-fraction band-stop trajectory by re-evaluating the 4 already-uploaded intermediate-fraction adapters (frac=0.25, 0.50, 0.75 plus the final 1.00) across all 10 cells, then refit the 6-predictor partial-Spearman to verify whether the sign reversal of shadow_angle (now ρ=−0.23) and d_nn (now ρ=+0.14) holds across the full trajectory, not just at the single band-stop final checkpoint that #530's analysis pinned.

## Motivation

#530's clean-result body lands at MODERATE confidence with a load-bearing scope-shrinkage caveat: the MarkerBandStopCallback fired at the first eval boundary (training_step=20), collapsing the planned 4-fraction trajectory to n_checkpoints_per_cell=1. The intermediate-fraction adapters were uploaded (`superkaiba1/explore-persona-space @ adapters/issue_530/<slug>/ckpt_frac{0.25,0.50,0.75}/`) but not consumed by the analysis. #530's Reproducibility (line 145) explicitly names this re-eval as the load-bearing missing read.

Two outcomes are interesting:

- **All four fractions show the reversed signs** → the n_checkpoints_per_cell=1 caveat dissolves and #530's claim sharpens toward HIGH confidence.
- **Only the final fraction shows the reversal, earlier fractions show the parent's direction (or null)** → the sign would then be confounded with implant maturity, not solely with anchor saturation, and the "saturation artifact" interpretation weakens.

## What changes from parent (#530)

Single variable: the analysis input set changes from `{final}` to `{frac=0.25, 0.50, 0.75, 1.00}` per cell. Everything else inherited verbatim from #530's plan v1:

- Same 10 cells (5 arms × 2 seeds).
- Same 54 held-out probes × 10 content-neutral question framings = 540 probe-question rows per cell.
- Same teacher-forced log-prob DV (no generation).
- Same 6-predictor partial-Spearman with Holm correction.
- Same bystander-resolution gate (median bystander log-prob marker ≤ −2 nats AND <60% probes at argmax = marker) applied per fraction.
- Same vLLM eval rig.

## Cost estimate

~1.5 GPU-h on 1× H100 `eval` intent: 10 cells × 4 fractions = 40 eval passes × 540 probe-question rows each, batched. Plus ~0.3 CPU-h equivalent for the partial-Spearman refits.

## Reuse footprint

Inherits #530's scripts verbatim:

- `scripts/i504_eval_trajectory.py` (the eval probe — already iterates over a fraction list when given one).
- `scripts/i530_phase_analyze.py` (the partial-Spearman refit).
- `scripts/i530_emit_bystander_resolution.py` (the bystander gate per cell × fraction).
- `scripts/issue530_make_figures.py` (the 3 figures; re-emit per-fraction or fraction-overlay).

The single new code item is a thin orchestrator that iterates over `{0.25, 0.50, 0.75, 1.00}` per cell and stitches results into a per-fraction trajectory JSON, mirroring the existing `trajectory.json` shape #530 already writes per cell.

## Acceptance criteria

1. All 10 cells × 4 fractions = 40 eval passes complete with bystander-resolution JSON per cell × fraction.
2. Partial-Spearman refit per fraction; sign + magnitude of `shadow_angle` and `d_nn` reported across the trajectory.
3. A trajectory figure (4-point per-cell line for each predictor's partial ρ) emitted under `figures/issue_<this-task>/`.
4. Clean-result body reaches one of three honest verdicts:
   - "#530's reversed signs hold across all 4 fractions" (sharpens #530 to HIGH).
   - "Reversal only at the final fraction" (weakens #530's interpretation).
   - "Mixed across fractions, no monotone direction" (puts the headline geometry signal in a new noise band).

## Auto-spawn lineage

This task was auto-spawned by #530's follow-up-proposer at /issue Step 9b on 2026-06-09, marked `auto_run: yes`. Spawned in autonomous mode because the diff is mechanical, the cost is ≤4 GPU-h, the input artifacts are already uploaded, and the result directly addresses #530's load-bearing single-checkpoint caveat.

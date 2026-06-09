---
title: Push the marker implant past emission onset to test whether per-context singleton
  structure emerges with stronger training
kind: experiment
tags: []
created_at: '2026-06-09T19:44:11Z'
has_clean_result: false
parent_id: 527
goal: Test whether pushing the marker implant past emission onset (source log P(marker)
  − base in the [14, 20] nat window where on-policy emission begins) makes the per-context
  singleton shifts develop effective rank ≥ 2 across held-out contexts, so that the
  additivity-cosine read becomes a diagnostic superposition test rather than a measurement
  of constant-direction steering.
relates_to:
- leak-single-vs-multi
- leak-predictor
- leak-from-cell-set
---
# Push the marker implant past emission onset to test whether per-context singleton structure emerges with stronger training

## Goal

Test whether pushing the marker implant past emission onset (source log P(marker) − base in the [14, 20] nat window where on-policy emission begins) makes the per-context singleton shifts develop effective rank ≥ 2 across held-out contexts, so that the additivity-cosine read becomes a diagnostic superposition test rather than a measurement of constant-direction steering.

## Background — what #527 found and what's unread

Task #527 (parent) tested the rank-one-map-plus-beacons additivity pillar with both validity fixes the parent #520 missed: a properly band-stopped marker implant (source `log P(marker) − base` inside [5, 12] nat) and orthogonal source pairs (base-model L20 centered cosine ≈ 0). Both fixes landed mechanically — band-stop fired at 5.00-7.47 nats source delta across all 18 cells at step 30-40; realized cos(florist, medical_doctor) = +0.001 and cos(librarian, police_officer) = −0.004 at L20.

Yet the implant geometry collapsed to near-rank-1 anyway: GD1 (joint per-context shift matrix top-1 SV 0.87, effective rank 1.3), GD2 (singleton cosine 0.90), AND GD3 (per-singleton effective rank 1.24-1.38) all fail uniformly on every cell. `fraction_dv1_diagnostic = 0.0`. The headline DV1 cosine of 0.99 is mechanical — grades parallel-vector arithmetic, not per-context superposition.

The marker-training-recipe rule predicts emission onset (the "firing cliff" where log P(marker) overtakes EOS at the end slot) ~step 60-100, AFTER the band-stop fired at step 30-40 in #527. So the parent's measurement window was **below** emission onset, where the model satisfies marker-only loss with a constant steering direction. That is consistent with the rank-1 collapse — at low-strength early-ramp dial, the model has no reason to differentiate which contexts emit the marker.

## Hypothesis

The rank-1 collapse of singleton shifts at the [5, 7.5] nat band-stop is a property of the early ramp regime — the model satisfies marker-only loss with a constant steering direction at low log-prob delta but, as training pushes log P(marker) past the EOS crossover (~step 60-100 per `.claude/rules/marker-training-recipe.md`), it has to differentiate which contexts emit the marker, so the per-context structure of singleton shifts grows. Singleton effective rank rises above 2.0 (GD3 passes) AND the joint matrix top-1 SV share drops below 0.75 (GD1 passes). If both gates pass at the harder dial point, the additivity-cosine DV1 read becomes structurally diagnostic.

## Falsification criterion

Singleton effective rank stays ≤ 2.0 at the harder dial point (GD3 still fails uniformly), AND the joint matrix top-1 SV share stays > 0.75 (GD1 still fails). That would mean the rank-1 attractor is a property of the marker-only loss objective itself, not of training depth — and the additivity-cosine construct cannot be rescued by training harder.

## Setup

Inherits #527's full pipeline EXCEPT the band-stop window. All else identical.

- **Base model:** Qwen/Qwen2.5-7B-Instruct (same as #527)
- **Marker:** ` ※` token id 83399 (assert encode == [83399]) (same)
- **Source pairs:** florist × medical_doctor (cos +0.001), librarian × police_officer (cos −0.004) — both inherited from #527's pair selection (same orthogonal cells)
- **Negative panel:** 4 personas — default_assistant + librarian + programmer + chef — strict 1:1 positives-to-total-negatives (same)
- **Seeds:** {42, 137, 256} (same)
- **Recipe:** rsLoRA r=16 / α=32, attn-only (q/k/v/o), lr=5e-6 cosine warmup 0.03, MarkerOnlyDataCollator(tail_tokens=0, suppress_at_post_response_slot=True), 1:1 positives-to-total-negatives (all same)
- **The one variable that changes:** band-stop window `[5, 12]` → `[14, 20]` nat. `marker_band_low_nats=14, marker_band_high_nats=20`. Epochs cap raised from 8 to 24 (the real stop is still the band-stop firing; expect step 80-150 per the recipe).
- **Phase A anchor smoke** gates bystander resolution at the NEW band before the Phase B sweep launches (per `.claude/rules/marker-training-recipe.md`: gate on bystander resolution, not source emission).

## Eval (unchanged from #527)

19 held-out personas × 20 fixed questions × 1 greedy sample per row = 380 measurements per cell. Same DV1/DV2/DV3 + DV4 + GD1/GD2/GD3 stack at L20 post-response slot. Same vLLM batched on-policy generation.

## Success criterion

Singleton effective rank ≥ 2.0 (GD3 passes) AND joint matrix top-1 SV share ≤ 0.75 (GD1 passes) on ≥ 4 of 6 cells. If both gates pass, read DV1 as the determinative superposition test — pass or fail.

## Kill criterion

GD3 still fails uniformly at the harder dial point (singleton eff rank ≤ 2.0 on ≥ 5 of 6 cells). That kills the rank-one-map-plus-beacons additivity pillar for marker-only loss: the rank-1 attractor is a loss-objective artifact, not a training-depth artifact. Next pivot is necessarily a different training objective (Proposal 3 in #527's follow-ups: whole-completion loss).

## Compute

~14 GPU-hours total on 1× H100 pod (intent `lora-7b`). Parent #527 landed at 12 GPU-h with stop step 30-40; this run stops at ~80-150 steps per cell, so training time scales ~3-4× the parent's training portion (parent ~7.5h sweep → ~11h here) + Phase A smoke (~1h) + eval/extract/analysis (~3h unchanged) ≈ 14 GPU-h.

Estimated GPU-hours (total): 14

## Pod preference

1× H100, intent `lora-7b`. Same as #527.

## References

- Parent: #527
- Grandparent: #520
- Marker training recipe: `.claude/rules/marker-training-recipe.md` § "Don't fix epochs — stop on the log-prob band" + § "emission onset ≠ saturation"
- Marker leakage measurement: `.claude/rules/marker-leakage-measurement.md`
- Contrastive negatives: `.claude/rules/contrastive-negatives.md`

---
title: 'Pre-implant null-geometry calibration: bound chance geometry-shaped partial
  correlations at step-5 no-implant snapshots across 5 fresh seed pairs'
kind: experiment
tags: []
created_at: '2026-06-10T10:01:02Z'
has_clean_result: false
parent_id: 534
goal: Bound how often geometry-shaped partial-correlation readings arise with no implant
  by retraining the four positioned arms for only the first five optimizer steps under
  five fresh seed pairs and fitting the same six-predictor regression at each no-implant
  snapshot, to test whether the pre-implant nearest-negative reading (+0.110 at step
  5) is chance-level noise or a systematic warmup artifact that confounds the stop-point
  bubble reversal.
---
## Goal

Bound how often geometry-shaped partial-correlation readings arise with no implant by retraining the four positioned arms for only the first five optimizer steps under five fresh seed pairs and fitting the same six-predictor regression at each no-implant snapshot, to test whether the pre-implant nearest-negative reading (+0.110 at step 5) is chance-level noise or a systematic warmup artifact that confounds the stop-point bubble reversal.


### 2. Pre-implant null-geometry calibration across fresh seeds — Type: Diagnostic

**Parent:** #534
**Goal:** Bound how often geometry-shaped partial-correlation readings arise with no implant by retraining the four positioned arms for only the first five optimizer steps under five fresh seed pairs and fitting the same six-predictor regression at each no-implant snapshot, to test whether the pre-implant nearest-negative reading (+0.110 at step 5) is chance-level noise or a systematic warmup artifact that confounds the stop-point bubble reversal.
**Hypothesis:** Sub-floor fits are noise: across 5 fresh no-implant replicates the nearest-negative partial rho scatters around zero with spread wide enough that +0.110 is unremarkable (0-1 of 5 replicates Holm-significant, no consistent sign). The parent's own step-5 to step-10 sign flip (+0.110 to -0.023) predicts this.
**Falsification:** A consistent positive nearest-negative sign in >= 4 of 5 replicates (or a pooled interval excluding zero) kills the noise hypothesis — warmup/init perturbations systematically organize along nearest-negative distance, the stop-point +0.125 inherits a real confound, and the bubble half of the lineage's headline must be re-derived with a pre-implant baseline subtracted. Shadow-angle doubles as the specificity control (parent read it flat-null pre-implant: +0.046/+0.012), so a geometry-generic artifact would also show there.
**Differs from parent:** Exactly ONE variable — the seed set (5 fresh seed pairs replacing 42/137). The 5-step training cap is a read-point truncation, not a recipe change: training is sequential, so the step-5 weights are identical whether the run stops at step 5 or continues to the band-stop; only the {0.25} read of the parent's fraction set is evaluated.

**Pre-filled spec (from parent):**
- Model: same as parent (`Qwen/Qwen2.5-7B-Instruct`, LoRA r=8/alpha=32/dropout 0.05/all-linear/rsLoRA, lr 5e-6, cosine + warmup 0.05, AdamW bf16, batch 4 x grad-accum 4, marker ` ※` id 83399, marker-only collator)
- Data: same construction as parent — train pools REBUILT per fresh seed via `build_cell_504` from the Hub-verified #472 R pools (the HF `issue530_desat_rerun/train_pools/` bytes exist only for seeds 42/137 and are NOT reused here; rebuilding per seed is what the seed variable means)
- Seeds: 5 fresh pairs, e.g. {7,11}, {19,23}, {71,73}, {101,103}, {211,223} — 4 positioned arms x 10 seeds = 40 cells; default-only cells skipped (excluded from the regression by construction)
- Eval: same as parent — 54 held-out probes x 10 framings, teacher-forced log P(marker) + z_marker at the post-response slot, evaluated at the step-5 snapshot only; same 6-predictor partial Spearman + Holm per replicate (432 rows each, same n as the parent fit)
- Config: same as parent EXCEPT: seed set (max_steps capped at 5 as the no-op cost optimization above; band-stop never reachable before min_steps=20 so its removal below the cap changes nothing)

**Estimated cost:** ~9 GPU-hours on `ft-7b` (4x H100, waves of 4; ~0.22 GPU-h/cell = 5 training steps + one 540-row eval, grounded in #534 actuals of 12.5 GPU-h for 10 cells x 20-step training x 4 evals)
**If it works:** (noise confirmed) The pre-implant confound on the bubble predictor relaxes to "low power below the floor"; the demotion stands but stops threatening the stop-point read, and the lineage can spend its next GPU on the anchor question instead of the confound.
**If it fails:** (systematic artifact) That is the bigger finding: the partialling manufactures geometry-shaped readings with nothing implanted — every partial-correlation claim in the #504/#530/#534 lineage then needs a pre-implant baseline subtraction, and proposal 3 / the mid-anchor inherit that correction before launch.

**auto_run:** yes
**auto_run_reason:** Clean single-variable diagnostic of a validity defect named in the parent body (the pre-implant nearest-negative confound), recipe inherited verbatim with zero new knobs, cost grounded in #534 actuals and far under the cap, no design-taste fork (seed pairs + decision rule pre-specified above), premise artifacts Hub-verified for this proposal.

**cost_class:** needs-gpu
**headline_affecting:** no

---

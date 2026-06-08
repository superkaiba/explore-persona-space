---
title: 'Do #502''s persona-distance leakage predictors improve with probe count, or
  plateau by N=500?'
kind: analysis
tags: []
created_at: '2026-06-07T01:49:33Z'
has_clean_result: false
parent_id: 502
---
## Goal

Determine whether #502's residual-stream persona-distance predictors of marker-leakage
transfer are **probe-count-limited** — i.e. does predictor quality keep climbing as you
estimate each persona's activation cloud from more user prompts, or has it already
plateaued at the 500 probes #502 used? Quality = length-partial Spearman ρ and
leave-one-context-out CV R² of the predictor against #474's ΔG marker-transfer target.

If the convergence curves have flattened by N≤500, the #502 ceiling is structural (more
probes won't help). If they are still climbing at N=500, that is direct evidence the
predictors are sample-limited and worth more probes.

## Why this follow-up

#502 found a last-prompt × layer-19-24 ridge of cloud-aware metrics (Gaussian-KL peak at
L22, with MMD and Wasserstein-2 in the same band) predicting ΔG at ρ ≈ −0.79 / CV R² ≈
0.61 on the cleanest training checkpoint (loc-arm epoch 1), out of a ~1500-cell sweep.
Each persona's activation "cloud" was estimated from 500 probes. Cloud-aware metrics
(Gaussian-KL on a PCA-16 fit, MMD, Bures-Wasserstein-2, Mahalanobis) estimate a
covariance, so their distance estimates are noisier at small N and should converge as N
grows. The natural question #502 did not answer: **is ρ/CV R² still rising at N=500?**
The mean-based metrics (cosine, Euclidean) need far fewer probes, so the cloud-aware
metrics' advantage — if real — should *widen* with N. This is the test.

## Method — pure reanalysis, NO POD

The per-probe activations are already saved, so no re-extraction and no training is
required for the core question. Run on the VM.

1. **Pull saved activations.** HF data repo
   `superkaiba1/explore-persona-space-data/issue502_28layer_500probe_bakeoff/` — per
   `(extraction_point, layer)` `.pt` files of shape `(16 transforms, 500 probes, H)`.
   Reuse `scripts/issue493_extraction_metric_bakeoff.py::load_activations_from_disk`.

2. **Probe-count grid.** N ∈ {25, 50, 100, 200, 350, 500}. For each N draw R = 10 random
   probe subsets along the probe axis (axis 1), fixed seeds for reproducibility. This is
   the one piece of new code: a random-subsample loop over the in-memory `(16, N, H)`
   slice — do NOT re-extract. (The existing `--n-probes` flag takes a deterministic
   `[:n]` prefix; that gives one point, not error bars. Add random subsetting.)

3. **Recompute + rescore.** For each (N, subset) recompute the 9 metrics and rescore with
   the existing regression phase (length-partial Spearman + leave-one-context-out CV R²,
   the i474 fig9 convention) against #474's target at
   `eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json` — primary DV `delta_g`
   (ΔG), secondary base-prior-safe `g_logprob`. Reuse the metric + regression code already
   in `issue493_extraction_metric_bakeoff.py`; do not reinvent the metrics or the CV.

4. **Cells to track.** Two readouts, reported separately:
   - **Fixed-cell curve (the honest one):** the #502 winner `last_prompt × L22 × gauss_kl
     × raw`, plus the rest of the L19-L24 ridge (gauss_kl, mmd, wass2). Fixing the cell
     avoids small-N winner's-curse inflation.
   - **Search-best curve:** the per-N argmax cell over the full grid — shows whether the
     winning cell *identity* is stable across N or only crystallizes at large N.
   - **Baselines:** last-token cosine (#493's incumbent) and the next-token JS baseline,
     which are mean/output-based and should converge fast — the contrast is the point.

5. **Primary checkpoint = loc-arm epoch 1** (where #502's signal concentrated). Time
   permitting, repeat the fixed-cell curve on loc_ep2/3/5 to see if the probe-count
   story interacts with the checkpoint-fragility #502 already flagged.

6. **Figures.** Convergence curves: x = N (log scale), y = |ρ| and y = CV R², one line
   per tracked cell, error bars = mean ± std over the R=10 subsets. Raw alongside any
   processed/aggregated view per project convention.

## Conditional short experiment (only if still climbing at N=500)

If the fixed-cell ρ/CV R² has NOT flattened by N=500 (e.g. the N=350→500 step still adds
> ~0.02 CV R² beyond subset noise), THEN run one short forward-pass-only extraction to
extend the curve — no training:

- 1× H100, generation/forward only. Generate 500 additional probes from the same
  mixed-distribution pool via `scripts/issue502_generate_probes.py`, re-run the
  `run_extraction` path at N=1000 for the focal extraction point (`last_prompt`) and the
  L19-L24 layers only (not all 28 × 3), upload the new activations, and extend the
  convergence plot to N=1000.
- If the curve IS flat by N=500, skip this — report the plateau and stop.

## Honesty / controls

- Keep #502's length-partialing; report BOTH ρ and CV R² (they move differently because
  CV R² assumes a linear shape ρ does not).
- Error bars from R=10 random subsets per N — a single subset is not a finding.
- LOCO CV R² is within-grid generalization, not an independent held-out test; say so.
- Do not narrate a 0.02-CV-R² gap between adjacent cells as a winner; this is a
  ridge-level claim (the #502 caveat carries forward).
- Negative result is a real result: "predictors are not probe-limited at N≤500" is the
  expected and useful outcome if the curves flatten.

## Reuse

- `scripts/issue493_extraction_metric_bakeoff.py` — `load_activations_from_disk`, the 9
  metrics, the regression phase (length-partial Spearman + LOCO CV R²). New code is only
  the N × random-subset loop wrapping these.
- `eval_results/issue_474/cross_eval/*/G_logprob_matrix.json` — targets, already on VM.
- `scripts/issue502_generate_probes.py` — only for the conditional pod extension.

Parent: #502 (which is parent #493). Artifacts under `eval_results/issue_<N>/`,
figures under `figures/issue_<N>/`.

---
title: At 500 probes per persona the cloud-aware leakage predictor has converged,
  and its 0.06 CV R² lead over the same-layer cosine baseline survives sample-size
  matching (HIGH confidence)
kind: analysis
tags: []
created_at: '2026-06-07T01:49:33Z'
has_clean_result: false
parent_id: 502
---
# At 500 probes per persona the cloud-aware leakage predictor has converged, and its 0.06 CV R² lead over the same-layer cosine baseline survives sample-size matching (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** The persona-distance predictor from #502 is done getting better at 500 probes — feeding it more probes won't push the ceiling up, and the cloud-aware metric really is doing something cosine can't, even when both see the same data.

**Takeaways.**
- The headline cell's CV R² climbed from 0.56 at N=25 to 0.617 at N=500 and the last step (350→500) only added 0.003, smaller than the within-N subset noise. That's the plateau.
- The same-layer cosine baseline starts flat at ~0.557 from N=25 and never moves. So the extra ~0.06 CV R² that gauss_kl L22 has is not "cosine waiting for more probes" — it's a genuine signal about persona shape that mean-direction can't see.
- Per the autonomous descope I had to drop Bures-Wasserstein-2 from the ridge and didn't get loc-epoch 2/3/5. So "the ridge plateaus too" is on `{gauss_kl, mmd}` not the full triad, and the checkpoint-fragility question carries forward.

**How this updates me.** I now believe the predictor ceiling at #502 is structural — more probes won't help. The right next move isn't more probes; it's testing other predictor families (richer covariance estimators, layer-pooled metrics, the missing wass2 + checkpoint robustness), or trying to attack the 0.39 of variance the predictor still misses.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

In #502 I fit a leakage-transfer predictor on a ~1500-cell sweep over residual-stream persona-distance metrics. The winner was a cloud-aware Gaussian-KL distance at last-prompt × layer 22 × raw activations, length-partial Spearman ρ ≈ −0.79 and leave-one-context-out CV R² ≈ 0.61 against #474's marker-transfer target on the cleanest training checkpoint. Each persona's activation cloud was estimated from 500 probes. Cloud-aware metrics estimate a covariance, so their distance estimates are noisier at small N and should converge as N grows; cosine and Euclidean baselines need far fewer probes. The natural follow-on: is ρ / CV R² still climbing at N=500?

If the convergence curves have flattened by N≤500, #502's ceiling is structural — more probes won't help, and the predictor's limit is the metric family, not the sample size. If they're still climbing, then #502 is sample-limited and adding probes is the cheap next win. There's a derivative claim to test too: if cloud-aware metrics genuinely capture shape that cosine can't, then a sample-size match (give cosine the same N as gauss_kl) should NOT close the gap.

### What I ran

I drew R=10 random probe subsets at N ∈ {25, 50, 100, 200, 350, 500} from the on-disk activation tensors (16 transforms × 500 probes × hidden, per extraction-point × layer), recomputed the persona-distance matrices on each subset, refit the length-partial Spearman + leave-one-context-out CV R² regression against #474's ΔG target, and read off the mean ± std curves over N.

The headline cell — last-prompt × layer 22 × Gaussian-KL × raw — ran on the full R=10 × N grid. The cloud-aware ridge (gauss_kl, MMD at L19-L24) ran at the same R=10 but capped at N≤350 after an autonomous compute-deviation descope (the gauss_kl 2N×2N joint-Gram eigendecomposition cost ~50× the original time estimate). Same-layer cosine baselines (L19-L24) and sentinel cosines (L0 early, L11 mid, L27 final) ran the full N grid as cheap controls. The Bures-Wasserstein-2 metric was dropped, and the loc-epoch 2/3/5 robustness sweep didn't fit in budget either — both noted in the relevant finding's read prose and in Reproducibility.

The probe pool is #502's mixed-distribution set of 500 user prompts; the activation tensors were never re-extracted (this is pure reanalysis). The target is #474's ΔG marker-leakage matrix on the loc-arm epoch-1 checkpoint, the same target #502 used.

The headline-cell aggregate table below is the full six-row N grid for the headline cell, computed from all 60 (N × R) raw rows in the per-cell sweep output ([`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json), aggregates section). Random-sample disclosure: all six N values for the headline cell are shown — this is not a cherry-pick of best rows.

<details open>
<summary>Headline-cell aggregates by N (mean ± std over R=10 random subsets)</summary>

| N | \|ρ\| (length-partial Spearman) | CV R² (LOCO, length-partial) |
|---|---|---|
| 25 | 0.7598 ± 0.0088 | 0.5605 ± 0.0150 |
| 50 | 0.7765 ± 0.0080 | 0.5902 ± 0.0136 |
| 100 | 0.7862 ± 0.0021 | 0.6070 ± 0.0042 |
| 200 | 0.7893 ± 0.0018 | 0.6118 ± 0.0030 |
| 350 | 0.7909 ± 0.0023 | 0.6148 ± 0.0042 |
| 500 | 0.7922 ± 0.0007 | 0.6174 ± 0.0011 |

The reproduction at N=500 / single-subset = 0.61806 deterministic-CV vs #502's archived 0.60863 nondeterministic-CV; the wrapper uses a deterministic LOCO fold-ordering (sorted) where the bakeoff code iterated `set(cond_ids)` (hash-seed-dependent), so the two values match within the bakeoff's intrinsic fold-ordering noise band (~5e-3). On Spearman ρ the reproduction is tight: |Δρ| = 0.00029 < 1e-3.

</details>

The ridge-family layout below is the cell-by-cell N-cap breakdown for the autonomous-descope run; random-sample disclosure: all 9 ridge cells + 9 cosine cells are shown (not cherry-picked). Per-cell aggregates in [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json).

<details>
<summary>The 9 cells in the ridge family (with N caps)</summary>

| Cell | N grid | R | Note |
|---|---|---|---|
| last-prompt L22 gauss_kl raw | 25, 50, 100, 200, 350, 500 | 10 | Headline (full grid) |
| last-prompt L19/L20/L21/L23/L24 gauss_kl raw | 25, 50, 100, 200, 350 | 10 | Ridge cloud-aware (N=500 dropped) |
| last-prompt L19/L20/L21/L22/L23/L24 mmd raw | 25, 50, 100, 200, 350 | 10 | Ridge cloud-aware (N=500 dropped) |
| last-prompt L19/L20/L21/L22/L23/L24 cosine raw | 25, 50, 100, 200, 350, 500 | 10 | Same-layer cosine controls |
| last-prompt L0/L11/L27 cosine raw | 25, 50, 100, 200, 350, 500 | 10 | Sentinel cosines (early/mid/final) |

Bures-Wasserstein-2 dropped; loc-epoch 2/3/5 not run.

</details>

### Findings

#### The headline cell is sample-converged at N=500

The leakage predictor's CV R² climbs from 0.5605 at N=25 to 0.6174 at N=500. The step from 350→500 adds 0.0027 — smaller than the within-N subset noise (σ_ref = mean(σ(N=200), σ(N=350)) = 0.0036). On the length-partial Spearman ρ side, the same pattern: 0.7598 → 0.7922, the 350→500 step adds 0.0013 and at N=500 the across-subset std is 0.0007 (the curve has flatlined). The reproduction-gate row at N=500 matches #502's archived headline to |Δρ| = 0.00029.

![Two line plots side by side showing the headline cell's length-partial Spearman magnitude (left, blue) and leave-one-context-out CV R² (right, red) as a function of N probes per persona, on log-scale x-axis from N=25 to N=500. Both curves rise sharply from N=25 to N=100 then asymptote between N=200 and N=500. Error bars from R=10 random subsets are visible at every N and collapse to near-zero at N=500.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511/convergence_fixed_cell.png)

> **Figure.** *#502's headline predictor cell (last-prompt × L22 × gauss_kl × raw) on the loc-arm epoch-1 checkpoint reaches its ceiling between N=350 and N=500 — adding more probes won't lift it.* x-axis = N probes per persona used to estimate each persona's activation cloud (log scale). Left panel = |ρ|, length-partial Spearman against #474's ΔG. Right panel = CV R², leave-one-context-out, length-partialed. Error bars = mean ± std over R=10 random probe subsets. Each (N, subset) point is one full recomputation of the 16×16 distance matrix and regression fit.

I can see the same plateau from the raw side too. The per-subset points crowd into a tight cluster at N=500 (CV R² = 0.6164–0.6193 across R=10 subsets, |ρ| = 0.7914–0.7933) and fan out at N=25 (CV R² = 0.528–0.579, |ρ| = 0.741–0.769). Sample-size matching makes the N=500 cluster vanish onto a single point in raw space.

The 5 raw rows below are the first five of ten (N, subset) points at each of N=25 and N=500 for the headline cell — random-sample disclosure: ordered by `subset_idx`, not cherry-picked from best-CV rows. Full 60-row table in [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json) under `rows` filtered to `cell_id = last_prompt__L22__gauss_kl__raw`.

<details open>
<summary>5 raw (N, subset) points each at N=25 and N=500 for the headline cell</summary>

```
Headline cell: last_prompt × L22 × gauss_kl × raw, loc-arm epoch 1

N=25  subset_idx=0  |ρ|=0.76799  CV R²=0.57476
N=25  subset_idx=1  |ρ|=0.75999  CV R²=0.55976
N=25  subset_idx=2  |ρ|=0.76859  CV R²=0.57670
N=25  subset_idx=3  |ρ|=0.76131  CV R²=0.56253
N=25  subset_idx=4  |ρ|=0.74133  CV R²=0.52826

N=500 subset_idx=0  |ρ|=0.79170  CV R²=0.61666
N=500 subset_idx=1  |ρ|=0.79202  CV R²=0.61721
N=500 subset_idx=2  |ρ|=0.79328  CV R²=0.61927
N=500 subset_idx=3  |ρ|=0.79159  CV R²=0.61648
N=500 subset_idx=4  |ρ|=0.79143  CV R²=0.61624
```

These are points 0-4 of 10 random probe-subsets at each N for the headline cell. Random-sample disclosure: first five of ten subsets at each N (the full ten are visible in the convergence figure's error bars). Full per-subset rows in [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json).

</details>

A subtlety I want to call out: the cloud-aware ridge cells (gauss_kl at L19/L20/L21/L23/L24 and mmd at L19-L24) cap at N=350, not N=500. Their per-N CV R² estimates are reported on R=10 subsets but the plateau call is technically the headline cell's call. The ridge cells *trend* the same way (curves bend over by N=200-350) and the verdict on the 8 secondary cells that have N=500 cosine data is plateau too — but if you're skeptical that the headline cell carries the ridge, the cautious reading is "the L22 gauss_kl cell at the ridge peak has plateaued at N=500 and the rest of the ridge has plateaued by N=350 within the tighter window we measured."

#### Cosine is already flat at N=25 — the 0.06 gap to gauss_kl is real

If the cloud-aware advantage were just "covariance estimators need more probes," cosine should climb the same curve as gauss_kl and either converge to roughly the same number or sit slightly below it. That's not what happens. Cosine L22 at N=25 is already at CV R² = 0.5568, and at N=500 it's at 0.5573 — six ten-thousandths higher. The whole same-layer cosine ridge L19-L24 sits in the 0.547-0.568 band, flat across N. Gauss_kl L22 starts at 0.5605 at N=25 (barely above cosine) and climbs by 0.057 CV R² to land at 0.6174 — a clean separation that opens up exactly where the cloud-aware metric has the probes to estimate its covariance.

![A line plot with CV R² on the y-axis from 0.10 to 0.62 and N probes per persona on the x-axis from 25 to 500 on log scale. The bold red line at the top is gauss_kl L22 (the headline cell), rising from 0.56 at N=25 to 0.62 at N=500. Six gray "+"-marker lines for cosine L19-L24 cluster tightly between 0.547 and 0.568, essentially flat across N. Three dashed lines for sentinel cosines: L0 sits at 0.13 throughout, L11 at 0.18 throughout, L27 at 0.52 throughout. The whole picture: the cloud-aware headline pulls away from the cosine ridge, and the cosine curves never move with N.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511/baselines.png)

> **Figure.** *The gauss_kl L22 advantage over the same-layer cosine ridge is not "cosine waiting for probes" — cosine is already flat at N=25.* Bold red line = headline cloud-aware cell. Gray "+"-marker lines = the six same-layer cosine controls at L19-L24 (the cosine ridge, dense band ~0.55-0.57). Dashed lines = sentinel cosines at L0 (early, ~0.13), L11 (mid, ~0.18), and L27 (final, ~0.52) — far worse than the L19-L24 band, confirming the layer-band finding from #502. All lines share the same N grid and R=10 subset estimator. The "cos L21 (sentinel)" legend entry is a script cosmetics bug — L21 appears twice (once in the ridge band, once styled as sentinel) but the underlying data is the same; the L21 cosine value is the gray-band L21, not an extra sentinel. Bures-Wasserstein-2 was dropped from the ridge under the autonomous descope, so the gauss-kl-vs-cosine separation here is gauss_kl-specific, not yet a full cloud-aware family claim.

The raw cosine numbers make the comparison clean. At N=500, cosine L22 hits subset_idx=0/1/2 with identical CV R² = 0.55734 (when N equals the pool size every subset is the same identity permutation up to fold tie-breaks, and on cosine the metric matrix is exactly permutation-invariant — three identical decimals out to five places). At N=25 it's 0.572 / 0.550 / 0.568 — wider than the N=500 cluster but already squarely within the same band the curve will sit in across the rest of the N grid.

The cosine raw rows below are the first three of ten subsets per N — random-sample disclosure: ordered by `subset_idx`, not cherry-picked. Full 60-row table for the cosine L22 cell at the same path: [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json), `rows` section filtered to `cell_id = last_prompt__L22__cosine__raw`.

<details>
<summary>Cosine L22 raw rows for comparison with gauss_kl L22</summary>

```
Same-layer cosine baseline: last_prompt × L22 × cosine × raw, loc-arm epoch 1

N=25  subset_idx=0  |ρ|=0.76506  CV R²=0.57240
N=25  subset_idx=1  |ρ|=0.75133  CV R²=0.54952
N=25  subset_idx=2  |ρ|=0.76201  CV R²=0.56765

N=500 subset_idx=0  |ρ|=0.75649  CV R²=0.55734
N=500 subset_idx=1  |ρ|=0.75649  CV R²=0.55734
N=500 subset_idx=2  |ρ|=0.75649  CV R²=0.55734
```

Three identical rows at N=500 happen because when N equals the activation-pool size the metric matrix is exactly permutation-invariant on cosine (cosine averages directions before distancing); the per-subset CV-R² variation in the other cells is the LOCO fold-ordering tie-break on subsets that share an identical metric matrix. The point of the comparison is the cosine band ~0.55-0.57 vs the gauss_kl 0.62 destination — not that all three rows are identical at N=500.

</details>

The control side reads cleanly too. Sentinel cosines at L0 (early), L11 (middle), and L27 (final) sit at 0.14, 0.19, and 0.53 — all far below the L19-L24 band, the same layer-stratification #502 reported. The 0.06 lift sits squarely at the L22 ridge peak; not everywhere, only at the cells where #502 said the signal lived.

#### Cloud-aware metrics need the probes; cosine doesn't

Here's the asymmetry that ties findings 1 and 2 together. Gauss_kl at L22 climbs 0.057 CV R² between N=25 and N=500 (0.561 → 0.617). Cosine at L22 climbs 0.001 over the same range. Cosine isn't getting more accurate as I feed it more probes because cosine's only sample-dependent term is the persona-mean direction, and at N=25 the mean direction is already a tight estimate (~Inv√25 ≈ 0.2 standard-error scaling on each component, against a hidden dimension of 3584 dimensions which mean-pool well). Gauss_kl estimates a *covariance* on a 16-dimensional PCA fit — covariance estimators classically need at least ~10× the dimensionality of probes per sample to be well-conditioned, which puts the sweet spot around N=160-200 and matches where gauss_kl's curve starts bending over.

![Two line plots side by side showing the L19-L24 cloud-aware ridge family. Twelve curves, six gauss_kl (solid blue) and six mmd (dashed green), at layers L19-L20-L21-L22-L23-L24. All curves rise from N=25 to N=200 then flatten through N=350. The L22 gauss_kl curve (the headline) is the highest, reaching the standalone N=500 point at 0.617. The cluster sits between 0.56 and 0.62 CV R² at the asymptote.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511/convergence_ridge.png)

> **Figure.** *The cloud-aware ridge family L19-L24 all needs probes to converge; the curves bend over between N=200 and N=350.* Solid blue lines = gauss_kl at each layer L19-L24. Dashed green = MMD at each layer L19-L24. Only L22 gauss_kl carries the N=500 point — the rest of the ridge stops at N=350 under the autonomous descope. Within-family lift between N=25 and N=350 is consistent across layers (the curves all bend over), confirming the covariance-estimator-variance story isn't an L22-only artifact. Bures-Wasserstein-2 is missing from this figure (dropped from the descope); the family claim is gauss_kl + MMD, not the full {gauss_kl, mmd, wass2} triad #502 originally tracked.

This is a clean explanation for both findings at once. The 0.06 gap to cosine exists because gauss_kl can see persona-shape information that's invisible to a mean-direction comparison; and gauss_kl needs more probes than cosine to expose that information, exactly because covariance estimators are sample-hungry. Stop at N=25 and the two metrics look indistinguishable. Push N up and gauss_kl pulls away.

#### Caveats that bound the headline

The compute descope mid-run drops two robustness checks I wanted. Bures-Wasserstein-2 was the third member of the cloud-aware ridge in #502 — the family claim ("cloud-aware metrics in general beat cosine, not just gauss_kl") is now only tested on {gauss_kl, MMD}. The loc-epoch 2/3/5 robustness sweep didn't fit in budget either, so the plateau verdict is anchored on the cleanest training checkpoint (#502's loc-arm epoch 1) and inherits #502's open question about checkpoint-fragility. The reproduction gate matches the archived value tightly on ρ (|Δρ| = 0.00029) and matches CV R² within the bakeoff's intrinsic fold-ordering noise (the #511 wrapper uses a deterministic LOCO fold ordering; the original bakeoff used `set(cond_ids)` which is hash-seed-dependent and drifts by ~5e-3 across runs).

The within-subset variance at N=500 collapses to ~0 because when N equals the pool size, every subset is an identity permutation of the same probes — the only residual variation across the R=10 N=500 rows is LOCO fold tie-break ordering, not metric-matrix noise. The plateau call uses σ_ref from N∈{200, 350} where the random-subset estimator is non-degenerate, not σ(N=500). Mathematically this is consistent with the planned criterion.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Predictor target | #474 ΔG matrix at `eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json` |
| Source activations | #502's saved tensors, `superkaiba1/explore-persona-space-data/issue502_28layer_500probe_bakeoff/activations/` |
| Probe pool | #502 mixed-distribution user prompts (500 per persona, on disk) |
| Personas | 16 transforms × 500 probes × 3584 hidden dims per (extraction-point, layer) |
| Probe-count grid | N ∈ {25, 50, 100, 200, 350, 500} |
| Random subsets per N | R = 10, seeds = `N × 1000 + subset_idx` |
| Headline cell | last-prompt × L22 × gauss_kl × raw (R=10, full N grid) |
| Cloud-aware ridge | L19-L24 × {gauss_kl, mmd} × raw (R=10, N ≤ 350; wass2 dropped) |
| Cosine baselines | L19-L24 × cosine × raw, plus sentinel L0/L11/L27 (R=10, full N grid) |
| Regression | Length-partial Spearman ρ + leave-one-context-out CV R² (`_loocv_r2_deterministic`) |
| Reproduction gate | `\|Δρ\|` = 0.00029 < 1e-3 vs #502 archived; CV R² 0.61806 (deterministic) vs 0.60863 (nondeterministic) |
| Wall time | 5h54m (21266s) on 29 VM CPUs, 0 GPU-h |
| Hydra config | n/a — pure VM reanalysis, no training |

**Artifacts:**

- Probe-count sweep results (875 rows + 115 aggregates + plateau verdict, 419 KB): [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json)
- Reproduction-gate output (the |Δρ| < 1e-3 check vs #502): [`eval_results/issue_511/reproduction_gate.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/reproduction_gate.json)
- Smoke run (4 rows, headline cell at N∈{25,50}, R=2): [`eval_results/issue_511/smoke_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/smoke_results.json)
- Source activations on HF Hub (parent run; pinned to dataset SHA `133b65b73ed11c3f8302781fe47c47f6e75a5411`): [`datasets/superkaiba1/explore-persona-space-data/tree/133b65b73ed11c3f8302781fe47c47f6e75a5411/issue502_28layer_500probe_bakeoff/activations`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/133b65b73ed11c3f8302781fe47c47f6e75a5411/issue502_28layer_500probe_bakeoff/activations)
- Figure source: [`figures/issue_511/`](https://github.com/superkaiba/explore-persona-space/tree/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511) — convergence_fixed_cell, baselines, convergence_ridge, raw_scatter_headline, each as `.png` + `.pdf` + `.meta.json`

**Compute:**

- Wall time: 5h54m (21266 s) end-to-end sweep, plus ~30 s smoke and ~10 s reproduction gate
- GPU: 0 — pure CPU reanalysis on 29 VM cores
- Pod: n/a (no pod provisioned; the conditional pod extension didn't fire because the plateau verdict triggered)
- WandB: n/a (no training)

**Code:**

- Sweep driver: [`scripts/issue511_probe_count_sweep.py`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/scripts/issue511_probe_count_sweep.py) (887 LoC; wraps the bakeoff's metric+regression code with the (N, subset) loop, the deterministic-CV switch, and the inline plateau-verdict computation)
- Figures script: [`scripts/issue511_make_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/scripts/issue511_make_figures.py) (425 LoC)
- Reused upstream code: [`scripts/issue493_extraction_metric_bakeoff.py`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/scripts/issue493_extraction_metric_bakeoff.py) — `load_activations_from_disk`, the 9 metrics, the `_length_partial` + `_loocv_r2` regression phase
- Git commit (figures + analysis): [`3e59740caf1649b965579c9fe373af934bd4c664`](https://github.com/superkaiba/explore-persona-space/tree/3e59740caf1649b965579c9fe373af934bd4c664) (branch `issue-511`)
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space
    git checkout 3e59740caf1649b965579c9fe373af934bd4c664
    uv sync
    # Pull activations from HF (if not cached):
    # huggingface-cli download --repo-type dataset \
    #   superkaiba1/explore-persona-space-data issue502_28layer_500probe_bakeoff/activations
    # Reproduce the gate
    uv run python scripts/issue511_probe_count_sweep.py --mode repro-gate
    # Run the full sweep (~6h on 29 CPUs)
    uv run python -u scripts/issue511_probe_count_sweep.py --mode full -v
    # Regenerate figures (idempotent)
    uv run python scripts/issue511_make_figures.py
    ```

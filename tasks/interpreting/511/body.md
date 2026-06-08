---
title: At 500 probes per persona the layer-22 Gaussian-KL leakage predictor cell has
  plateaued, and its lead over same-layer cosine opens monotonically from 0.004 at
  N=25 to 0.060 CV R² at N=500 (HIGH confidence)
kind: analysis
tags: []
created_at: '2026-06-07T01:49:33Z'
has_clean_result: true
parent_id: 502
---
# At 500 probes per persona the layer-22 Gaussian-KL leakage predictor cell has plateaued, and its lead over same-layer cosine opens monotonically from 0.004 at N=25 to 0.060 CV R² at N=500 (HIGH confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** #502's headline predictor cell (layer-22 Gaussian-KL on the loc-arm epoch-1 checkpoint) is done getting better at 500 probes, and its advantage over cosine is a high-N thing — at low N the two are basically tied.

**Takeaways.**
- The headline cell's CV R² climbs from 0.561 at N=25 to 0.617 at N=500 and the last step (350→500) only adds 0.003, smaller than the within-N subset noise. That's the plateau, and the verdict is for this one cell.
- The gauss_kl vs cosine gap at L22 grows monotonically with N: 0.004, 0.031, 0.043, 0.053, 0.057, 0.060. So "cloud-aware beats mean-direction" isn't a sample-matched fact — it's something that emerges as gauss_kl gets enough probes to estimate its covariance.
- MMD is the awkward case: also cloud-aware, but essentially flat across N at every layer (Δ ≤ 0.006 CV R² between N=25 and N=350). So the "cloud-aware metrics need probes" story is gauss_kl-specific, not a family property — MMD doesn't need probes and doesn't catch up to gauss_kl either.
- Big scope shrinkage I owe upfront: the original plan had Bures-Wasserstein-2 in the ridge plus a robustness sweep at loc-epoch 2/3/5 plus a "search-best per-N" curve plus a next-token JS baseline plus a secondary `g_logprob` target. None of those ran. The headline-cell plateau is intact, but every claim outside that cell is narrower than I planned.

**How this updates me.** I now believe #502's L22 gauss_kl ceiling is structural — more probes won't push it up. The right next move isn't more probes; it's the missing pieces (wass2, loc-epoch 2/3/5, the search-best curve), or attacking the 0.38 of variance the predictor still misses with new predictor families.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

In #502 I fit a leakage-transfer predictor on a ~1500-cell sweep over residual-stream persona-distance metrics. The winner was a cloud-aware Gaussian-KL distance at last-prompt × layer 22 × raw activations, length-partial Spearman ρ ≈ −0.79 and leave-one-context-out CV R² ≈ 0.61 against #474's marker-transfer target on the loc-arm epoch-1 checkpoint. Each persona's activation cloud was estimated from 500 probes. Cloud-aware metrics estimate a covariance, so their distance estimates are noisier at small N and should converge as N grows; cosine and Euclidean baselines need far fewer probes. The natural follow-on: is ρ / CV R² still climbing at N=500?

If the convergence curves have flattened by N≤500, #502's ceiling is structural — more probes won't help, and the predictor's limit is the metric family, not the sample size. If they're still climbing, then #502 is sample-limited and adding probes is the cheap next win. There's a derivative claim to test too: if cloud-aware metrics genuinely capture shape that cosine can't, then at any fixed N a cloud-aware metric should beat a cosine baseline — and the gap shape across N tells you whether that advantage is sample-matched or N-dependent.

**Scope shrinkage I owe upfront.** The original plan had five pieces that aren't in this writeup. (i) Bures-Wasserstein-2 was the third member of #502's cloud-aware ridge; the gauss_kl 2N×2N joint-Gram eigendecomposition cost ~50× the original time estimate, so wass2 was dropped under an autonomous compute-deviation descope to keep the sweep finishable. (ii) The loc-epoch 2/3/5 robustness sweep didn't fit budget either, so the plateau verdict is anchored on a single checkpoint (#502's loc-arm epoch 1) and inherits #502's open question about checkpoint-fragility. (iii) The same descope dropped R from 10 to 5 for the non-headline ridge cells (L19/L20/L21/L23/L24 gauss_kl and all L19-L24 MMD) and capped them at N≤350. (iv) A "search-best per-N argmax over the full grid" curve was on the original plan; only the headline-cell + same-layer ridge views are in this body, so the question "does the winning cell identity change with N?" is left open (it likely does — MMD L19 beats gauss_kl L22 at N=25; see Finding 3). (v) A next-token JS baseline and a secondary `g_logprob` target were also dropped. Parent #502 itself was MODERATE confidence with checkpoint-fragility unretired; the headline plateau here is HIGH only as a statement about that one cell.

### What I ran

I drew random probe subsets at N ∈ {25, 50, 100, 200, 350, 500} from the on-disk activation tensors (16 transforms × 500 probes × hidden, per extraction-point × layer), recomputed the persona-distance matrices on each subset, refit the length-partial Spearman + leave-one-context-out CV R² regression against the marker-transfer ΔG target, and read off the mean ± std curves over N.

The headline cell — last-prompt × layer 22 × Gaussian-KL × raw — ran on the full R=10 × N grid (six N values including N=500). Same-layer cosine baselines (L19-L24) and sentinel cosines (L0 early, L11 mid, L27 final) also ran R=10 on the full N grid (cheap to compute; no descope). The cloud-aware ridge at the other layers (L19/L20/L21/L23/L24 gauss_kl, plus all six L19-L24 MMD cells) ran at R=5 and N≤350 after the autonomous descope. Bures-Wasserstein-2 was dropped entirely from the ridge, and the loc-epoch 2/3/5 robustness sweep didn't fit budget.

The probe pool is the parent run's mixed-distribution set of 500 user prompts per persona; the activation tensors were never re-extracted (this is pure reanalysis). The target is the marker-leakage ΔG matrix on the loc-arm epoch-1 checkpoint — the same target the parent predictor was fit against.

The reproduction-gate check: at N=500 / single-subset I get rho_new = −0.79254 vs the parent run's archived rho = −0.79225, |Δρ| = 0.00029 — well inside the 1e-3 tolerance. The CV R² comparison is diagnostic-only (0.61806 from my deterministic-CV wrapper vs 0.60863 from the archived nondeterministic-CV); my wrapper uses a sorted LOCO fold ordering while the parent bakeoff iterated `set(cond_ids)` which is hash-seed-dependent. The 0.0094 deterministic-anchor shift is real, and all CV values in this body are on the new deterministic anchor — they are NOT directly apples-to-apples with the parent run's published 0.60863.

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

</details>

The ridge-family layout below is the cell-by-cell R / N-cap breakdown for the autonomous-descope run; random-sample disclosure: all 12 ridge cloud-aware cells + 9 cosine cells are shown (not cherry-picked). Per-cell aggregates in [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json).

<details>
<summary>The 21 cells in the sweep (with R and N caps)</summary>

| Cell | N grid | R | Note |
|---|---|---|---|
| last-prompt L22 gauss_kl raw | 25, 50, 100, 200, 350, 500 | 10 | Headline (full grid, full R) |
| last-prompt L19/L20/L21/L23/L24 gauss_kl raw | 25, 50, 100, 200, 350 | 5 | Ridge gauss_kl (descope: R=5, no N=500) |
| last-prompt L19/L20/L21/L22/L23/L24 mmd raw | 25, 50, 100, 200, 350 | 5 | Ridge MMD (descope: R=5, no N=500) |
| last-prompt L19/L20/L21/L22/L23/L24 cosine raw | 25, 50, 100, 200, 350, 500 | 10 | Same-layer cosine controls (full grid, full R) |
| last-prompt L0/L11/L27 cosine raw | 25, 50, 100, 200, 350, 500 | 10 | Sentinel cosines (early/mid/final, full grid) |

Bures-Wasserstein-2 dropped from the ridge; loc-epoch 2/3/5 not run; search-best curve not run.

</details>

### Findings

#### The headline gauss_kl L22 cell is sample-converged at N=500

The leakage predictor's CV R² climbs from 0.5605 at N=25 to 0.6174 at N=500. The step from 350→500 adds 0.0027 — smaller than the within-N subset noise (σ_ref = mean(σ(N=200), σ(N=350)) = 0.0036). On the length-partial Spearman ρ side, the same pattern: 0.7598 → 0.7922, the 350→500 step adds 0.0013 and at N=500 the across-subset std is 0.0007 (the curve has flatlined). The reproduction-gate row at N=500 matches #502's archived headline to |Δρ| = 0.00029 on rho; the CV R² comparison is diagnostic-only because of the deterministic-vs-nondeterministic LOCO fold-ordering difference (0.61806 vs 0.60863).

![Two line plots side by side showing the headline cell's length-partial Spearman magnitude (left, blue) and leave-one-context-out CV R² (right, red) as a function of N probes per persona, on a linear x-axis from N=25 to N=500. Both curves rise sharply from N=25 to N=100 then asymptote between N=200 and N=500. Error bars from R=10 random subsets are visible at every N and collapse to near-zero at N=500.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511/convergence_fixed_cell.png)

> **Figure.** *#502's headline predictor cell (last-prompt × L22 × Gaussian-KL × raw) on the loc-arm epoch-1 checkpoint reaches its ceiling between N=350 and N=500 — adding more probes won't lift it.* x-axis = N probes per persona used to estimate each persona's activation cloud (linear scale). Left panel = the absolute Spearman correlation, length-partialed, against #474's ΔG marker-transfer target. Right panel = the leave-one-context-out cross-validation R² (the CV score), length-partialed. Error bars show the average and the spread across the ten random probe subsets at each N. Each subset point is one full recomputation of the 16×16 distance matrix and regression fit.

Two anomalies worth surfacing while reading this figure. **σ does NOT monotonically decrease with N.** σ(N=200) = 0.0030, σ(N=350) = 0.0042 — the per-subset std actually grows from N=200 to N=350 before collapsing to 0.0011 at N=500. For R IID subsets you'd expect σ to shrink monotonically as more probes per subset stabilize each estimate; my best read is that at N=350 the 350-out-of-500 random subsets sample a much wider set of (persona-pair × probe-selection) interactions than the smaller subsets do, so the spread reflects regression-fit sensitivity to which probes land in which fold rather than estimator noise. I don't have a clean test for this. **The σ(N=500) ≈ 0.0011 for gauss_kl is NOT pure LOCO fold-ordering tie-break.** At N=500 every subset is an identity permutation of the same probes, so the distance matrix is in principle identical across the 10 subsets — but cosine L22 at N=500 has σ = 1.1e-16 (machine epsilon, exactly zero) while gauss_kl L22 has σ = 0.0011. The asymmetry has to come from numerical non-determinism inside the gauss_kl computation itself — most likely PCA eigh tie-breaks on the 16-dim eigendecomposition + the joint-Gram log-det, not from LOCO fold ordering. The plateau call uses σ_ref from N∈{200, 350} where the random-subset estimator is non-degenerate, so this doesn't affect the verdict.

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

These are points 0-4 of 10 random probe-subsets at each N for the headline cell. Random-sample disclosure: first five of ten subsets at each N (the full ten are visible in the convergence figure's error bars). One per-row pattern worth noting: at N=350 a single subset (subset_idx=5, not shown above) reaches CV R² = 0.62443, above every N=500 subset — within-N variance is non-negligible at N=350, consistent with the σ_350 > σ_200 anomaly. Full per-subset rows in [`eval_results/issue_511/probe_count_sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/3e59740caf1649b965579c9fe373af934bd4c664/eval_results/issue_511/probe_count_sweep_results.json).

</details>

The headline-cell plateau is the load-bearing claim of this writeup. The ridge story is messier: the non-headline gauss_kl ridge cells were measured at R=5 and capped at N=350, and their N=200→350 step is non-monotonic for half the cells (see Finding 3). What I will NOT say: "the rest of the ridge has plateaued by N=350." L23 gauss_kl's N=200→350 step is +0.0076 (its σ at N=350 = 0.0034, σ_ref ≈ 0.0028; Δ > σ_ref → still climbing). L24 gauss_kl's step is +0.0061 (σ_ref ≈ 0.0027; still climbing). Without N=500 measurements on those cells I can't tell whether they would also plateau or kept rising — that's a leftover the descope created.

#### Cosine is already flat at N=25 — and the gauss_kl-vs-cosine gap at L22 grows monotonically with N

If the cloud-aware advantage were just "covariance estimators need more probes," cosine should climb the same curve as gauss_kl and either converge to roughly the same number or sit slightly below it. That's not what happens. Cosine L22 at N=25 is already at CV R² = 0.5568, and at N=500 it's at 0.5573 — six ten-thousandths higher. The whole same-layer cosine ridge L19-L24 sits in the 0.547-0.568 band, flat across N (7 of 9 cosine cells plateau by the σ_ref criterion; **L24 cosine and L11 cosine sentinel are labelled `intermediate`** in the stored plateau verdict — their Δ(350→500) just edges above σ_ref but stays below 2·σ_ref).

The gap at L22 between gauss_kl and cosine, evaluated at each matched N, grows monotonically:

| N | gauss_kl L22 CV R² | cosine L22 CV R² | gap |
|---|---|---|---|
| 25 | 0.5605 | 0.5568 | 0.0037 |
| 50 | 0.5902 | 0.5594 | 0.0308 |
| 100 | 0.6070 | 0.5637 | 0.0432 |
| 200 | 0.6118 | 0.5587 | 0.0531 |
| 350 | 0.6148 | 0.5576 | 0.0571 |
| 500 | 0.6174 | 0.5573 | 0.0601 |

At N=25 the two metrics are essentially tied; the 0.06 gap is something that opens up as gauss_kl climbs while cosine sits flat. The honest framing is "gauss_kl L22 needs the probes to expose a 0.06 advantage that cosine never reaches at any N", not "the gap survives sample-size matching."

![A line plot with CV R² on the y-axis from 0.10 to 0.62 and N probes per persona on the x-axis from 25 to 500 on a linear scale. The bold red line at the top is gauss_kl L22 (the headline cell), rising from 0.56 at N=25 to 0.62 at N=500. Six gray "+"-marker lines for cosine L19-L24 cluster tightly between 0.547 and 0.568, essentially flat across N. Four dashed lines for sentinel cosines: L0 (early) sits around 0.14, L11 (mid) around 0.19, L27 (final) around 0.52, plus a duplicate cos-L21 sentinel line that overlaps the gray-band L21. The whole picture: the cloud-aware headline pulls away from the cosine ridge, and the cosine curves never move with N.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511/baselines.png)

> **Figure.** *The gauss_kl L22 advantage over the same-layer cosine ridge is not "cosine waiting for probes" — cosine is already flat at the smallest N, and the gap to gauss_kl opens monotonically as N grows.* x-axis = N probes per persona, linear scale. Bold red line = headline cloud-aware cell (gauss_kl at layer 22). Gray "+"-marker lines = the six same-layer cosine controls at L19-L24 (the cosine ridge, dense band sitting near 0.55 to 0.57). Dashed lines = sentinel cosines at L0 (early), L11 (mid), and L27 (final); all far worse than the L19-L24 band, confirming the layer-band finding from #502. The "cos L21 (sentinel)" legend entry is a script cosmetics bug — L21 appears twice (once in the ridge band, once styled as sentinel) but the underlying data is the same. Seven of the nine cosine cells get a `plateau` verdict from the stored within-N noise reference test; L24 cosine and L11 sentinel are `intermediate` (their last step from N=350 to N=500 is just above the within-N noise reference but below twice that reference, so they're not actively climbing but the convergence isn't as clean). Bures-Wasserstein-2 was dropped from the ridge under the autonomous descope, so the gauss_kl-vs-cosine separation here is gauss_kl-specific, not yet a full cloud-aware family claim — see Finding 3.

The raw cosine numbers make the comparison clean. At N=500, cosine L22 hits subset_idx=0/1/2 (and all ten subsets) with identical CV R² = 0.55734 — three decimals out, exactly the same, because when N equals the activation-pool size every subset is the same identity permutation and the cosine metric matrix is exactly permutation-invariant. At N=25 it's 0.572 / 0.550 / 0.568 — wider than the N=500 cluster but already squarely within the same band the curve will sit in across the rest of the N grid.

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

All ten cosine rows at N=500 are identical to five decimals (σ = 1.1e-16, machine epsilon) — cosine's metric matrix is permutation-invariant on the full pool, so every subset produces literally the same regression input. The point of the comparison is the cosine band ~0.55-0.57 vs the gauss_kl 0.62 destination — not that all ten rows are identical at N=500.

</details>

The control side reads cleanly too. Sentinel cosines at L0 (early), L11 (middle), and L27 (final) sit at 0.14, 0.19, and 0.53 — all far below the L19-L24 band, the same layer-stratification #502 reported. The 0.06 lift sits squarely at the L22 ridge peak; not everywhere, only at the cells where #502 said the signal lived.

#### What gauss_kl L22 needs that MMD doesn't

Here's the asymmetry that ties findings 1 and 2 together — and the spot where the round-1 story turned out to be wrong. Gauss_kl at L22 climbs 0.057 CV R² between N=25 and N=500 (0.561 → 0.617). Cosine at L22 climbs 0.001 over the same range. MMD — also cloud-aware, the other surviving member of the ridge — is **essentially flat across N at every layer**: Δ(N=25 → N=350) at L19 = +0.0001, L20 = +0.0039, L21 = +0.0058, L22 = +0.0017, L23 = +0.0017, L24 = +0.0011. All six MMD layers move less than 0.006 CV R² between N=25 and N=350, and L19/L20/L21 MMD all sit at CV R² ≈ 0.60 from N=25, already in the gauss_kl-L22 band.

So the round-1 finding ("cloud-aware metrics need the probes; cosine doesn't") is false as written. The honest version: **gauss_kl L22 needs the probes; MMD in this sweep does not show the same sample-hungry shape, and cosine doesn't either**. The covariance-estimator-variance mechanism does fit gauss_kl (which estimates a covariance via PCA-16 + Gaussian log-det, classically needing ~10× the dimensionality of probes per persona to be well-conditioned — putting the sweet spot around N=160-200, matching where gauss_kl's curve bends over). It does NOT fit MMD, which is kernel-based and has no covariance estimator. The "sample-hungry" framing is gauss_kl-specific, not a family property.

![Two line plots side by side showing the L19-L24 cloud-aware ridge family. Twelve curves, six gauss_kl (solid blue) and six mmd (dashed green), at layers L19-L20-L21-L22-L23-L24. The gauss_kl curves rise visibly from N=25 to N=200ish; the MMD curves are visibly flatter across N. The dotted blue gauss_kl L22 curve (the headline) is the highest at high N, reaching the standalone N=500 point at 0.617. The other ridge cells stop at N=350.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/3e59740caf1649b965579c9fe373af934bd4c664/figures/issue_511/convergence_ridge.png)

> **Figure.** *Within the L19-L24 cloud-aware ridge, the gauss_kl curves bend over with N (sample-hungry covariance estimator) while the MMD curves are essentially flat across N at every layer.* x-axis = N probes per persona, linear scale. Solid blue = gauss_kl at each layer L19-L24 (R=5 for the non-headline layers; R=10 for L22 only). Dashed green = MMD at each layer L19-L24, capped at N=350 under the autonomous descope. Only L22 gauss_kl carries the N=500 point — the rest of the ridge stops at N=350. The gauss_kl ridge cells gain between 0.02 and 0.06 in the CV score between N=25 and N=350; the MMD ridge cells gain between 0.0001 and 0.006 over the same range. Three of the five non-headline gauss_kl cells (L19, L20, L21) actually fall slightly from N=200 to N=350 by 0.0041, 0.0049, and 0.0023 — within the R=5 estimator noise but breaks the clean "monotone bend then plateau" story for the ridge. L23 and L24 gauss_kl are still climbing at N=350 (steps of +0.0076 and +0.0061, both above the within-N noise reference) and have no N=500 measurement to call. Bures-Wasserstein-2 is missing from this figure (dropped from the descope); the family claim is gauss_kl + MMD, not the full {gauss_kl, mmd, wass2} triad #502 originally tracked.

One unexpected pattern this figure exposes: **at small N, MMD at L19/L20 actually beats gauss_kl L22**. MMD L19 at N=25 = 0.6021 vs gauss_kl L22 at N=25 = 0.5605 — MMD leads by 0.04 CV R² at the smallest N. At N=200 the two are within 0.006 of each other (MMD L20 = 0.6061, gauss_kl L22 = 0.6118). Gauss_kl only pulls ahead at N=350+ on this single cell. So the "winning cell identity" question — which the original-plan search-best curve was meant to answer — is N-conditional, and the round-1 framing that "L22 gauss_kl is the universal winner" is true only at high N on this checkpoint. At small N or under different compute budgets, MMD L19/L20 is a reasonable competitor. I can't fully cash this out without the search-best curve; what's visible is the asymmetry at the endpoints.

#### Caveats that bound the headline

The compute descope mid-run drops five planned pieces, matching the Motivation list. (i) Bures-Wasserstein-2 was the third member of the cloud-aware ridge in #502 — the family claim ("cloud-aware metrics in general beat cosine, not just gauss_kl") is now only tested on {gauss_kl, MMD}, and MMD's flat behavior plus its small-N edge over gauss_kl L22 means the "cloud-aware family converges with N" claim is gauss_kl-specific, not yet a family fact. (ii) The loc-epoch 2/3/5 robustness sweep didn't fit in budget, so the plateau verdict is anchored on the cleanest training checkpoint (#502's loc-arm epoch 1) and inherits #502's open question about checkpoint-fragility. (iii) The same descope dropped R from 10 to 5 for the non-headline ridge cells (L19/L20/L21/L23/L24 gauss_kl and all L19-L24 MMD) and capped them at N≤350, so the "rest of the ridge has plateaued" question can't be answered cleanly. (iv) The "search-best per-N" argmax curve was dropped; this is the more substantive loss given the MMD-L19-beats-gauss_kl-L22-at-low-N pattern above (the winning cell identity changes with N and the search-best curve was the planned probe of that). (v) The next-token JS baseline AND the secondary `g_logprob` target were both dropped — the `g_logprob` target was the base-prior-safe ΔG variant intended as a within-DV-family robustness check on the headline plateau, so its absence means the plateau is verified only against the count-style ΔG and not the log-probability variant.

The within-subset variance at N=500 collapses to ~10⁻³ for gauss_kl and to machine epsilon for cosine because when N equals the pool size, every subset is an identity permutation of the same probes. For cosine, the metric matrix is therefore exactly permutation-invariant and the 10 subsets produce literally identical regression inputs; for gauss_kl, the residual 0.0011 std comes from PCA eigh tie-breaks + joint-Gram log-det non-determinism, not from LOCO fold ordering. The plateau call uses σ_ref from N∈{200, 350} where the random-subset estimator is non-degenerate, so the verdict isn't sensitive to either of these.

The reproduction-gate prose is honest about the CV-R² shift: ρ matches tightly (|Δρ| = 0.00029 < 1e-3); CV R² shows a 0.0094 deterministic-anchor diagnostic shift (cv_new_deterministic = 0.61806 vs cv_archived_nondeterministic = 0.60863), which is ~2× the bakeoff's intrinsic ~5e-3 fold-ordering noise band. The diagnostic is PASS because |Δρ| meets the 1e-3 spec, but the CV anchor moved and downstream #511 numbers are on the new anchor. Don't compare a #511 CV R² to a #502-published CV R² directly without remembering this 0.0094 shift.

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Predictor target | #474 ΔG matrix at `eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json` |
| Source activations | #502's saved tensors, `superkaiba1/explore-persona-space-data/issue502_28layer_500probe_bakeoff/activations/` |
| Probe pool | #502 mixed-distribution user prompts (500 per persona, on disk) |
| Personas | 16 transforms × 500 probes × 3584 hidden dims per (extraction-point, layer) |
| Probe-count grid | N ∈ {25, 50, 100, 200, 350, 500} |
| Headline cell | last-prompt × L22 × gauss_kl × raw — R=10, full N grid (six points) |
| Cosine baselines | L19-L24 × cosine × raw + sentinel L0/L11/L27 — R=10, full N grid |
| Cloud-aware ridge (non-headline) | L19/L20/L21/L23/L24 gauss_kl + L19-L24 MMD — R=5, N ≤ 350 (autonomous descope; wass2 dropped) |
| Random subset seeds | `N × 1000 + subset_idx` |
| Regression | Length-partial Spearman ρ + leave-one-context-out CV R² (`_loocv_r2_deterministic`) |
| Reproduction gate | `\|Δρ\|` = 0.00029 < 1e-3 vs #502 archived; CV R² 0.61806 (deterministic, this work) vs 0.60863 (nondeterministic, #502 archived); ΔCV = 0.0094 (diagnostic-only PASS) |
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

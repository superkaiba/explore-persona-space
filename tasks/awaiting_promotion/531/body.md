---
title: Off-ceiling, base prior still tracks marker shift negatively but predicts absolute
  trained log-prob strongly positively — propensity hides in the subtraction (MODERATE
  confidence)
kind: analysis
tags: []
created_at: '2026-06-09T05:44:57Z'
has_clean_result: true
parent_id: 478
---
# Off-ceiling, base prior still tracks marker shift negatively but predicts absolute trained log-prob strongly positively — propensity hides in the subtraction (MODERATE confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** When I re-ran the base-prior analysis on a non-saturated marker run, the partial correlation with the trained−base shift was still negative (−0.48), same sign as the saturated case but smaller — so the "ceiling arithmetic kills the sign" story is incomplete. But when I looked at absolute trained log-prob instead of the shift, the partial correlation flipped strongly positive (+0.74), consistent with the propensity story from the facts experiment.

**Takeaways.**
- The two readings are mathematically consistent: a row-level fit gives trained ≈ 0.62 × base + 1.10, so shift ≈ −0.38 × base + 1.10 — a positive slope on trained with coupling below 1 mechanically becomes a negative slope on shift.
- The data are consistent with propensity / rank preservation off-ceiling, but they don't independently distinguish propensity from a fixed compressed-coupling read — both stories produce the same sign pattern.
- The earlier prediction that desaturation would flip the shift correlation positive does NOT cleanly hold; the magnitude shrinks (−0.87 → −0.48) but the sign stays.
- For future predictor work: report absolute trained log-prob alongside the shift. Shift's −base term confounds the propensity read.
- A follow-up pass re-measured everything in logit and probability space. The rank-preservation read is space-robust (partial ρ +0.70 to +0.77 in every representation), but the negative shift read is not: −0.48 in log-prob, −0.42 in EOS-margin, only −0.13 against the base logit, and +0.74 in probability space (where the −base term is numerically invisible because trained mass is ~17,000× base).
- The softmax normalizer at the slot drops 2.7 nats on average after training, so log-prob shifts run ~2.7 nats above the marker-logit shifts — the usual off-saturation shortcut "Δlog P ≈ Δz" does not hold at these cells.

**How this updates me.** I now think base prior is at minimum a strong rank-preserving signal — high-prior personas land at higher trained log-prob — and the saturated case's strong negative was partly ceiling arithmetic AND partly a real underlying negative partial on the shift DV that just happens to be there off-ceiling too. Two readings, mathematically reconciled. Whether the underlying mechanism is propensity or a compressed-coupling artifact isn't settled by this analysis.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

Two prior reads pointed in opposite directions. On the saturated marker run, the partial Spearman ρ between the held-out persona's base-model prior on the marker and the trained−base shift was −0.87. But bystanders were 92% pinned at log-prob ≈ 0 there, which forces shift ≈ −base_prior almost by identity. The facts run, well off-ceiling, gave the opposite sign: base prior on the taught fact predicted leakage positively. Which sign survives off-ceiling, and how much of the saturated −0.87 was just the identity?

I already had the right data sitting in another task. The parallel marker run was firmly off-ceiling: source trained log P around −11 nats, zero emission across 2,800 held-out trials. I'd only used it to ask about distance flattening and never extracted the base-prior relationship. So this is an analysis-only pull: re-fit the partial Spearman in the non-saturated regime, and see which sign the marker wants when the identity isn't doing the work.

### What I ran

Per-cell log-prob arrays come straight from the CORE-track cells of the parallel run: **40 cells × 2 seeds = 80 trained runs**, each evaluated on **35 held-out personas × 20 questions = 700 probe rows per run**, for **56,000 rows total**. Each row carries the trained log-prob on the marker at the post-response slot, the base-model log-prob at the same slot, the shift (trained − base), the held-out persona's cosine distance to the nearest source in the training mix, and the K factor (number of source personas trained on the marker per cell). The 12 decomposition-arm cells are excluded; they assign a different marker per source, so within-cell exposure isn't uniform.

First I confirmed non-saturation: mean trained log P = −12.70 nats, 0% of rows above −1 nat. Then I ran two partial Spearmans, both controlling for min_dist and K, with a 1000-resample persona-cluster bootstrap so within-persona dependence is respected. DV #1 is the shift, matching the saturated #504 analysis. DV #2 is absolute trained log-prob: does high base prior predict ending up at higher absolute trained log-prob, independent of how much it moved?

A follow-up pass then re-measured the same two relationships in the other two spaces. Probability space is a free transform of the stored values (P = exp(log P)). Logit space required new measurement — the parent run stored only resolved log-probs, and logits cannot be recovered from log-probs without the per-slot normalizer — so I re-scored every stored on-policy response through the corresponding trained adapter (PEFT hot-swap on the shared base, adapter disabled for the base side) on 2× H100, capturing the raw marker logit, the EOS logit, and log Z at the same slot from the same forward pass. The recomputed log-probs match the stored ones to a worst-case per-cell MAE of 0.33 nats and worst Spearman 0.996, so the reconstruction is faithful to the original scoring. Same 56,000 rows, same controls, same persona-cluster bootstrap throughout.

### Findings

#### The shift partial stays negative off-ceiling — desaturation didn't kill the sign

![Scatter of trained minus base marker log-prob shift against the base-model marker prior, colored by held-out persona distance band; clear negative slope across all bands.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.png)

> **Figure.** *Off the ceiling, the shift still slopes down against base prior — same sign as the saturated case, weaker.* Each point is a (cell, seed, held-out persona, question) row, sub-sampled to ~600 per distance band for legibility. x-axis: base-model log P(marker) at the model's own post-response slot (nats). y-axis: trained − base log-prob shift (nats). Colors are persona-distance bands (near to tail). Raw Spearman ρ = −0.659 [95% CI −0.766, −0.522], partial ρ (controlling for min_dist + K) = −0.480 [−0.603, −0.333], n = 56,000 rows (40 cells × 2 seeds × 35 personas × 20 questions), 1000-resample persona-cluster bootstrap.

The simple read of the saturated −0.87 was: trained pinned at zero forces shift ≈ −base_prior by the identity. If that were the whole story, desaturation should null out the correlation, or even flip it positive if there's a real per-persona effect underneath. Neither happens. Mean trained log-prob sits at −12.7 nats, the ceiling is gone, and the partial is still −0.48 with 95% CI [−0.603, −0.333]. That's about 55% the magnitude of the saturated case, with the same sign. The identity explanation alone is wrong. A real negative partial on the shift DV survives desaturation.

The signs are stable across K. Controlling for K=1, 2, 4, and 8 individually keeps the shift partial at ρ ≈ −0.64 to −0.69 and the absolute-trained partial at ρ ≈ +0.69 to +0.73. The shift partial's magnitude does shift by distance band: the near band is weaker than near-mid and far.

#### Absolute trained log-prob points the opposite way — base-prior rank order survives training

![Scatter of trained marker log-prob against the base-model marker prior, colored by persona-distance band; clear positive slope above the y=x diagonal.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.png)

> **Figure.** *On absolute trained log-prob, base prior predicts strongly positively — high-prior personas land at higher trained mass.* x-axis: base-model log P(marker) at the post-response slot. y-axis: absolute trained log P(marker) at the same slot. Same 56,000 rows (40 cells × 2 seeds × 35 personas × 20 questions), same bands, same bootstrap. Raw Spearman ρ = +0.705 [+0.610, +0.777], partial ρ (controlling for min_dist + K) = +0.739 [+0.658, +0.807]. The cloud forms a strong upward band: a row-level affine fit gives trained ≈ 0.62 × base + 1.10, which sits well above y=x in absolute log-prob coordinates — every persona's trained log-prob is shifted up relative to its base, but the rank order in base prior is substantially preserved in trained.

Two persona-mean comparisons make the rank story concrete. `joker` has the highest base prior (−18.5 nats) and ends up at the highest trained log-prob (−10.4 nats). `medical_doctor` sits among the lowest base priors (−26.0 nats) and ends up among the lowest trained log-probs (−13.6 nats). Training raises log-prob for everyone, and the low-prior personas actually move farther: `medical_doctor`'s shift (+12.3 nats) is larger than `joker`'s (+8.1). But training doesn't equalize them. The base-prior rank order survives. That matches the facts experiment's direction: high base prior on the target predicts higher trained mass.

Two stories are consistent with this. Read one (propensity): training preserves a per-persona preference, so the persona-specific gap between trained and base shrinks but doesn't invert the rank order. Read two (compressed coupling): training applies a fixed sub-unit affine map (slope 0.62, intercept +1.10) to base prior across all bystanders, with no per-persona term needed. The two readings differ in what they predict under varying the affine slope (LoRA rank, learning rate, training steps). At fixed training they don't. Both produce the positive absolute-trained partial AND the negative shift partial via the `shift = trained − base` identity, and this analysis can't separate them.

#### The two findings reconcile via shift = trained − base, and the saturated case was both a ceiling effect and a real negative

The shift partial (−0.48) and the absolute-trained partial (+0.74) point opposite ways but describe the same rows. Shift is just trained − base, so partial ρ(trained, base) and partial ρ(shift, base) differ only by what the −base subtraction adds. The row-level affine fit makes this concrete. Trained ≈ 0.62 × base + 1.10, so shift ≈ −0.38 × base + 1.10. With coupling below 1, the −base term dominates the partial on shift and pulls it negative. The same data give a positive partial on trained. Both numbers are right. They describe different cuts of the same joint distribution.

What this means for the saturated #504 result: the strong negative there (−0.87) was not purely a ceiling artifact. The desaturated read still shows −0.48 on shift. The saturated case carried both a real underlying negative partial on shift and additional ceiling-driven amplification. Base-prior rank order shows through to trained log-prob (high base-prior personas land at higher trained log-prob) in these data too. But it only surfaces in the absolute-trained cut; the shift cut hides it behind the −base term. Whether the underlying mechanism is per-persona propensity or a fixed compressed affine map across bystanders, this analysis doesn't say.

Head-to-head against the two siblings:

| | base prior → leakage, raw ρ | base prior → leakage, partial ρ | regime |
|---|---|---|---|
| This re-analysis (marker, non-saturated) | shift: −0.66; absolute trained: +0.71 | shift: −0.48; absolute trained: **+0.74** | off-ceiling |
| #504 (marker, saturated) | shift: −0.90 | shift: −0.87 | bystanders 92% saturated |
| #500 (facts, non-saturated) | leak: +0.80 (marine) | β_prior = +0.78 (joint fit) | off-ceiling |

The prediction going in was that desaturation would flip the shift partial from −0.87 toward positive, finishing the identity-explains-everything story. That doesn't cleanly hold. The shift partial stays negative at −0.48 — smaller, but the wrong sign for that prediction. A positive sign does show up, but on a different DV (absolute trained, +0.74), not the one the prediction was about. The plan called shift; absolute trained is a cut I added in the analysis, not what I went in to test. So this is a DV swap, not a confirmation of the original frame. The saturated −0.87 was partly identity and partly something real on shift, and the positive sign from the facts experiment shows up in absolute-trained space but not in shift. The data don't separate per-persona propensity from a fixed compressed affine map across bystanders.

#### In logit space the negative shift nearly vanishes while rank preservation survives — the shift's sign is representation-dependent

A follow-up re-scoring pass put raw logits behind every row: each stored on-policy response was teacher-forced through its trained adapter and through base, reading the marker logit z(※), the EOS logit, and log Z at the post-response slot (recomputed log-probs match the stored values, worst-case MAE 0.33 nats). The plot below is the shift panel with both axes in logit space: x = base marker logit, y = trained − base marker logit.

![Scatter of trained minus base marker logit against the base-model marker logit, colored by held-out persona distance band; the cloud is nearly flat with no strong slope.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f11eefa91aa3876faaedba768bb906e6bf783201/figures/issue_478/base_prior_reanalysis/shift_vs_base_prior_logit.png)

> **Figure.** *Against the base marker logit, the logit shift is nearly flat — the strong log-prob negative does not carry over.* Same 56,000 rows, same bands. Raw Spearman ρ = −0.024 [95% CI −0.173, +0.117], partial ρ (controlling for min_dist + K) = −0.134 [−0.247, −0.015].

The cross-space grid sharpens the parent's headline. Absolute rank preservation is space-robust: the trained value tracks the base value at partial ρ = +0.70 in raw logits, +0.77 in the EOS margin z(※) − z(EOS), and +0.74 in log-prob. The negative shift partial is not: −0.48 in log-prob, −0.42 in the EOS margin, but only −0.13 [−0.25, −0.02] when both axes are logits. Keeping the parent's x-axis (base log P) and swapping only the DV to the logit shift gives −0.39 [−0.52, −0.25], so most of the collapse comes from the x-axis side: base log P = z(※) − log Z carries the per-context normalizer, and that normalizer component does real work in the parent's −0.48.

The same pass surfaced a measurement caveat worth keeping: mean Δlog Z = −2.73 nats (p95 |Δlog Z| = 4.8). Training drops the slot's normalizer, so log-prob shifts run ~2.7 nats above the marker-logit shifts on the same rows (rank agreement between the two stays high, ρ = 0.94). The usual off-saturation shortcut "Δlog Z ≈ 0, so the log-prob shift reads as the marker-direction push" does not hold at these cells, even though they are far from the ceiling.

#### Probability space flips the shift positive — ΔP is just trained P because the base prior is negligible

The probability-space version of the shift panel needs no new measurement (P = exp(log P) of the stored values). Trained marker mass is ~e^9.7 ≈ 17,000× the base prior on the median row, so ΔP = P_trained − P_base is numerically P_trained, and the −base subtraction that drives the log-space negative does nothing here.

![Scatter of trained minus base marker probability against the base-model marker probability; nearly all mass hugs the origin with a positive fan, axes in scientific notation.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/f11eefa91aa3876faaedba768bb906e6bf783201/figures/issue_478/base_prior_reanalysis/shift_vs_base_prior_probability.png)

> **Figure.** *In probability space the shift correlation is positive and identical to the absolute panel — the subtraction is invisible at these magnitudes.* Same rows, same bands, same bootstrap. Raw Spearman ρ = +0.704 [+0.610, +0.777], partial ρ (controlling for min_dist + K) = +0.739 [+0.658, +0.807] — matching the absolute-trained partial to three decimals because ΔP ≈ P_trained. The absolute probability panel is rank-identical to the log-prob one by construction (Spearman is invariant under exp), so its numbers are unchanged.

This is the cleanest illustration of the parent's "propensity hides in the subtraction" point: the same shift question gives −0.48, −0.13, or +0.74 depending purely on the representation, while the absolute question gives +0.70 to +0.77 everywhere. Probability space stays a sanity read, not an analysis space — absolute probability change scales with the base prior, so it over-weights high-prior contexts by construction.

## Reproducibility

**Parameters:**

| field | value |
|---|---|
| base model | Qwen-2.5-7B-Instruct (id `Qwen/Qwen2.5-7B-Instruct`, per #478's per-cell `result.json` `training.base_model`; an earlier draft of this table said the non-Instruct id — corrected during the logit follow-up, which validated against the stored scores using the Instruct model) |
| analysis target | task #478 marker run, CORE track only (40 cells, K ∈ {1, 2, 4, 8}) |
| seeds | 42, 137 (2 per cell) |
| trained runs | 80 = 40 cells × 2 seeds |
| n_rows | 56,000 = 80 trained runs × 35 held-out personas × 20 questions |
| marker | ` ※` (Qwen-2.5-7B token id 83399, leading space) |
| measurement | on-policy log P(marker) at the post-response slot |
| ceiling check | mean trained log P = −12.70 nats, share above −1 nat = 0.00% |
| controls | min_dist (held-out persona cosine distance to nearest source), K (sources per cell) |
| inference | partial Spearman ρ with 1000-resample persona-cluster bootstrap (resampling held-out personas, not rows; respects within-persona dependence) |
| ARM cells | excluded (12 of 92) — they assign different markers per source, mixing two regimes |
| per-persona-mean Spearman | not emitted as a separate number; the persona-cluster bootstrap IS the per-persona inference |
| affine fit (row-level) | trained ≈ 0.62 × base + 1.10; implies shift ≈ −0.38 × base + 1.10 |
| config | n/a (analysis-only; no Hydra run) |
| logit re-scoring pass (follow-up) | all 80 CORE runs' stored on-policy responses teacher-forced through trained (base + LoRA adapter, PEFT hot-swap, bf16) and base (adapter disabled), capturing z(※) (id 83399), z(EOS) (`<\|im_end\|>` id 151645), and log Z at the post-response slot; 2× H100, 2 shard workers; adapters from HF `superkaiba1/explore-persona-space` `issue_478/<cell>/adapter` (repo revision recorded per cell JSON) |
| re-scoring validation | recomputed log P vs stored, per cell × side: worst MAE 0.33 nats, worst Spearman 0.996 (residual = PEFT-runtime vs bf16-merged rounding) |
| gauge assert | adapter `target_modules` exclude `lm_head`/`embed_tokens`, `modules_to_save` empty — checked per adapter before any logit readout |

**Artifacts:**
- Analysis script: [scripts/issue531_base_prior_reanalysis.py](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/scripts/issue531_base_prior_reanalysis.py)
- Tidy table (56,000 rows): [eval_results/issue_478/base_prior_reanalysis/tidy.parquet](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/eval_results/issue_478/base_prior_reanalysis/tidy.parquet)
- Summary stats (full bootstrap output, head-to-head quotes): [eval_results/issue_478/base_prior_reanalysis/summary.json](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/eval_results/issue_478/base_prior_reanalysis/summary.json)
- Shift figure source: [figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.png](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/shift_vs_base_prior.png) + PDF + meta.json
- Absolute-trained figure source: [figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.png](https://github.com/superkaiba/explore-persona-space/blob/f9ad999dcd082f48c1305525995ad4bac3cdd21f/figures/issue_478/base_prior_reanalysis/absolute_trained_vs_base_prior.png) + PDF + meta.json
- Source data (per-cell log-prob arrays): per-cell `result.json` files from task #478's `issue-478` branch at commit [7efb037736831c66cf87aaa79c11237ac9268b83](https://github.com/superkaiba/explore-persona-space/tree/7efb037736831c66cf87aaa79c11237ac9268b83/eval_results/issue_478). Read via `git show <sha>:<path>` from this worktree. The HF data revision pinned in summary.json (`a9fc5a9...`) stores response text but not per-question log-prob arrays — those live in the per-cell `result.json`.
- min_dist + band assignments: reused byte-for-byte from #478's aggregate `tidy.csv` at the same SHA, not recomputed.
- Figure `.meta.json` files record `git_commit_at_render = 22d73be7e...` (intermediate worktree state during the render pass). Canonical artifact commit on `issue-531` is `f9ad999dcd082f48c1305525995ad4bac3cdd21f`; the data referenced is identical between the two SHAs.
- **Logit/probability follow-up** (canonical commit [f11eefa91](https://github.com/superkaiba/explore-persona-space/commit/f11eefa91aa3876faaedba768bb906e6bf783201)):
  - Re-scoring worker: [scripts/issue531_logit_rescore.py](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/scripts/issue531_logit_rescore.py) + launcher [scripts/issue531_logit_rescore_launch.sh](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/scripts/issue531_logit_rescore_launch.sh)
  - Logit analysis + figures: [scripts/issue531_logit_plots.py](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/scripts/issue531_logit_plots.py); probability figures: [scripts/issue531_probability_space_plots.py](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/scripts/issue531_probability_space_plots.py)
  - Per-cell raw logits (80 JSONs, trained+base z(※)/z(EOS)/log Z/log P/argmax per question): [eval_results/issue_478/logit_rescore/](https://github.com/superkaiba/explore-persona-space/tree/f11eefa91aa3876faaedba768bb906e6bf783201/eval_results/issue_478/logit_rescore)
  - Merged logit tidy table + stats: [tidy_logit.parquet](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet), [summary_logit.json](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/eval_results/issue_478/base_prior_reanalysis/summary_logit.json), [summary_probability.json](https://github.com/superkaiba/explore-persona-space/blob/f11eefa91aa3876faaedba768bb906e6bf783201/eval_results/issue_478/base_prior_reanalysis/summary_probability.json)
  - Figures (each + PDF + meta.json): `shift_vs_base_prior_logit`, `absolute_trained_vs_base_prior_logit`, `shift_vs_base_prior_eos_margin`, `shift_vs_base_prior_probability`, `absolute_trained_vs_base_prior_probability` under [figures/issue_478/base_prior_reanalysis/](https://github.com/superkaiba/explore-persona-space/tree/f11eefa91aa3876faaedba768bb906e6bf783201/figures/issue_478/base_prior_reanalysis)

**Compute:** Main analysis: local VM only, no GPU, wall time 4m55s (bulk in the 1000-resample persona-cluster bootstrap × 4 partial-Spearman refits per resample); smoke on 2 cells = 42 s. Logit follow-up: 2× H100 ephemeral pod (pod-531), ~20-30 s of scoring per (cell, seed) run, ~25 min of GPU scoring total across the 80 runs (wall time longer due to repeated pod auto-stops by a session watcher, recovered via per-cell checkpointing + resume); local CPU bootstrap for the 12 logit/probability Spearman fits ≈ 13 min.

**Code:** Branch `issue-531`, head commit [f9ad999dcd082f48c1305525995ad4bac3cdd21f](https://github.com/superkaiba/explore-persona-space/commit/f9ad999dcd082f48c1305525995ad4bac3cdd21f). Reproduce:

```bash
git checkout f9ad999dcd082f48c1305525995ad4bac3cdd21f
uv run python scripts/issue531_base_prior_reanalysis.py
# writes eval_results/issue_478/base_prior_reanalysis/{tidy.parquet, summary.json}
# writes figures/issue_478/base_prior_reanalysis/{shift, absolute_trained}_vs_base_prior.{png,pdf,meta.json}
# add --limit-cells 4 for a fast smoke pass, --include-arm to add the 12 decomposition cells

# Logit/probability follow-up (on main at f11eefa91aa3876faaedba768bb906e6bf783201):
uv run python scripts/issue531_probability_space_plots.py     # free, from tidy.parquet
bash scripts/issue531_logit_rescore_launch.sh 2 16            # on a 2-GPU pod; per-cell JSONs, idempotent
uv run python scripts/issue531_logit_plots.py                 # merge + stats + 3 logit figures
```

**Scope caveats:**
- Single base model (Qwen-2.5-7B); marker-specific (` ※`). Do not generalize across markers without re-running.
- 2 seeds per cell. The persona-cluster bootstrap captures the cross-persona variance, which is the dominant source of uncertainty here, but the seed-direction CI is narrower than a many-seed run would give.
- 80 trained runs from 40 CORE cells × 2 seeds, out of 92 total cells in #478 — the 12 ARM cells are excluded by design (different marker per source within an arm).
- The plan asked for partial Spearman on the shift DV only; absolute-trained log-prob wasn't in the plan. The shift result (partial ρ = −0.48) is the primary the plan called out; the absolute-trained partial (+0.74) is a secondary cut added during analysis that turned out to be more informative. Treat the absolute-trained read as the surprise finding, not the confirmed prediction.
- This analysis does NOT independently distinguish a propensity mechanism (persona-level pre-training preference carried through training) from a fixed compressed-coupling read (training applies a sub-unit affine map to base prior, independent of propensity). Both produce the observed positive absolute-trained partial AND the negative shift partial.
- The logit-space cross-check was delivered by the follow-up re-scoring pass (the parent run stored only resolved log-probabilities). The trained side reconstructs the original bf16-merged scoring via PEFT runtime adapter application — equivalent up to bf16 rounding, quantified at worst-case 0.33 nats MAE / 0.996 Spearman per cell against the stored values. The logit conclusions inherit that small reconstruction noise.
- The −0.13 logit-shift partial vs −0.48 log-prob-shift partial gap is an observation about which representation carries the negative; this analysis does not identify WHY the per-context normalizer component correlates with the shift (mean Δlog Z = −2.73 nats; candidate mechanisms — EOS reallocation under the contrastive negatives, overall logit-scale change under LoRA — are untested here).

**Follow-ups:**
- ~~One-off forward-pass to extract raw logits + `log Z` and report the logit-space partial alongside log P~~ — DONE (the logit/probability findings above; it ran over ALL 80 runs, not a subset, and turned out headline-relevant: the logit-vs-logit shift partial is −0.13, not −0.48).
- Sensitivity: re-fit the partials with ARM cells included via `--include-arm`, as a robustness supplement (cost_class: free-analysis, headline_affecting: no).
- Explain the Δlog Z = −2.73 nats normalizer drop: decompose log Z movement at the slot into EOS vs marker vs rest-of-vocab contributions from the stored per-cell logits (cost_class: free-analysis, headline_affecting: no — the per-cell JSONs already carry z(EOS), so the EOS share is computable without new GPU work).
- Add absolute-trained log-prob as a standard secondary DV in future marker analyses so rank-preservation reads aren't accidentally hidden behind shift's −base term.
- Disentangle propensity from compressed coupling: design a follow-up where the model's pre-training preference for the marker per-persona is held fixed while training compression is varied (e.g. via LoRA rank or LR sweeps that change the effective affine slope), and test whether the absolute-trained partial tracks the slope change (compressed-coupling prediction) or stays anchored to the base prior (propensity prediction) (cost_class: needs-gpu, headline_affecting: yes).

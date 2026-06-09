---
title: A 320-predictor bake-off finds competing metrics converge within ~0.02 CV R²;
  no robust win over last-token cosine (LOW confidence)
kind: analysis
tags: []
created_at: '2026-06-05T09:49:22Z'
has_clean_result: true
parent_id: 474
classification: useful
promoted_at: '2026-06-09T21:43:03Z'
---
# A 320-predictor bake-off finds competing metrics converge within ~0.02 CV R²; no robust win over last-token cosine (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I swept 320 base-model persona-distance predictors looking for something dramatically better than the last-token cosine #474 used, and there isn't one — seven metric families cluster within a narrow CV band and no predictor reliably beats cosine across loc-arm checkpoints.

**Takeaways.**
- On the cleanest loc-arm checkpoint, seven metric families all land between CV R² = 0.513 and 0.536 (a 0.023-wide band), and the top family beats the #474 incumbent by only 0.045.
- About 60% of that gap to the #474 incumbent closes by ONE change: moving the cosine probe from the last prompt token to the mean over the model's own response tokens. The extraction point moved the needle more than the metric did.
- The "winner" on the cleanest cell drops to rank ≈ 20 on the other three loc checkpoints; on those cells the chosen winner and the #474 incumbent tie within 0.001-0.013 CV R², and a different predictor (last-prompt L27 Δ-spectrum) tops both by 0.06-0.07.
- The whole positive-arm (4 of 8 cells) is unusable here — 78-99% of pairs have the trained model emitting the marker with probability ≈ 1, so ΔG has no dynamic range left to predict.

**How this updates me.** I no longer expect a dramatic improvement over last-token cosine on this substrate; cloud and centroid metrics are interchangeable at this level of training. The most actionable lesson for follow-up work is the extraction-point one — switch the cosine probe to the mean-response position before adding metric machinery. What would change my mind: replicating the loc-arm rankings across multiple training seeds and on a substrate with more non-saturated cells (less-trained anchors).

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

[#474](https://eps.superkaiba.com/tasks/474) measured on-policy marker transfer (trained − base log P( ※ )) across 240 ordered (source, target) transformation pairs for two training arms × four epochs. The headline base-model predictor of transfer was last-prompt-token cosine of mean activation at layer 21, reaching ρ ≈ −0.57 on the non-stylized panel of the cleanest cell (using the convention that more-negative cosine-of-mean → larger predicted transfer; [#474](https://eps.superkaiba.com/tasks/474) originally reported magnitude in cosine-distance convention with opposite sign). That cosine is a single-token centroid metric on one position — it compresses each transformation to one vector. Two natural questions follow: does a **richer extraction point** (the assistant's own response, where transfer actually happens) carry more signal, and does a **paired-cloud metric** (one that uses the full per-prompt activation cloud, not just the centroid) beat the centroid?

I ran the bake-off to answer both. The goal was to find — by transparent search over a fixed grid — the single best base-model predictor of [#474](https://eps.superkaiba.com/tasks/474)'s already-measured ΔG, and to report it with the multiple-comparisons honesty that a 320-predictor search requires.

### What I ran

A grid search over (extraction point × residual-stream layer × distance metric × variant), regressed against the parent transfer matrices. No retraining; just base-model forward passes plus the existing ΔG matrices.

**Grid.** 3 extraction points × 8 layers × 9 metrics × {raw, centered} variants, replicated across 8 training-checkpoint cells (arm × epoch). Each cell yields up to 496 predictor rows; 320 of those make it to the full-panel (n = 240) regression after dropping the small-subpanel system-tail extraction rows and the N/A combinations (cloud metrics on the single-vector extraction point).

| Axis | Values |
|---|---|
| Extraction point | system-tail (`end_of_system`, one vector per transformation — diagnostic only) · last-prompt (`last_prompt`, token after the user question; one vector per (transformation, question) → a cloud) · mean-response (`mean_response`, mean-pool over assistant response tokens; cloud) |
| Layer | 0, 5, 7, 11, 14, 15, 21, 27 |
| Metric | cosine of mean · Euclidean of mean · Mahalanobis (per-cloud and pooled-context) · RBF-MMD · C2ST classifier AUC · Δ-spectrum {coherence, mean_norm, effective_dim} · Gaussian KL · Wasserstein-2 |
| Variant | raw · prompt-centered (subtract per-prompt mean across transformations) |
| Cells | arm ∈ {positive-arm, loc-arm} × epoch ∈ {1, 2, 3, 5} — eight training checkpoints |

**Selection rule.** Headline predictor is the row with highest **leave-one-context-out CV R²** predicting `length-partialled rank correlation (predictor, ΔG)` on the full 240-pair panel. The length partial controls for prompt length being a nuisance that correlates with both activation geometry and the marker emission rate; without it, a predictor that happens to track length alone could pass selection. Reported alongside: the same length-partialled rank correlation against ΔG and against the base-prior-safe `g_logprob` (trained log-prob), on both the full-240 and non-stylized-156 panels. Single seed 42 throughout — load-bearing caveat.

**Cross-check.** Every layer of the re-extracted last-prompt cosine matrix matches the established cosine baseline within 0.0022 (tolerance 0.003) — the bake-off rig reproduces the established cosine baseline exactly, so any difference vs last-prompt cosine is a real metric-space difference, not extraction drift.

<details open>
<summary>Probe prompts used for activation extraction (2 cherry-picked for illustration out of 50; full pool at the link)</summary>

The 50 probes are the established predictor set — the SAME questions the parent run used as base-model evaluation inputs, so all distances live on a matched question distribution. They are mixed-domain (capabilities, opinion, neutral chat, hypothetical scenarios). The extraction wraps each into the chat template for each transformation; layer activations are read at the chosen extraction point. Full 50-probe pool: `eval_results/issue_406/predictor_inputs.json`.

Two probes, cherry-picked for illustration:

```
Probe 1:  "What is the most important challenge facing the world today, and why?"
Probe 2:  "If you could change one thing about how humans communicate, what would it be?"
```

</details>

### Findings

#### The extraction point moves the needle more than the metric

The cleanest cell in the grid is loc-arm epoch 1 — the only one where every metric has dynamic range to push against. On this cell, family-best CV R² looks like a tight band, not a leaderboard with a clear winner:

![Bar chart: best CV R² per metric family on loc-arm epoch 1. Δ-spectrum coherence at mean-response L21 leads at 0.536 (blue bar); six other families sit between 0.513 and 0.534; cosine of mean at mean-response L21 (red bar) at 0.519; the established last-prompt L21 cosine baseline sits at 0.491 (orange dashed line).](https://raw.githubusercontent.com/superkaiba/explore-persona-space/904d30968f84393a80ea64dfe6d2ac3f9876b0a9/figures/issue_493/family_best_cv_bar.png)

> **Figure.** *Seven metric families converge inside a 0.023-wide CV R² band (0.513-0.536); the chosen winner beats the established last-prompt cosine baseline (orange dashed line at 0.491) by 0.045, but beats the next family by only 0.002.* Bar = best variant per family across (extraction × layer × raw/centered), CV R² is leave-one-context-out on the n = 240 full panel. Highlighted: the chosen winner (Δ-spectrum coherence, blue) and the family-best cosine variant (red, mean-response L21).

The chosen winner is mean-response L21 Δ-spectrum coherence (centered) at CV R² = 0.536. The runner-up trio — RBF-MMD, Δ-spectrum mean-norm, and Euclidean of mean, all at last-prompt L27 — sits within 0.003 CV R² of the winner. On a 320-row search, a 0.002 lead over the next family is well within search noise.

The mentor-facing read is the load-bearing answer to the parent question: **most of the gap to the established cosine baseline is closed by moving the cosine probe, not by changing the metric.** Switching the cosine probe from last-prompt L21 (CV R² = 0.491) to mean-response L21 (CV R² = 0.519) closes 0.028 of the 0.045 winner-over-baseline gap — about 61% — with no change to the metric. The remaining ~40% then needs cloud-aware machinery (Δ-spectrum / MMD / Euclidean), and that machinery converges across families to roughly the same number.

Length-partial p-values on the selected winner row are descriptive — they describe the SELECTED row's fit, not an inferential test on a population of predictors (the CV ranking IS the selection evidence here). I searched 320 valid predictor rows on the cleanest cell; CV-selection guards against over-fitting any single ρ, but it cannot manufacture a margin where there isn't one.

#### Coherence vs ΔG is not just two-cluster leverage

The winner's headline ρ of −0.747 on the full 240-pair panel is partly a clean cluster split. Pairs that include one of the three stylized personas (A3 pirate, A4 comedian, A5 villain — n = 84 ordered pairs) sit at high coherence (mean ≈ 0.61) and low transfer (mean ΔG ≈ 8.1 nats); non-stylized pairs (n = 156) sit at low coherence (mean ≈ 0.16) and high transfer (mean ΔG ≈ 14.9). Two well-separated clouds will give a strong correlation even if neither cloud has any internal structure.

![Scatter: Δ-spectrum coherence (x) vs marker transfer ΔG (y) on loc-arm epoch 1. Two visually separated clouds — coral (stylized-touching pairs, ρ = -0.44) on the right at high coherence; blue (non-stylized only, ρ = -0.51, p = 1×10⁻¹¹) on the left at low coherence. Black line: quintile means within the non-stylized cloud, falling monotonically from 18.1 down to 12.6 nats as coherence rises from 0.06 to 0.35.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/904d30968f84393a80ea64dfe6d2ac3f9876b0a9/figures/issue_493/winner_scatter_within_nonstylized.png)

> **Figure.** *Within the non-stylized cloud alone, the coherence → ΔG gradient survives: quintile means drop from 18.1 → 12.6 nats.* Each dot is one ordered (source → target) transformation pair on loc-arm epoch 1. Coral = at least one of the pair is A3/A4/A5; blue = neither. The black line connects mean ΔG within each quintile of within-non-stylized coherence (error bars = SE). The Spearman on the non-stylized panel alone is ρ = −0.51 (p = 1×10⁻¹¹, n = 156); on the stylized-touching cluster alone ρ = −0.44 (n = 84). The full-panel ρ = −0.75 inherits genuine within-cluster signal as well as the visible cluster offset.

The within-non-stylized gradient is monotonic across all five quintiles and statistically clean. The length-partialled rank correlation on the non-stylized panel is ρ = −0.513 (p = 7×10⁻¹², n = 156), nearly identical to the raw Spearman of −0.510 above — so the gradient survives the length control.

The claim "Δ-spectrum coherence tracks marker transfer" therefore survives the cluster-leverage check on this cell. The mechanism (plain reading: *coherence measures how consistently the two contexts move activations in the same direction across the probe set; pairs that displace activations more coherently transfer the marker less*) is suggestive but not established — the negative sign is robust, the causal story is not.

#### The chosen winner does not generalise — and the substrate itself is mostly unusable

The loc-arm has four checkpoints (epochs 1, 2, 3, 5) — only the loc-arm gives usable predictor signal at all (the positive-arm is saturated; covered below). The chosen winner from loc-arm epoch 1 is rank 1 only on that single cell. At the other three loc cells it drops out of the top 20 — AND its CV R² is within 0.001-0.013 of the established cosine baseline there, so the choice between them is statistically meaningless on three of four loc cells.

What IS consistent across loc cells is a DIFFERENT predictor: the per-epoch top-1 on loc-arm epochs 2/3/5 is always a last-prompt L27 Δ-spectrum row (coherence on ep 2, mean-norm on ep 3 and ep 5). The extraction point and layer stay fixed across these three cells; only the Δ-spectrum sub-predictor varies. On ep 3 and ep 5 this candidate IS the per-epoch top-1; on ep 1 it ties the runner-up trio at 0.5333 (rank 3-tied, within 0.003 of the chosen winner); on ep 2 it sits at 0.3576, 0.002 below the per-epoch top-1 (also a last-prompt L27 Δ-spectrum coherence row at 0.3599).

![Grouped bar chart: 4 loc-arm epochs × 4 series. Green = per-epoch top-1 (always last-prompt L27 Δ-spectrum on ep 2/3/5). Blue = chosen winner (mean-response L21 Δ-spec coherence) — leads on ep 1 only, ties incumbent on ep 2/3/5. Purple = cross-cell candidate (last-prompt L27 Δ-spec mean-norm) — within 0.003 of top-1 on every cell. Red = established last-prompt L21 cosine baseline.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/904d30968f84393a80ea64dfe6d2ac3f9876b0a9/figures/issue_493/cross_cell_robustness.png)

> **Figure.** *The chosen winner ties the established cosine baseline within 0.001-0.013 CV R² on three of four loc cells; a last-prompt L27 Δ-spectrum predictor matches or tops the per-epoch top-1 on every cell.* Green = the CV-selected top-1 predictor at each epoch; blue = the chosen winner from loc-arm epoch 1 (mean-response L21 Δ-spectrum coherence); purple = the cross-cell candidate (last-prompt L27 Δ-spectrum mean-norm); red = the established last-prompt L21 cosine baseline. Positive-arm checkpoints are omitted because 78-99% of their pairs sit at the marker-emission ceiling — predictor information collapses there (covered below).

The per-epoch top-1 CV R² falls from 0.54 at epoch 1 to about 0.40 at epoch 5 — the predictability of ΔG from base-model geometry weakens as training progresses. This is NOT saturation: per-cell saturation fractions on the loc arm are loc-arm epoch 1 = 0.0%, epoch 2 = 1.3% (non-stylized) / 0.8% (full), epoch 3 = 0.0%, epoch 5 = 0.0% — far from the marker-emission ceiling at every epoch. The mechanism is instead **shrinking ΔG dynamic range**: the non-stylized panel's ΔG standard deviation falls from 4.59 → 4.93 → 3.95 → 3.64 nats across loc epochs 1 → 2 → 3 → 5 (and IQR from 7.19 → 6.32 → 5.55 → 4.98), so there is less variance for any predictor to explain even though no single cell is pinned at the ceiling.

On ep 2/3/5 the chosen winner and the cosine baseline are within 0.001-0.013 CV R² of each other (epoch 2: 0.297 vs 0.299; epoch 3: 0.308 vs 0.321; epoch 5: 0.330 vs 0.330) — three of four cells could be summarised as "predictor choice is in search noise". The last-prompt L27 Δ-spectrum candidate tops both by 0.06-0.07 on those three cells (0.36-0.40 vs 0.30-0.33). Combined with the in-band convergence above, the bake-off rules out a dramatically-better predictor than last-token cosine on this substrate — the actionable carry-forward is the consistent last-prompt L27 Δ-spectrum family, not the cleanest-cell chosen winner.

The other half of the substrate — the positive-arm cells — is dynamically dead: the positive arm trained the marker into the source persona without contrastive negatives. The result is that 78% of positive-arm epoch 1 pairs and 99.6% of positive-arm epoch 2/3/5 pairs have the trained model emitting the marker with probability ≈ 1 (per-cell saturation fractions at the g_logprob ≥ −0.1 ceiling rule: positive-arm epoch 1 = 0.78 full / 0.99 non-stylized; positive-arm epoch 2/3/5 ≈ 1.00 on both). At that ceiling the on-policy DV is mechanically near-constant — the trained log P( ※ ) is essentially 0 everywhere, so ΔG is dominated by base log P rather than by anything about transfer.

Diagnostically, this shows up as predictor sign-flips: the chosen winner's non-stylized Spearman on positive-arm epoch 1 is **+0.376** (positive!), opposite the loc-arm sign, and the full-panel ρ falls to −0.44 (vs −0.75 on the cleanest cell). No predictor in the grid recovers anything meaningful on the positive-arm cells. That is not a deficiency of the predictors — it is the measurement-validity gate firing on a saturated DV; any cross-cell summary in this clean-result restricts to the loc arm.

The system-tail extraction point produces a single vector per transformation (input-independent), so only centroid metrics apply and only the A-class (5 personas → 20 ordered pairs) subpanel has any data. On that subpanel the best CV R² is −0.3999 — uninformative either way. The plan registered system-tail as diagnostic-only specifically for this reason; reporting here confirms it carries no claim.

## Reproducibility

**Parameters:**

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Eval (DV source) | [#474](https://eps.superkaiba.com/tasks/474) cross-eval `G_logprob_matrix.json` for arm ∈ {pos, loc} × epoch ∈ {1, 2, 3, 5}; on-policy ΔG = trained − base log P( ※ ) (token id 83399) at the post-response slot |
| Transformation set | 16 transformations from `src/explore_persona_space/experiments/i406_conditions.py` — A1-A5 (personas, A3/A4/A5 stylized), B1-B5 (query-wrap), C1 (chat template), D1-D5 (register rewrite); 240 ordered off-diagonal pairs, 156 non-stylized |
| Probe set | 50 questions from `eval_results/issue_406/predictor_inputs.json` |
| Extraction | residual-stream activations via forward hooks at layers {0, 5, 7, 11, 14, 15, 21, 27}; three extraction points (`end_of_system`, `last_prompt`, `mean_response`); max response tokens = 512 for `mean_response` |
| Metrics | cosine of mean · Euclidean of mean · Mahalanobis (per-cloud, pooled-ctx) · RBF-MMD (median-heuristic bandwidth) · C2ST linear-probe AUC · Δ-spectrum (PCA-k=16 on per-question displacements; reports coherence / mean_norm / effective_dim) · Gaussian symmetric-KL · Wasserstein-2; raw + prompt-centered |
| Regression | length-partial Spearman (rank-residualise on log prompt_tokens) on full-16 (n = 240) and non-stylized (n = 156) panels; predicts ΔG and `g_logprob` (base-prior-safe); leave-one-context-out CV R² as the selection criterion |
| Sample | single seed 42 throughout (inherited from [#474](https://eps.superkaiba.com/tasks/474)) — load-bearing caveat |
| Hardware | 1× H100 (intent `eval`), pod `epm-issue-493`, ~95 min wall time |
| Predictors searched | 496 grid rows per cell × 8 cells = 3,968 entries; 320 full-panel-regressable rows per cell |

**Artifacts:**

- Bake-off grid + winner record: `eval_results/issue_493/bakeoff/bakeoff_grid.json` (commit `904d30968`)
- Per-cell regressions: `eval_results/issue_493/bakeoff/regression/{pos,loc}_ep{1,2,3,5}.json` (8 files)
- Per-(extraction × layer × metric × variant) metric records: `eval_results/issue_493/bakeoff/metrics/*.json` (464 files; each has `matrix` or `matrices`, ρ_full_{deltag,glogp}, ρ_nonstylized_{deltag,glogp}, cv_*)
- Cosine cross-check vs [#406](https://eps.superkaiba.com/tasks/406): `eval_results/issue_493/bakeoff/cosine_cross_check.json` (max abs diff 0.00217 at L27 vs tol 0.003, PASS all layers)
- Smoke-run digest: `eval_results/issue_493/bakeoff/smoke_digest.json`
- Run metadata: `eval_results/issue_493/bakeoff/meta.json` (git_sha, env, args)
- Figures: `figures/issue_493/{family_best_cv_bar, winner_scatter_within_nonstylized, cross_cell_robustness, winner_scatter_vs_deltaG, metric_layer_grid_heatmap}.{png, pdf, meta.json}`
- ΔG source (parent [#474](https://eps.superkaiba.com/tasks/474)): `eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json` (8 files)
- Raw activations not persisted (re-extractable from base model in ≈ 90 min on 1× H100)

**Compute:** 1× H100 on RunPod pod `epm-issue-493`, ≈ 95 minutes wall time, pod terminated post-run; figure regeneration runs on the VM in ≈ 30 s.

**Code:**

- Driver: `scripts/issue493_extraction_metric_bakeoff.py` (single entry point; CLI flags for `--phase {smoke, hooks_check, full, all}`, `--extraction-points`, `--layers`, `--metrics`)
- Analyzer figures: `scripts/issue493_make_clean_figures.py` (regenerates family_best_cv_bar, winner_scatter_within_nonstylized, cross_cell_robustness)
- Canonical metric definitions: `.claude/rules/persona-distance-metrics.md`
- Extraction-template reference: `scripts/recompute_predictors_i415.py`
- Length-partial Spearman + LOCO-CV reference: `scripts/i474_cosine_followup.py`
- Run commit: `a6ce330f1bc9fe6c285ecf7cb974ea8fd2534851` (issue branch `issue-493`); figures + body on `main` at `904d30968f84393a80ea64dfe6d2ac3f9876b0a9`

Reproduce:

```bash
uv run python scripts/issue493_extraction_metric_bakeoff.py \
    --phase all \
    --extraction-points end_of_system last_prompt mean_response \
    --layers 0 5 7 11 14 15 21 27 \
    --metrics cosine euclidean mahal mahal_pooled_ctx mmd c2st delta_spec gauss_kl wass2 \
    --arms pos loc --epochs 1 2 3 5 \
    --n-probes 50 --pca-k 16 --max-response-tokens 512
```

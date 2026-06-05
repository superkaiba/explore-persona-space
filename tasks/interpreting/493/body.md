---
title: 'Extraction-point × metric bake-off: best base-model predictor of #474 on-policy
  marker transfer'
kind: analysis
tags: []
created_at: '2026-06-05T09:49:22Z'
has_clean_result: false
parent_id: 474
---
# A paired-cloud predictor edges out last-token cosine by 0.05 CV R², but the lead is loc_ep1-specific (LOW confidence)

<!-- clean-result-v2 -->

## Human TL;DR

**Headline.** I swept 320 base-model persona-distance predictors looking for something that beats the last-token cosine #474 used; the paired-cloud Δ-spectrum at mean-response activations wins on the cleanest cell, but only by a hair and only on that cell.

**Takeaways.**
- On the cleanest training checkpoint (loc-arm, epoch 1), seven different metric families all cluster within ~0.05 CV R² of each other (range 0.49 - 0.54). There's no single "right" predictor.
- The winner's lead over last-token cosine vanishes on the other three loc epochs: there it falls to rank ~20 and a different predictor (last_prompt · L27 · Δ-spectrum) takes the top, never by more than 0.01.
- The full-panel ρ = −0.75 partly reflects a clean two-cluster split (pirate/comedian/villain personas sit at high coherence + low transfer; everyone else sits at low coherence + high transfer), but the within-non-stylized gradient is real (quintile means fall monotonically 18 → 12 nats over the n=156 panel).
- The whole positive-arm (pos_ep1/2/3/5) is unusable here: 78%-99% of cells have the trained model emitting the marker with probability ≈ 1, so ΔG carries no predictor signal at all.

**How this updates me.** I no longer expect a dramatic improvement over last-token cosine for predicting on-policy marker transfer at this substrate. Cloud metrics are competitive, not transformative. What would change my mind: replicating the loc_ep1 winner under multiple training seeds and on a substrate with more non-saturated cells.

*(First pass — Thomas refines this before sending to the mentor.)*

## TL;DR

### Motivation

[#474](https://eps.superkaiba.com/tasks/474) measured on-policy marker transfer (`trained − base log P( ※ )`) across 240 ordered (source, target) transformation pairs for two training arms × four epochs. The headline base-model predictor for transfer was last-prompt-token **cosine of mean activation** at layer 21: ρ ≈ +0.57 on the non-stylized 156-pair panel of the cleanest cell. That cosine is a centroid metric on a single token position — it compresses each transformation to one vector. Two natural questions follow: does a **richer extraction point** (the assistant's own response, where transfer actually happens) carry more signal, and does a **paired-cloud metric** (one that uses the full per-prompt activation cloud, not just the centroid) beat the centroid?

I ran the bake-off to answer both. The goal was to find — by transparent search over a fixed grid — the single best base-model predictor of #474's already-measured ΔG, and to report it with the multiple-comparisons honesty that a 320-predictor search requires.

### What I ran

A grid search over (extraction point × residual-stream layer × distance metric × variant), regressed against #474's measured ΔG. No retraining; just base-model forward passes plus the existing ΔG matrices.

**Grid.** 3 extraction points × 8 layers × 9 metrics × {raw, centered} variants, replicated across 8 cells (arm × epoch). Each cell yields up to 496 predictor rows; 320 of those make it to the full-panel (n=240) regression after dropping the small-subpanel `end_of_system` rows and the N/A combinations (cloud metrics on the single-vector extraction point).

| Axis | Values |
|---|---|
| Extraction point | `end_of_system` (system-prompt tail only; one vector per transformation — diagnostic only) · `last_prompt` (token after the user question; one vector per (transformation, question) → a cloud) · `mean_response` (mean-pool over assistant response tokens; cloud) |
| Layer | 0, 5, 7, 11, 14, 15, 21, 27 |
| Metric | cosine of mean · Euclidean of mean · Mahalanobis (per-cloud and pooled-context) · RBF-MMD · C2ST classifier AUC · Δ-spectrum {coherence, mean_norm, effective_dim} · Gaussian KL · Wasserstein-2 |
| Variant | raw · prompt-centered (subtract per-prompt mean across transformations) |
| Cells | arm ∈ {pos, loc} × epoch ∈ {1, 2, 3, 5} — eight training checkpoints from [#474](https://eps.superkaiba.com/tasks/474) |

**Selection rule.** Headline predictor is the row with highest **leave-one-context-out CV R²** predicting `length-partial Spearman(predictor, ΔG)` on the full 240-pair panel. The reported scores include the length-partial Spearman against ΔG and against the base-prior-safe `g_logprob` (trained log-prob), on both the full-240 and non-stylized-156 panels. Single seed 42 throughout — load-bearing caveat.

**Cross-check.** Every layer of the re-extracted last-prompt cosine matrix matches the [#406](https://eps.superkaiba.com/tasks/406) cosine matrix within 0.0022 (tolerance 0.003) — the bake-off rig reproduces the established cosine baseline exactly, so any difference vs. last-prompt cosine is a real metric-space difference, not extraction drift.

<details open>
<summary>Probe prompts used for activation extraction (2 cherry-picked for illustration out of 50; full pool at the link)</summary>

The 50 probes are [#406](https://eps.superkaiba.com/tasks/406)'s predictor set — the SAME questions [#474](https://eps.superkaiba.com/tasks/474) used as base-model evaluation inputs, so all distances live on a matched question distribution. They are mixed-domain (capabilities, opinion, neutral chat, hypothetical scenarios). The extraction wraps each into the chat template for each transformation; layer activations are read at the chosen extraction point. Full 50-probe pool: `eval_results/issue_406/predictor_inputs.json`.

Two probes, cherry-picked for illustration:

```
Probe 1:  "What is the most important challenge facing the world today, and why?"
Probe 2:  "If you could change one thing about how humans communicate, what would it be?"
```

</details>

### Findings

#### A thin band of competitive predictors, not a clear winner

The headline cell `loc_ep1` is the only one in the grid where every metric has dynamic range to push against (saturation is 0.00 on both the full and non-stylized panels — the trained model has not collapsed `P( ※ )` to ceiling yet). On that cell, family-best CV R² ranks like this:

![Bar chart: best CV R² per metric family on loc_ep1. Δ-spectrum coherence at mean-response L21 leads at 0.536; six other families sit between 0.513 and 0.534; cosine of mean (best variant: mean_response L21) at 0.519; the actual #474 incumbent (last_prompt L21 cosine) sits at 0.491.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2116eec753a18d25dcc07e9b35a31fe32e80a967/figures/issue_493/family_best_cv_bar.png)

> **Figure.** *Seven metric families on `loc_ep1` cluster inside a 0.05-wide CV R² band; the chosen winner beats the #474 cosine incumbent (dashed line) by +0.045.* Bar = best variant per family across (extraction × layer × raw/centered), CV R² is leave-one-context-out on the n=240 full panel. Highlighted: the winner row (Δ-spectrum coherence, blue) and the actual #474 incumbent (last_prompt · L21 cosine, orange). Note that the family-best cosine variant is `mean_response · L21` at 0.519 — moving the cosine probe from `last_prompt` to `mean_response` already closes most of the gap to the winner.

The chosen winner is the row with the highest CV: **`mean_response · L21 · Δ-spectrum (coherence, centered)`** at CV R² = 0.536, with length-partial ρ(ΔG) = −0.747 (p = 5×10⁻⁴⁴) on the full panel and ρ(ΔG) = −0.513 (p = 7×10⁻¹²) on the non-stylized 156-pair panel. But the runner-up trio (RBF-MMD, Δ-spectrum mean-norm, Euclidean of mean — all at `last_prompt · L27`) sits within 0.003 CV R², and the family-best cosine sits at 0.519. **The honest framing is "all competitive cloud and centroid metrics converge to CV R² ≈ 0.52–0.54 at upper-mid layers", not "a paired-cloud metric dethrones cosine".**

I searched 320 valid predictor rows on `loc_ep1`. CV-selection guards against over-fitting any single ρ, but it cannot manufacture a margin where there isn't one. A +0.017 CV R² lead over the next family in a 320-row search is essentially within the noise of the search itself.

#### Coherence vs ΔG is not just two-cluster leverage

The winner's headline ρ of −0.747 on the full 240-pair panel is partly a clean cluster split: pairs that include one of the three stylized personas (A3 pirate, A4 comedian, A5 villain — n=84 ordered pairs) sit at high coherence (mean ≈ 0.61) and low transfer (mean ΔG ≈ 8.1 nats), while non-stylized pairs (n=156) sit at low coherence (mean ≈ 0.16) and high transfer (mean ΔG ≈ 14.9). Two well-separated clouds will give a strong correlation even if neither cloud has any internal structure.

![Scatter: Δ-spectrum coherence (x) vs marker transfer ΔG (y) for loc_ep1. Two visually separated clouds — orange (stylized-involving pairs, ρ = −0.44) on the right, blue (non-stylized only, ρ = −0.51, p = 1e-11) on the left. Black line: quintile means within the non-stylized cloud, falling monotonically from 18 down to 12 nats as coherence rises from 0.06 to 0.34.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2116eec753a18d25dcc07e9b35a31fe32e80a967/figures/issue_493/winner_scatter_within_nonstylized.png)

> **Figure.** *Within the non-stylized cloud alone, the coherence → ΔG gradient survives: quintile means drop from 18.0 → 12.4 nats.* Each dot is one ordered (source → target) transformation pair on `loc_ep1`. Orange = at least one of the pair is A3/A4/A5; blue = neither. The black line connects mean ΔG within each quintile of within-non-stylized coherence (error bars = SE). The Spearman on the non-stylized panel alone is ρ = −0.51 (p = 1×10⁻¹¹, n=156); on the stylized-touching cluster alone ρ = −0.44 (n=84). The full-panel ρ = −0.75 inherits genuine within-cluster signal as well as the visible cluster offset.

The within-non-stylized gradient is monotonic across five quintiles and statistically clean. The claim "Δ-spectrum coherence tracks marker transfer" therefore survives the cluster-leverage check on this cell. The mechanism (plain reading: *coherence measures how consistently the two contexts move activations in the same direction across the probe set; pairs that displace activations more coherently transfer the marker less*) is suggestive but not established — the negative sign is robust, the causal story is not.

#### The winner does not generalise across loc epochs

The loc-arm has four checkpoints (epochs 1, 2, 3, 5) — only the loc-arm gives usable predictor signal at all (the pos-arm is saturated; see below). The chosen winner is rank 1 only at `loc_ep1`. At the other three loc cells it drops out of the top 20.

![Grouped bar chart: 4 loc epochs × 3 predictors. The top-1 CV-selected predictor at each epoch (red) achieves CV R² ≈ 0.54 / 0.36 / 0.39 / 0.40. The chosen winner (Δ-spec coherence at mean_response L21, blue) drops from 0.54 → 0.30 / 0.31 / 0.33 across ep 2/3/5. The #474 incumbent cosine (orange) is nearly tied with the winner on ep 2/3/5.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/2116eec753a18d25dcc07e9b35a31fe32e80a967/figures/issue_493/cross_cell_robustness.png)

> **Figure.** *The loc_ep1 lead does not transfer.* Red = the CV-selected top-1 predictor at this specific epoch; blue = the chosen winner from `loc_ep1` (`mean_response · L21 · Δ-spec coherence`); orange = the [#474](https://eps.superkaiba.com/tasks/474) incumbent (`last_prompt · L21 · cosine`). On `loc_ep2/3/5` the top predictor shifts to `last_prompt · L27 · Δ-spectrum` (mean-norm or coherence depending on epoch), and the chosen winner family falls to rank ≈ 20. The pos-arm checkpoints are omitted because 78%–99% of their cells sit at the marker-emission ceiling (see next finding) — predictor information collapses there.

Two things matter here. **First**, the CV-selected top-1 at each loc epoch (red bars) drops from 0.54 at epoch 1 to about 0.40 at epoch 5 — the predictability of ΔG from base-model geometry weakens as training progresses (more pairs saturating toward the marker ceiling, less ΔG dynamic range). **Second**, on epochs 2/3/5 the chosen winner and the cosine incumbent are within 0.01 CV R² of each other — the choice between them is statistically meaningless on three of four loc cells.

Combined with the in-band convergence of Finding 1, this strongly suggests the "winner" should be read as "loc_ep1 best of a tight cluster", not as "the predictor base-model geometry was hiding from us".

#### The positive-arm cells are dynamically dead

The pos-arm trained the marker into the source persona without contrastive negatives. The result is that 78% of `pos_ep1` cells and 99.6% of `pos_ep2/3/5` cells have the trained model emitting the marker with probability ≈ 1 (per-cell saturation fractions: pos_ep1 0.78 full / 0.99 non-stylized; pos_ep2-5 ≈ 1.00 on both). At that ceiling the on-policy DV is mechanically near-constant — the trained `log P( ※ )` is essentially 0 everywhere, so ΔG is dominated by `base log P` rather than by anything about transfer.

Diagnostically, this shows up as predictor sign-flips: the winner's non-stylized Spearman on `pos_ep1` is **+0.376** (positive!), opposite the loc-arm sign, and the full-panel ρ falls to −0.44 (vs −0.75 on loc_ep1). The base-prior-safe trained-logp regression is similarly degraded. **No predictor in the grid recovers anything meaningful on the pos-arm cells.** That is not a deficiency of the predictors — it is the measurement-validity gate firing on a saturated DV. Any cross-cell summary in this clean-result restricts to the loc arm.

For the `end_of_system` extraction point: it produces a single vector per transformation (input-independent), so only centroid metrics apply and only the A-class (5 personas → 20 ordered pairs) subpanel has any data. On that subpanel the best CV R² is **−0.40** (cosine L11, raw) — uninformative. The plan registered `end_of_system` as diagnostic-only specifically for this reason; reporting here confirms it carries no claim.

## Reproducibility

**Parameters:**

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-7B-Instruct` |
| Eval (DV source) | [#474](https://eps.superkaiba.com/tasks/474) cross-eval `G_logprob_matrix.json` for arm ∈ {pos, loc} × epoch ∈ {1, 2, 3, 5}; on-policy ΔG = trained − base `log P( ※ )` (token id 83399) at the post-response slot |
| Transformation set | 16 transformations from `src/explore_persona_space/experiments/i406_conditions.py` — A1-A5 (personas, A3/A4/A5 stylized), B1-B5 (query-wrap), C1 (chat template), D1-D5 (register rewrite); 240 ordered off-diagonal pairs, 156 non-stylized |
| Probe set | 50 questions from `eval_results/issue_406/predictor_inputs.json` |
| Extraction | residual-stream activations via forward hooks at layers {0, 5, 7, 11, 14, 15, 21, 27}; three extraction points (`end_of_system`, `last_prompt`, `mean_response`); max response tokens = 512 for `mean_response` |
| Metrics | cosine of mean · Euclidean of mean · Mahalanobis (per-cloud, pooled-ctx) · RBF-MMD (median-heuristic bandwidth) · C2ST linear-probe AUC · Δ-spectrum (PCA-k=16 on per-question displacements; reports coherence / mean_norm / effective_dim) · Gaussian symmetric-KL · Wasserstein-2; raw + prompt-centered |
| Regression | length-partial Spearman (rank-residualise on log prompt_tokens) on full-16 (n=240) and non-stylized (n=156) panels; predicts ΔG and `g_logprob` (base-prior-safe); leave-one-context-out CV R² as the selection criterion |
| Sample | single seed 42 throughout (inherited from [#474](https://eps.superkaiba.com/tasks/474)) — load-bearing caveat |
| Hardware | 1× H100 (intent `eval`), pod `epm-issue-493`, ~95 min wall time |
| Predictors searched | 496 grid rows per cell × 8 cells = 3,968 entries; 320 full-panel-regressable rows per cell |

**Artifacts:**

- Bake-off grid + winner record: `eval_results/issue_493/bakeoff/bakeoff_grid.json` (commit `2116eec75`)
- Per-cell regressions: `eval_results/issue_493/bakeoff/regression/{pos,loc}_ep{1,2,3,5}.json` (8 files)
- Per-(extraction × layer × metric × variant) metric records: `eval_results/issue_493/bakeoff/metrics/*.json` (464 files; each has `matrix` or `matrices`, ρ_full_{deltag,glogp}, ρ_nonstylized_{deltag,glogp}, cv_*)
- Cosine cross-check vs [#406](https://eps.superkaiba.com/tasks/406): `eval_results/issue_493/bakeoff/cosine_cross_check.json` (max abs diff 0.00217 at L27 vs tol 0.003, PASS all layers)
- Smoke-run digest: `eval_results/issue_493/bakeoff/smoke_digest.json`
- Run metadata: `eval_results/issue_493/bakeoff/meta.json` (git_sha, env, args)
- Figures: `figures/issue_493/{family_best_cv_bar, winner_scatter_within_nonstylized, cross_cell_robustness, winner_scatter_vs_deltaG, metric_layer_grid_heatmap}.{png, pdf, meta.json}`
- ΔG source (parent [#474](https://eps.superkaiba.com/tasks/474)): `eval_results/issue_474/cross_eval/{arm}_ep{ep}/G_logprob_matrix.json` (8 files)
- Raw activations not persisted (re-extractable from base model in ≈ 90 min on 1× H100)

**Compute:** 1× H100 on RunPod pod `epm-issue-493`, ≈95 minutes wall time, pod terminated post-run; figure regeneration runs on the VM in ≈ 30 s.

**Code:**

- Driver: `scripts/issue493_extraction_metric_bakeoff.py` (single entry point; CLI flags for `--phase {smoke, hooks_check, full, all}`, `--extraction-points`, `--layers`, `--metrics`)
- Canonical metric definitions: `.claude/rules/persona-distance-metrics.md`
- Extraction-template reference: `scripts/recompute_predictors_i415.py`
- Length-partial Spearman + LOCO-CV reference: `scripts/i474_cosine_followup.py`
- Run commit: `a6ce330f1bc9fe6c285ecf7cb974ea8fd2534851` (issue branch `issue-493`); figures + body on `main` at `2116eec753a18d25dcc07e9b35a31fe32e80a967`

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

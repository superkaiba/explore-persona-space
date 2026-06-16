---
title: Centered cosine is not a stronger leakage predictor than raw — one of four
  banks resolves out-of-sample, and that one favors raw (LOW confidence)
kind: analysis
tags: []
created_at: '2026-06-15T22:40:29Z'
has_clean_result: true
parent_id: 536
origin_prompt: 'queue a centered-vs-raw predictive-skill comparison on the cached
  predictor panels (CV R²_raw vs CV R²_centered on the same cell): #536 settled which
  recipe is valid + that calls survive, but not which maximizes predictive skill;
  effect on skill goes both directions by bank (100-persona raw inflates 0.60->0.77;
  19-bank headline exists only centered -0.348 vs -0.037)'
---
# Centered cosine is not a stronger leakage predictor than raw — one of four banks resolves out-of-sample, and that one favors raw (LOW confidence)

<!-- clean-result-v3 -->

**Methodology:** [docs/methodology/issue_648.md](https://github.com/superkaiba/explore-persona-space/blob/3d809d40332eae0ddcd10167e407d845473ee98a/docs/methodology/issue_648.md) · [gist](https://gist.github.com/superkaiba/40a5bba5664579ea7b2aae4d1480916c)

## Takeaways

- Of 4 verdict-eligible banks, **only the 505 persona-vector bank resolves out-of-sample (ΔR² = −0.071, CI excludes 0)** — and it favors raw, not the canonical centered recipe.
- That sole determinate bank is the panel's **structural outlier** (only persona-vector, only layer-21, only multi-arm design), so the raw edge may be specific to that bank — hence LOW.
- The other 3 eligible banks straddle 0 (**+0.030, +0.017, −0.297**): no centered-favored determinate bank exists, so "centering is uniformly stronger" is falsified.
- The resolving 505 bank is also the **largest** (52 folds vs 5-35 elsewhere); centering's CV-hygiene was cleanest there yet **still** lost, ruling out a centering-leak artifact.
- On the 505 bank, centering **improves** in-sample fit (Δρ = +0.208, CI excludes 0) yet **hurts** held-out skill — the reverse of raw-inflation.
- On three mid-size banks, **both recipes score CV R² < 0** (worse than the train mean), so their in-sample correlations do not predict held-out leakage. Centering stays canonical on validity.

## What I ran

- **Why:** the parent metric-recipe audit ([#536](https://eps.superkaiba.com/tasks/536)) settled that mean-centering the centroid bank is the *valid* cosine recipe and that the published persona-distance calls survive the swap, but it never asked which recipe maximizes *predictive skill*. Its survival check pointed both ways by bank — on one bank raw inflates the headline correlation, on another the headline exists only under centering — so the skill question was genuinely open.
- **Design:** for each recoverable centroid bank from the parent audit, score the cosine-distance leakage predictor two ways — raw (un-centered) vs centered (global-mean-subtracted) — on the SAME cells, target, length residualization, and cross-validation folds. The single manipulated variable is the centering recipe.
- **Training:** n/a — no training; CPU-only re-analysis on cached centroid banks and leakage targets from the parent audit.
- **Eval:** primary DV is held-out cross-validation R² (leave-one-group-out, the bank's own natural held-out unit; centering mean refit on the train fold inside every split so no held-out persona enters its own predictor), differenced across recipes (ΔR² = centered − raw) with a paired bootstrap 95% CI (10,000 resamples) in original-group space. Secondary DV is the in-sample length-partialled Spearman difference Δρ, reported for continuity with the parent's survival check. 9 banks; the 2 with only 5 folds report their numbers but are excluded from the verdict.
- **Scope shrinkage:** of the 9 banks, only **4 are verdict-eligible** (more than 5 folds AND at least one recipe beats the train-mean baseline — equivalently: NOT both recipes fail). Two marker banks have only 5 folds (a 5-from-5 bootstrap is too coarse for a determinate call) and three mid-size banks have both recipes failing out-of-sample — all five are reported but mechanically excluded from the hypothesis verdict.

## Findings

### Only the 505 bank resolves — and centering is the worse predictor there

A bank has a determinate winner only when its paired-bootstrap ΔR² CI (centered − raw) excludes 0. One of 4 eligible banks clears that bar.

![Forest plot of held-out CV R-squared difference (centered minus raw) per bank, with paired-bootstrap 95% CI bars; only the 505 persona-vector bank has a filled marker with its CI fully left of zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/hero_forest_delta_cvR2.png)

> **Figure.** *Centering is not the stronger leakage predictor.* x = ΔR² (centered − raw), held-out CV, paired-bootstrap 95% CI. Filled = determinate (CI excludes 0); open = indistinguishable, low-fold, or both-recipes-fail. n = fold count per bank. Only the 505 bank (n=52) resolves, at ΔR² = −0.071 — raw-favored.

- The 505 bank gives ΔR² = −0.071 (CI excludes 0): centered predicts held-out leakage *worse* than raw. No eligible bank favors centering with a CI excluding 0, so "centering is uniformly stronger" is falsified.
- But the only determinate bank is the panel's structural outlier (only persona-vector, only layer-21, only multi-arm design), so the raw edge may be specific to that bank — the binding LOW-confidence caveat.
- It is the largest panel (52 folds), where centering's train-fold-refit leak is smallest, so its cleanest-CV shot still lost — ruling out a centering-leak artifact.

### In-sample fit and held-out skill disagree in sign — the opposite of raw-inflation

Raw-inflation predicts raw fits the sample better than it predicts held-out data. Each bank's in-sample Δρ against its held-out ΔR² tests that.

![Scatter of held-out CV R-squared difference against in-sample Spearman difference, both centered minus raw, one point per bank; the 505 bank sits at positive in-sample delta but negative held-out delta.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/insample_vs_cv_scatter.png)

> **Figure.** *In-sample and held-out deltas point opposite ways on the resolving bank.* x = in-sample Δρ (centered − raw); y = held-out ΔR² (centered − raw). Filled = verdict-eligible, open = excluded (low-fold or both-fail). On the 505 bank centering helps in-sample (Δρ = +0.208) yet hurts held-out (ΔR² = −0.071).

- On the 505 bank, centering *improves* in-sample fit (Δρ = +0.208, CI excludes 0) while *worsening* held-out skill — centered fits the sample better, raw generalizes better, the reverse of raw-inflation.
- Only the two low-fold marker banks show raw fitting the sample better (Δρ = −0.164, −0.232, CIs exclude 0) — the inflation candidates the parent flagged, but with 5 folds each they cannot carry a verdict.

### Three mid-size banks: both recipes fail out-of-sample entirely

The parent reported in-sample correlations on several small banks; whether those translate to held-out prediction is separate. On three mid-size banks, neither recipe does.

![Grouped bar chart of held-out CV R-squared (raw vs centered) for the three both-fail banks; the 24-persona marker, 24-persona logit, and 19-persona joint-leakage banks all have both bars below zero, each labeled with its held-out unit.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c745da0abc227b880b16fc0a94b19bac8f427607/figures/issue_648/paired_r2_raw_vs_centered.png)

> **Figure.** *On the three both-fail banks, both recipes score below the train-mean baseline.* Held-out CV R² (per-fold train-mean baseline) for the three banks excluded for both-recipes-failing; below 0 = worse than predicting the mean. Each bar labeled with its held-out unit (persona / source / bystander) and n. Raw (orange) and centered (blue) both fail on all three.

- All three land at CV R² < 0 (worse than the train mean) under both recipes. Their held-out units differ (leave-one-persona / -source / -bystander-out), so this is not a single-unit artifact.
- The exclusion is itself the finding: at the small-N end (8-24 held-out folds), the cosine-distance predictor does not generalize out-of-sample regardless of centering.

### Raw cosine is compressed, but compression does not pick the winner

Centering exists because raw cosine is squeezed into the anisotropy ridge. The compression is real and uniform on the three banks with this read.

![Three side-by-side histograms of off-diagonal cosine values, raw vs centered, for the 111-persona, 20-persona, and 505 banks; raw piles into 0.7-1.0 while centered spreads across -0.7 to 1.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/compression_offdiag.png)

> **Figure.** *Raw cosine collapses into the anisotropy ridge on all three banks with this read.* Off-diagonal cosine distributions: raw (orange) spans only ~0.7 to 1.0; centered (blue) spreads across ~−0.7 to 1.0. Available for 3 of 9 banks (the off-diagonal payload only persists for these families).

- Raw cosine compresses into ~0.7 to 1.0 on all three families — the degeneracy that makes centering valid.
- But the held-out winner does not follow compression: the raw-leaning (505, 20-persona) and flat (111-persona) banks are all equally compressed, so geometry does not explain the ΔR² signs.
- The raw-inflation mechanism is present in the geometry but does not drive the skill verdict.

## Data

### Trained on

n/a — no training in this task. This is a CPU-only re-analysis of centroid banks and leakage targets produced by prior `kind: experiment` tasks (the persona-distance predictor line; provenance links in Reproducibility).

### Evaluated with

Per recoverable bank, two inputs joined exactly as the parent audit joined them: a persisted base-model centroid bank (Qwen-2.5-7B, the layer each parent task used) and that bank's per-cell continuous leakage target (marker leakage rate, marker/logit shift, or fact-leakage delta, depending on the bank). The cosine-distance predictor is built under each centering recipe on the same cells; the target, the length covariate + residualization, the cross-validation fold map, and the bootstrap resample indices are held identical across recipes. Before any skill is read, each bank's raw recompute must reproduce the parent's published Spearman within tolerance — every bank passed (e.g. the 19-persona bank's centered headline reproduced at −0.348 = published −0.348; the core-11 marker subset at 0.5674 ≈ published 0.567).

The 9 banks, plain-English names, layer / bank type, and fold counts (first 9 of 9 rows — the complete set, not a sample; full per-bank metrics at the artifact link below):

<details>
<summary>Per-bank join + fold map (first 9 of 9 rows — complete set)</summary>

| Bank | Layer / bank type | Held-out unit | Folds | Verdict-eligible? |
|---|---|---|---|---|
| 100-persona marker bank | L20 / single-token marker | source | 5 | No — only 5 folds |
| core-11 marker subset | L20 / single-token marker | source | 5 | No — only 5 folds |
| 20-persona extraction bank | L20 / extraction-method | held persona | 8 | Yes |
| 111-persona marker bank | L20 / single-token marker | held persona | 35 | Yes |
| 111-persona dose-matched bank | L20 / single-token marker | persona | 11 | Yes |
| 24-persona marker bank | L15 / n24 marker | persona | 24 | No — both recipes fail |
| 24-persona logit bank | L15 / n24 logit-surface | source | 24 | No — both recipes fail |
| 19-persona joint-leakage bank | L20 / joint-leakage | bystander | 17 | No — both recipes fail |
| 505 persona-vector bank | L21 / persona-vector | bystander | 52 | Yes |

</details>

The sole determinate bank (505) is visibly the lone L21 / persona-vector row — every other bank is an L20 or L15 marker / logit / extraction bank. That structural uniqueness is the binding caveat behind the LOW verdict.

Complete per-bank table (all metrics, CIs, eligibility flags, join-sanity values): [`eval_results/issue_648/per_bank_skill_table.json`](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/eval_results/issue_648/per_bank_skill_table.json) (CSV mirror alongside).

### Generated

n/a — no model generation in this task. Each bank produces numeric metrics (two CV R² values, two Spearman ρ values, paired-bootstrap CIs), not completions. The reader-facing numbers are in the figures and the per-bank table linked above; there are no raw completions to sample.

## Reproducibility

**Parameters:**

| Field | Value |
|---|---|
| Model | none (reads base-model Qwen-2.5-7B centroid banks from the parent audit) |
| Centering switch | raw = `centering='none'`; centered = `centering='global_mean'` (train-fold-refit for primary DV) |
| Primary DV | ΔR² = CV R²_centered − CV R²_raw, leave-one-group-out, per-fold train-mean SS_tot baseline |
| Secondary DV | Δρ = ρ_centered − ρ_raw, length-partialled Spearman, in-sample (bank-global centering) |
| Bootstrap | 10,000 cluster (original-group) resamples; percentile 95% CI; RNG seed 20648 |
| Verdict eligibility | n_groups > 5 AND not both-recipes-fail (i.e. at least one recipe beats baseline); 4 of 9 banks eligible |
| Banks | 9 recoverable exact-row banks (#66, #142, #405, #478, #490, #380, #396/#415, #311, #505) |

**Artifacts:**

- **Methodology reference:** [docs/methodology/issue_648.md](https://github.com/superkaiba/explore-persona-space/blob/3d809d40332eae0ddcd10167e407d845473ee98a/docs/methodology/issue_648.md) · [gist](https://gist.github.com/superkaiba/40a5bba5664579ea7b2aae4d1480916c)
- Per-bank table: [`eval_results/issue_648/per_bank_skill_table.json`](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/eval_results/issue_648/per_bank_skill_table.json) + `.csv` mirror (git-tracked on the issue branch).
- Hero forest: [`figures/issue_648/hero_forest_delta_cvR2.png`](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/hero_forest_delta_cvR2.png); supplementary `insample_vs_cv_scatter.png` + `compression_offdiag.png` (pinned at `d83bcc25`), `paired_r2_raw_vs_centered.png` (re-pinned at `c745da0a` — restricted to the three both-fail banks in round 2). Each PNG ships PDF + meta.json sidecars.
Reused inputs (all from the parent audit's program; this task adds no new training data):

- **Base-model centroid banks from [#536](https://eps.superkaiba.com/tasks/536)** — the identical Qwen-2.5-7B centroid tensors the parent's recompute-driver consumed, read per `family` from these on-disk paths (large `.pt` tensors, git-ignored — local VM / HF, not committed blobs): `single_token_100p_L20` / `_core11` / 478 / 490 banks all read `eval_results/single_token_100_persona/centroids/centroids_layer20.pt`; `extraction_method_a_L20` reads `eval_results/extraction_method_comparison/centroids_method_a.pt`; `issue274_n24_L15` (both the 24-marker and 24-logit banks) reads `eval_results/issue_274/centroids/centroids_n24_layers0_27.pt`; `issue311_19bank_L20` reads `eval_results/issue_311/centroids_base.pt`; `issue505_pv_L21` reads `data/issue_505/centroids_pv/centroids_pv_L21.pt` (falling back to HF `superkaiba1/explore-persona-space-data` blob `issue505_loo_contrastive/geometry/centroids_pv_L21.pt`). Fit: same base-model centroids, same recipe; the parent's exact join gate (`GATE_MATRIX_TOL = 1e-4`, `GATE_RHO_TOL = 0.02`, imported verbatim from `issue536_recompute_driver`) passes per bank before any skill is read — every `join_gate_max_dev` is < 1e-6 (well under the 1e-4 matrix-gate tolerance), and every centered headline reproduces the published Spearman within `GATE_RHO_TOL` (505 PV at L21, 24-persona marker/logit at L15, 19-bank joint-leakage at L20 — e.g. the 19-bank centered headline reproduces −0.348 = published −0.348; core-11 marker 0.5674 ≈ published 0.567).
- **Producing-task leakage targets** — the same per-cell continuous targets the parent joined, read unchanged from these repo-relative paths: [#66](https://eps.superkaiba.com/tasks/66) and [#142](https://eps.superkaiba.com/tasks/142) read the 5 source dirs `eval_results/single_token_100_persona/{villain,comedian,assistant,software_engineer,kindergarten_teacher}/marker_eval.json` (#142 on the core-11 target subset); [#405](https://eps.superkaiba.com/tasks/405) reads `eval_results/issue_405/aggregate/per_cell_persona_tidy.csv`; [#478](https://eps.superkaiba.com/tasks/478) reads `eval_results/issue_536/inputs/i478_tidy_69b34b94.csv`; [#490](https://eps.superkaiba.com/tasks/490) reads `eval_results/issue_490/aggregate/persona_level.csv`; [#380](https://eps.superkaiba.com/tasks/380) reads `eval_results/issue_380/cosine_pairwise_n24/correlation.json`; [#396](https://eps.superkaiba.com/tasks/396)/[#415](https://eps.superkaiba.com/tasks/415) read `eval_results/issue_396/analysis_summary.json` + `eval_results/issue_415/base_model_predictors_v2.json`; [#311](https://eps.superkaiba.com/tasks/311) reads `eval_results/issue_311/analysis.json` (+ `pair_selection.json`); [#505](https://eps.superkaiba.com/tasks/505) reads `eval_results/issue_505/analysis/delta_leakage_per_seed.json`. Fit: identical targets to the parent's; the raw-recipe reproduction matches the parent's published Spearman within `GATE_RHO_TOL`.
- **111-persona distance JSON from [#560](https://eps.superkaiba.com/tasks/560)** (`eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json`): restored read-only from git blob [`776c7c3b75`](https://github.com/superkaiba/explore-persona-space/blob/776c7c3b758942f5719557fc69e1e2420af0c36b/eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json) (immutable blob; the 1e-4 matrix join gate is the content-identity check). Fit: same base-model centroids, exact join gate passes.
- The persona-distance predictor program originates in [#404](https://eps.superkaiba.com/tasks/404)/[#458](https://eps.superkaiba.com/tasks/458) (cosine / JS-divergence predictor line).

**Compute:** CPU on the VM, < ~10 min wall-clock single-process. No pod, no GPU.

**Code:**

- Driver: [`scripts/issue648_centered_vs_raw_predictive_skill.py`](https://github.com/superkaiba/explore-persona-space/blob/4ff0a15c43282bf8e6e88653898c98816b96efdc/scripts/issue648_centered_vs_raw_predictive_skill.py) (imports the parent audit's loaders, gates, and `length_partial_spearman` verbatim; adds the leave-one-group-out CV R² with train-fold-only centering and the paired bootstrap).
- Figures: [`scripts/issue648_analyzer_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/c745da0abc227b880b16fc0a94b19bac8f427607/scripts/issue648_analyzer_figures.py).
- Reproduce: `uv run python scripts/issue648_centered_vs_raw_predictive_skill.py --data-root "$REPO_ROOT" --out-dir eval_results/issue_648 --fig-dir figures/issue_648` then `uv run python scripts/issue648_analyzer_figures.py`.
- Result metadata: git_commit `4ff0a15c43`, data_root_commit `cbae80d9ec`, bank111 restore `776c7c3b75`, rng_seed 20648, n_boot 10000.

**Context:**

- Created / run: filed 2026-06-15T22:40Z; analysis ran and figures landed 2026-06-15/16.
- Follow-up to: [#536](https://eps.superkaiba.com/tasks/536) — the metric-recipe validity audit that settled "centered is valid" but not "centered is the stronger predictor".
- Originating prompt(s), verbatim:

  > queue a centered-vs-raw predictive-skill comparison on the cached predictor panels (CV R²_raw vs CV R²_centered on the same cell): #536 settled which recipe is valid + that calls survive, but not which maximizes predictive skill; effect on skill goes both directions by bank (100-persona raw inflates 0.60->0.77; 19-bank headline exists only centered -0.348 vs -0.037)

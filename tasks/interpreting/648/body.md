---
title: Centered cosine is not a stronger leakage predictor than raw — one of four
  banks resolves out-of-sample, and it favors raw (LOW confidence)
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
# Centered cosine is not a stronger leakage predictor than raw — one of four banks resolves out-of-sample, and it favors raw (LOW confidence)

<!-- clean-result-v3 -->

## Takeaways

- Of 4 verdict-eligible banks, **only the 505 persona-vector bank resolves out-of-sample (ΔR² = −0.071, CI [−0.122, −0.007])** — and it favors raw, not the canonical centered recipe.
- The other 3 eligible banks span 0 (**+0.030, +0.017, −0.297**): no centered-favored determinate bank exists. "Centering is uniformly stronger" is falsified; the panel is too thin for a bank-dependent verdict.
- On the 505 bank, centering **improves** in-sample fit (Δρ = +0.208 [+0.153, +0.264]) yet **hurts** held-out skill — the reverse of the raw-inflation mechanism.
- On three small banks (24/24/17 personas), **both recipes score CV R² < 0** — worse than the train mean — so their reported in-sample correlations do not predict held-out leakage.
- Centering stays canonical on validity grounds (raw is anisotropy-compressed into a ~[0.7, 1.0] ridge); this run only finds it buys no extra predictive skill.

## What I ran

- **Why:** the parent metric-recipe audit ([#536](https://eps.superkaiba.com/tasks/536)) settled that mean-centering the centroid bank is the *valid* cosine recipe and that the published persona-distance calls survive the swap, but it never asked which recipe maximizes *predictive skill*. Its survival check pointed both ways by bank — on one bank raw inflates the headline correlation, on another the headline exists only under centering — so the skill question was genuinely open.
- **Design:** for each recoverable centroid bank from the parent audit, score the cosine-distance leakage predictor two ways — raw (un-centered) vs centered (global-mean-subtracted) — on the SAME cells, target, length residualization, and cross-validation folds. The single manipulated variable is the centering recipe.
- **Eval:** primary DV is held-out cross-validation R² (leave-one-group-out, the bank's own natural held-out unit; centering mean refit on the train fold inside every split so no held-out persona enters its own predictor), differenced across recipes (ΔR² = centered − raw) with a paired bootstrap 95% CI (10,000 resamples) in original-group space. Secondary DV is the in-sample length-partialled Spearman difference Δρ, reported for continuity with the parent's survival check. 9 banks; the 2 with only 5 folds report their numbers but are excluded from the verdict.
- **Scope shrinkage:** of the 9 banks, only **4 are verdict-eligible** (more than 5 folds AND both recipes beat the train-mean baseline). Two marker banks have only 5 folds (a 5-from-5 bootstrap is too coarse for a determinate call) and three mid-size banks have both recipes failing out-of-sample — all five are reported but mechanically excluded from the hypothesis verdict.

## Findings

### Only the 505 bank resolves — and centering is the worse predictor there

The verdict rests on held-out ΔR² (centered − raw): a bank has a determinate winner only when its paired-bootstrap CI excludes 0. One of 4 eligible banks clears that bar.

![Forest plot of held-out CV R-squared difference (centered minus raw) per bank, with paired-bootstrap 95% CI bars; only the 505 persona-vector bank has a filled marker with its CI fully left of zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/hero_forest_delta_cvR2.png)

> **Figure.** *Centering is not the stronger leakage predictor.* x = ΔR² (centered − raw), held-out CV, paired-bootstrap 95% CI. Filled = determinate (CI excludes 0); open = indistinguishable, low-fold, or both-recipes-fail. n = fold count per bank. Only the 505 bank (n=52) resolves, at ΔR² = −0.071 — raw-favored.

The 505 bank gives ΔR² = −0.071 [−0.122, −0.007]: centered predicts held-out leakage *worse* than raw. The 20-persona bank leans the same way (−0.297, CI touches 0); the two 111-persona banks span 0 (+0.030, +0.017). No eligible bank favors centering with a CI excluding 0, so "centering is uniformly stronger" is falsified — yet a single determinate sign across four banks is not a clean bank-dependent result either.

### In-sample fit and held-out skill disagree in sign — the opposite of raw-inflation

Raw-inflation predicts raw should fit the sample better than it predicts held-out data. Each bank's in-sample Δρ against its held-out ΔR² tests that.

![Scatter of held-out CV R-squared difference against in-sample Spearman difference, both centered minus raw, one point per bank; the 505 bank sits at positive in-sample delta but negative held-out delta.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/insample_vs_cv_scatter.png)

> **Figure.** *In-sample and held-out deltas point opposite ways on the resolving bank.* x = in-sample Δρ (centered − raw); y = held-out ΔR² (centered − raw). Filled = verdict-eligible, open = excluded (low-fold or both-fail). On the 505 bank centering helps in-sample (Δρ = +0.208) yet hurts held-out (ΔR² = −0.071).

On the 505 bank, centering *improves* in-sample fit (Δρ = +0.208 [+0.153, +0.264]) while *worsening* held-out skill — centered fits the sample better but raw generalizes better, the reverse of raw-inflation. Only the two low-fold marker banks show raw fitting the sample better (Δρ = −0.164, −0.232, CIs exclude 0) — the inflation candidates the parent flagged, but with 5 folds each they cannot carry a verdict.

### Three small banks: both recipes fail out-of-sample entirely

The parent reported in-sample correlations on several small banks. Whether those translate to held-out prediction is separate — and on three mid-size banks, neither recipe does.

![Grouped bar chart of held-out CV R-squared (raw vs centered) per bank; the 24-persona marker, 24-persona logit, and 19-persona joint-leakage banks have both bars below zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/paired_r2_raw_vs_centered.png)

> **Figure.** *On the 24/24/17-persona banks, both recipes score below the train-mean baseline.* Bars are held-out CV R² (per-fold train-mean baseline); below 0 = worse than predicting the mean. Raw (orange) and centered (blue) both fail on the three small banks; the larger banks stay positive.

On all three small banks both recipes land at CV R² < 0 — worse than guessing the train mean. They are excluded from the verdict (a recipe contrast is meaningless when neither recipe predicts), but the exclusion is the finding: leave-one-persona-out generalization of the cosine-distance predictor breaks down below ~30 personas, regardless of centering.

### Raw cosine is compressed, but compression does not pick the winner

Centering exists because raw cosine is squeezed into the anisotropy ridge. The compression is real and uniform on the three banks with the read — but does not predict the held-out winner.

![Three side-by-side histograms of off-diagonal cosine values, raw vs centered, for the 111-persona, 20-persona, and 505 banks; raw piles into 0.7-1.0 while centered spreads across -0.7 to 1.0.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/compression_offdiag.png)

> **Figure.** *Raw cosine collapses into the anisotropy ridge on all three banks with this read.* Off-diagonal cosine distributions: raw (orange) spans only ~[0.7, 1.0]; centered (blue) spreads across ~[−0.7, 1.0]. Available for 3 of 9 banks (the off-diagonal payload only persists for these families).

Raw cosine compresses into ~[0.7, 1.0] on all three families — the degeneracy that makes centering valid. But the held-out winner does not follow compression: the raw-leaning (505, 20-persona) and flat (111-persona) banks are all equally compressed, so compression does not explain the ΔR² signs. The raw-inflation mechanism is present in the geometry but does not drive the skill verdict here.

## Data

### Trained on

n/a — no training in this task. This is a CPU-only re-analysis of centroid banks and leakage targets produced by prior `kind: experiment` tasks. The persona-distance predictor program is the parent line ([#536](https://eps.superkaiba.com/tasks/536) recipe audit; [#404](https://eps.superkaiba.com/tasks/404)/[#458](https://eps.superkaiba.com/tasks/458) predictor origin).

### Evaluated with

Per recoverable bank, two inputs joined exactly as the parent audit joined them: a persisted base-model centroid bank (Qwen-2.5-7B, the layer each parent task used) and that bank's per-cell continuous leakage target (marker leakage rate, marker/logit shift, or fact-leakage delta, depending on the bank). The cosine-distance predictor is built under each centering recipe on the same cells; the target, the length covariate + residualization, the cross-validation fold map, and the bootstrap resample indices are held identical across recipes. Before any skill is read, each bank's raw recompute must reproduce the parent's published Spearman within tolerance — every bank passed (e.g. the 19-persona bank's centered headline reproduced at −0.348 = published −0.348; the core-11 marker subset at 0.5674 ≈ published 0.567).

The 9 banks, plain-English names and fold counts (first 9 of 9 rows — the complete set, not a sample; full per-bank metrics at the artifact link below):

<details>
<summary>Per-bank join + fold map (first 9 of 9 rows — complete set)</summary>

| Bank | Held-out unit | Folds | Verdict-eligible? |
|---|---|---|---|
| 100-persona marker bank (#66) | source | 5 | No — only 5 folds |
| core-11 marker subset (#142) | source | 5 | No — only 5 folds |
| 20-persona extraction bank (#405) | held persona | 8 | Yes |
| 111-persona marker bank (#478) | held persona | 35 | Yes |
| 111-persona dose-matched bank (#490) | persona | 11 | Yes |
| 24-persona marker bank (#380) | persona | 24 | No — both recipes fail |
| 24-persona logit bank (#396/#415) | source | 24 | No — both recipes fail |
| 19-persona joint-leakage bank (#311) | bystander | 17 | No — both recipes fail |
| 505 persona-vector bank (#505) | bystander | 52 | Yes |

</details>

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
| Verdict eligibility | n_groups > 5 AND not both-recipes-fail; 4 of 9 banks eligible |
| Banks | 9 recoverable exact-row banks (#66, #142, #405, #478, #490, #380, #396/#415, #311, #505) |

**Artifacts:**

- Per-bank table: [`eval_results/issue_648/per_bank_skill_table.json`](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/eval_results/issue_648/per_bank_skill_table.json) + `.csv` mirror (git-tracked on the issue branch).
- Hero forest: [`figures/issue_648/hero_forest_delta_cvR2.png`](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/hero_forest_delta_cvR2.png); supplementary `insample_vs_cv_scatter.png`, `paired_r2_raw_vs_centered.png`, `compression_offdiag.png` (each with PDF + meta.json sidecar).
- Reused inputs from [#536](https://eps.superkaiba.com/tasks/536): the centroid banks + leakage targets the parent audit consumed. The 111-persona distance JSON (`cosine_distance_matrix_layer20.json`) was restored read-only from git `776c7c3b75` (immutable; the 1e-4 matrix join gate is the content-identity check) — fit: same base-model centroids, same recipe, the parent's exact join gate passes per bank before any skill is read.

**Compute:** CPU on the VM, < ~10 min wall-clock single-process. No pod, no GPU.

**Code:**

- Driver: [`scripts/issue648_centered_vs_raw_predictive_skill.py`](https://github.com/superkaiba/explore-persona-space/blob/4ff0a15c43282bf8e6e88653898c98816b96efdc/scripts/issue648_centered_vs_raw_predictive_skill.py) (imports the parent audit's loaders, gates, and `length_partial_spearman` verbatim; adds the leave-one-group-out CV R² with train-fold-only centering and the paired bootstrap).
- Figures: [`scripts/issue648_analyzer_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/scripts/issue648_analyzer_figures.py).
- Reproduce: `uv run python scripts/issue648_centered_vs_raw_predictive_skill.py --data-root "$REPO_ROOT" --out-dir eval_results/issue_648 --fig-dir figures/issue_648` then `uv run python scripts/issue648_analyzer_figures.py`.
- Result metadata: git_commit `4ff0a15c43`, data_root_commit `cbae80d9ec`, bank111 restore `776c7c3b75`, rng_seed 20648, n_boot 10000.

**Context:**

- Created / run: filed 2026-06-15T22:40Z; analysis ran and figures landed 2026-06-15/16.
- Follow-up to: [#536](https://eps.superkaiba.com/tasks/536) — the metric-recipe validity audit that settled "centered is valid" but not "centered is the stronger predictor".
- Originating prompt(s), verbatim:

  > queue a centered-vs-raw predictive-skill comparison on the cached predictor panels (CV R²_raw vs CV R²_centered on the same cell): #536 settled which recipe is valid + that calls survive, but not which maximizes predictive skill; effect on skill goes both directions by bank (100-persona raw inflates 0.60->0.77; 19-bank headline exists only centered -0.348 vs -0.037)

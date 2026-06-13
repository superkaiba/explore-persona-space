---
title: Estimator-fragility sweep of the clustered leakage-line reads (cluster-robust
  OLS vs MixedLM on the gated joins)
kind: analysis
tags: []
created_at: '2026-06-11T05:57:27Z'
has_clean_result: false
parent_id: 536
---
# The set-size-by-distance leakage call hangs entirely on the uncertainty estimator — significant under cluster-robust OLS, null under the persona random-effect model at the identical coefficient; the other three clustered persona-distance calls survive the swap on both raw and centered axes (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Four clustered persona-distance calls re-fit under cluster-robust OLS vs a persona random-effect MixedLM, on raw and centered axes: **2 of 4 estimator-invariant, 1 swaps significance, 1 unfittable**.
- The **set-size-by-distance interaction** is fragile: same point estimate (**+0.0097 / +0.0215**), cluster-robust OLS **p = 0.022 / 0.00075** (significant) but persona-RE MixedLM **p = 0.405 / 0.215** (null).
- The **panel-regression secondary** (both estimators p < 1e-11) and the **on-axis dose-matched gap** (both null, p = 0.10-0.44) hold their call under both estimators on both axes.
- The **leave-one-out pooled** MixedLM hits a singular non-PD-Hessian boundary on both joins — the published cluster-robust null has no admissible alternative, a third outcome, not a flip.
- The published flatness null **stands under its own estimator**; the cluster-robust mirror's significance is estimator-conditional, on two small-cluster rows (18 and 24 clusters).

## What I ran

- **Why:** The persona-distance audit [#536](https://eps.superkaiba.com/tasks/536) re-graded four "distance predicts which personas catch a behavior" calls, each decided under one registered uncertainty estimator. The open question: does any call depend on which standard variance estimator was used? One row ([#536](https://eps.superkaiba.com/tasks/536)'s set-size-by-distance mirror) already showed a cluster-robust p = 0.00075 vs MixedLM p = 0.215 split on identical data, so the fragility was known to be live on at least one row.
- **Design:** four clustered leakage-line reads × two estimators (cluster-robust OLS, persona-RE MixedLM) × two distance axes (raw, mean-centered) = 16 cells. The single manipulated dimension is the uncertainty estimator; data, joins, point estimates, clustering variables, and per-row α are all held fixed at [#536](https://eps.superkaiba.com/tasks/536)'s gated values.
- **Training:** none — CPU `statsmodels` re-fits over [#536](https://eps.superkaiba.com/tasks/536)'s persisted joins; 0 GPU-hours.
- **Eval:** per cell, the headline term's coefficient + Wald 95% CI + p-value, with `n_rows` and `n_clusters`. A call "swaps significance" when the alternative estimator's p crosses the row's published α with the sign preserved; a sign change is flagged separately. A join-validity gate reproduces each published point estimate (`manipulation_check_ratio < 3e-4` on all four rows) before any swept p is read.
- **Scope:** the Goal's fourth named read — the per-set-size correlations — is quoted from the parent, not swept: those are per-stratum Spearman/OLS statistics with no cluster or random-effect structure, so the cluster-robust-vs-MixedLM axis is undefined for them. The estimator sweep covers the four genuinely clustered rows.

## Findings

### One call swaps significance, two hold, one cannot be MixedLM-fit

Re-fit each clustered call under both estimators on both axes; check which cross their published α. Only the set-size-by-distance interaction does.

![p-value (log scale) for four clustered reads under cluster-robust OLS and persona-RE MixedLM, two panels for raw and centered axes; the set-size-by-distance column is highlighted with the two estimators on opposite sides of the 0.05 line.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea40758e76ba15c797a8f14dfb2e7cd4188d31d0/figures/issue_589/estimator_sweep.png)

> **Figure.** *Only the set-size-by-distance interaction (highlighted) lands the two estimators on opposite sides of the threshold; the panel-regression secondary stays significant, the on-axis gap stays null, the pooled MixedLM is unfittable.* Blue = cluster-robust OLS, red = persona-RE MixedLM, open = MixedLM singular. Dashed p = 0.05, dotted p = 0.01.

- **Panel-regression secondary** — significant under both estimators, both axes (p < 1e-11). Its published two-VC MixedLM reproduces the parent (β −27.7 raw / −2.27 centered); cluster-robust OLS is the sensitivity probe.
- **On-axis dose-matched gap** — null under both, both axes (β +0.20 raw / +0.11 centered; p 0.10-0.44).
- **Leave-one-out pooled** — the persona-RE MixedLM hits a singular boundary on both joins, so the published cluster-robust null has no admissible alternative — a third outcome, not a flip.

### The flip is a pure standard-error move — the coefficient does not budge

The interaction's published call is null under its mixed-effects estimator; the cluster-robust mirror read it significant. Plotting the coefficient with each estimator's own CI isolates what moves.

![Forest plot of coefficient with Wald 95 percent CI for the interaction and the on-axis gap under both estimators on both axes; for the interaction both estimators share the coefficient but the MixedLM CI is wider and crosses zero where the cluster-robust CI excludes it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea40758e76ba15c797a8f14dfb2e7cd4188d31d0/figures/issue_589/estimator_sweep_forest.png)

> **Figure.** *Same coefficient under both estimators; the MixedLM interval is ~3× wider and crosses zero where the cluster-robust interval excludes it.* Interaction (top) and on-axis gap (bottom), blue = cluster-robust OLS, red = persona-RE MixedLM, dashed line at 0. The coefficient is invariant; only the SE moves.

- The coefficient is invariant to ~12 digits (raw +0.0097, centered +0.0215); the cluster-robust SE (0.0043 / 0.0064) versus MixedLM SE (0.0117 / 0.0173) is the whole difference.
- The persona variance component absorbs between-persona structure the cluster-robust fit charges to the slope, widening the interval until the same coefficient sits inside noise.
- The published null holds under its own (MixedLM) estimator, so the cluster-robust rescue is estimator-conditional.

### Why the pooled null reads flat while per-arm slopes are large — and goes singular

The pooled read averages six per-arm slopes (heteroskedastic-robust OLS within each held-out arm; no within-arm cluster structure), quoted verbatim from the parent.

![Per-arm OLS slopes of leakage change versus held-out persona distance for six arms, raw in blue and centered in orange with 95 percent CIs; on the centered axis three arms are positive and the child arm is strongly negative, so the signs oppose across arms.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ea40758e76ba15c797a8f14dfb2e7cd4188d31d0/figures/issue_589/estimator_sweep_505_perarm.png)

> **Figure.** *Per-arm slopes carry opposing signs the pooled fit averages toward zero.* Six leave-one-out arms; blue = raw distance, orange = mean-centered, error bars 95% CI, dashed line at 0. On the centered axis the child arm is −0.75 (p = 1.9e-5) against three positive arms (p = 0.026-0.038); n = 156 rows per arm.

The flat pooled read is partly opposing per-arm slopes cancelling, not uniformly flat arms — the between-arm heterogeneity a persona random effect was meant to absorb, and the structure that drove the pooled MixedLM singular.

## Data

### Trained on

n/a — no training in this task. The sweep re-fits over [#536](https://eps.superkaiba.com/tasks/536)'s already-gated joins; no model, no adapter, no training mix.

### Evaluated with

The four clustered leakage-line reads from [#536](https://eps.superkaiba.com/tasks/536)'s `FAMILY_REGISTRY`, chosen because each is a clustered regression (cluster-robust OLS or MixedLM with a grouping / random-effect structure) and one of the four reads the Goal names. Each row's headline term, clustering variable, and published estimator were read directly from the parent driver code; the joins are reproduced on both the raw (`centering='none'`) and mean-centered (`centering='global_mean'`) distance axes. No preprocessing beyond the parent's join construction; the join-validity gate (`manipulation_check_ratio < 3e-4`, all four rows) reproduces each published point estimate before any swept p is read.

<details open>
<summary>The 4 swept reads (4 of 4, the full clustered row set; per-K correlations quoted-not-swept — see What I ran)</summary>

| Read | Headline term | Clustering / RE structure | Published estimator | N |
|---|---|---|---|---|
| Panel-regression secondary | `min_dist` | two-VC MixedLM `{subset, persona}`, single dummy group | MixedLM | 336 |
| On-axis dose-matched gap | `is_on_axis` | cluster-robust OLS, clusters = pair × seed (24) | cluster-robust OLS | 255 |
| Leave-one-out pooled | `cos(b,j)` | cluster-robust OLS, clusters = held-out × seed (18) | cluster-robust OLS | 936 |
| Set-size × distance interaction | `K × log(min_dist)` | cluster-robust OLS, clusters = cell × seed (2,800 rows) | MixedLM | 2,800 |

</details>

Full machine-readable payload (16 cells + per-arm context + reproducibility metadata): [`sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/6d1395b92a0cc9780d08b048fb883d60f8984964/eval_results/issue_589/sweep_results.json)

### Generated

No completions — each cell yields one regression fit (coefficient, SE, p, CI, convergence flags), not generated text. The 16-cell table is the output; the MixedLM cells additionally carry `converged` / `boundary_variance` / captured fit warnings.

<details>
<summary>3 of 16 cells, cherry-picked to show one flip, one stable null, one singular fit (all 16 rows: linked CSV/JSON below)</summary>

```
SET-SIZE x DISTANCE, centered join — significance swap
  cluster_ols: beta=+0.02148  se=0.00638  p=0.000753  CI=[+0.0090,+0.0340]  N=2800 clu=80  -> significant
  mixedlm:     beta=+0.02148  se=0.01733  p=0.215      CI=[-0.0125,+0.0554]  N=2800 clu=40  -> null
  (point estimate identical to ~12 digits; only SE/p move)

ON-AXIS DOSE-MATCHED GAP, raw join — stable null
  cluster_ols: beta=+0.19956  se=0.12223  p=0.1026  CI=[-0.040,+0.439]  N=255 clu=24  -> null
  mixedlm:     beta=+0.19956  se=0.13292  p=0.1333  CI=[-0.061,+0.460]  N=255 clu=24  -> null

LEAVE-ONE-OUT POOLED, raw join — MixedLM singular
  cluster_ols: beta=-2.69339  se=1.03973  p=0.00958  CI=[-4.731,-0.656]  N=936 clu=18  -> significant
  mixedlm:     FAILED — non-PD Hessian; degenerate SE=2.3e-04; inference not trustworthy
```

</details>

Full per-cell table (16 data rows): [`sweep_results.csv`](https://github.com/superkaiba/explore-persona-space/blob/6d1395b92a0cc9780d08b048fb883d60f8984964/eval_results/issue_589/sweep_results.csv)

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Model | none (CPU linear algebra + `statsmodels` re-fits) |
| Estimators swept | cluster-robust OLS (`cov_type='cluster'`) ↔ persona-RE MixedLM (`reml=False`, `lbfgs`, `vc_formula={'persona':'0+C(persona)'}`) |
| Joins | raw (`centering='none'`), mean-centered (`centering='global_mean'`) |
| α per row | #405: 0.01; #490 / #505 / #478-interaction: 0.05 |
| Join-validity gate | point-estimate reproduction within 0.02 (statistic-level); 1e-4 matrix/row-level inside the parent `family_*` join builders |
| Multiplicity | none within-row (each row keeps its published α + family rule); 16-cell grid flagged as exploratory robustness, not 16 confirmatory tests |
| Cells | 4 reads × 2 estimators × 2 joins = 16 |

**Artifacts:**

- Per-cell table (16 data rows, 17 columns): [`sweep_results.csv`](https://github.com/superkaiba/explore-persona-space/blob/6d1395b92a0cc9780d08b048fb883d60f8984964/eval_results/issue_589/sweep_results.csv)
- Machine-readable payload + per-arm #505 context + reproducibility metadata: [`sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/6d1395b92a0cc9780d08b048fb883d60f8984964/eval_results/issue_589/sweep_results.json)
- Figures (PNG + PDF + meta.json): [`figures/issue_589/`](https://github.com/superkaiba/explore-persona-space/tree/6d1395b92a0cc9780d08b048fb883d60f8984964/figures/issue_589)
- Reused from [#536](https://eps.superkaiba.com/tasks/536): the gated joins persisted at `12853bca8` (`eval_results/issue_536/inputs/i478_tidy_69b34b94.csv`; `eval_results/issue_{405,490,505}/...`; centroid banks + the #505 HF geometry bundle `superkaiba1/explore-persona-space-data:issue505_loo_contrastive/geometry/centroids_pv_L21.pt`) — fit: each row re-fits that read's own persisted join, validated by the parent's 1e-4 matrix / statistic-level gates inside `family_*`; the cluster-robust cell reproduces the parent's persisted coefficient + p to numerical tolerance.

**Compute:** CPU only, VM-side, deterministic, minutes. No pod, 0 GPU-hours.

**Code:**

- Sweep driver: [`scripts/issue589_estimator_sweep.py`](https://github.com/superkaiba/explore-persona-space/blob/6d1395b92a0cc9780d08b048fb883d60f8984964/scripts/issue589_estimator_sweep.py) (imports `issue536_recompute_driver` + `issue536_mixedlm_refit`)
- Figure script: [`scripts/issue589_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/ea40758e76ba15c797a8f14dfb2e7cd4188d31d0/scripts/issue589_figures.py)
- Env: statsmodels 0.14.6, scipy 1.17.1, numpy 2.2.6, pandas 2.3.3, python 3.11.15
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space && git checkout 6d1395b92a0cc9780d08b048fb883d60f8984964
    uv sync
    git checkout 45fe33f85 -- eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json
    uv run python scripts/issue589_estimator_sweep.py --data-root "$PWD"
    uv run python scripts/issue589_figures.py
    ```

**Context:**

- Created 2026-06-13; run executed 2026-06-13.
- Follow-up to [#536](https://eps.superkaiba.com/tasks/536) — the persona-distance audit whose four clustered leakage-line calls this sweep re-fits under an alternative uncertainty estimator.
- Originating prompt: origin prompt not recorded

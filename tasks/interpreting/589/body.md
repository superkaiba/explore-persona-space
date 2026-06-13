---
title: Two clustered persona-distance calls survive the estimator swap, while the
  set-size and leave-one-out calls remain estimator-dependent (MODERATE confidence)
kind: analysis
tags: []
created_at: '2026-06-11T05:57:27Z'
has_clean_result: true
parent_id: 536
---
# Two clustered persona-distance calls survive the estimator swap, while the set-size and leave-one-out calls remain estimator-dependent (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Four clustered persona-distance calls, two estimators, two axes: **2 of 4 estimator-invariant, 1 swaps significance, 1 estimator-conflicted on the raw axis**.
- The **set-size-by-distance interaction** is fragile: same point estimate, cluster-robust OLS **p = 0.022 / 0.00075** but persona-RE MixedLM **p = 0.405 / 0.215** — a pure standard-error move.
- The **leave-one-out pooled** raw read disagrees on sign: cluster-robust **β = −2.69, p = 0.0096** vs groups-only-REML MixedLM **β = +0.831, p = 0.0024**, both significant, opposite directions.
- That pooled read is **null and consistent on the centered axis** (cluster-robust **p = 0.96**, groups-only-REML **β = +0.055, p = 0.168**) — the published null holds where centered.
- The **panel-regression secondary** stays significant under both estimators (p < 1e-11); the **on-axis dose-matched gap** stays null under both (p = 0.10-0.44).

## What I ran

- **Why:** The persona-distance audit [#536](https://eps.superkaiba.com/tasks/536) re-graded four "distance predicts which personas catch a behavior" calls, each decided under one registered uncertainty estimator, and one row (the set-size-by-distance mirror) already showed a cluster-robust p = 0.00075 vs MixedLM p = 0.215 split on identical data. The open question: does any call depend on which standard variance estimator was used?
- **Design:** four clustered leakage-line reads × two estimators (cluster-robust OLS, persona-RE MixedLM) × two distance axes (raw, mean-centered) = 16 cells, plus 2 cells re-fitting the one singular read under a simpler admissible MixedLM spec (18 cells total). The single manipulated dimension is the uncertainty estimator; data, joins, point estimates, clustering variables, and per-row α are held fixed at the parent audit's gated values.
- **Training:** none — CPU `statsmodels` re-fits over the parent's persisted joins; 0 GPU-hours.
- **Eval:** per cell, the headline term's coefficient + Wald 95% CI + p-value, with `n_rows` and `n_clusters`; a call "swaps significance" when the alternative estimator's p crosses the row's published α with the sign preserved (a sign change is flagged separately and is graver). A join-validity gate reproduces each published point estimate (`manipulation_check_ratio < 3e-4` on all four rows) before any swept p is read.
- **Rounds:**

| Round | What changed | Cells added | Outcome |
|---|---|---|---|
| 1 — initial sweep (2026-06-13) | 4 clustered reads × 2 estimators × 2 axes | 16 | Set-size interaction swaps significance; panel-regression secondary + on-axis gap estimator-invariant; leave-one-out pooled persona-VC MixedLM singular on the raw axis. |
| 2 — 9a-ter free-analysis follow-up (2026-06-13, `issue589-505-altvc-refit`) | Re-fit the singular leave-one-out read under a groups-only-REML MixedLM (no persona VC) | 2 | Raw-axis pooled slope flips sign (cluster-robust −2.69 vs groups-only-REML +0.831, both significant); centered axis stays a consistent null. |

- **Scope:** the Goal's fourth named read — the per-set-size correlations — is quoted from the parent, not swept: those are per-stratum Spearman/OLS statistics with no cluster or random-effect structure, so the cluster-robust-vs-MixedLM axis is undefined for them. The estimator sweep covers the four genuinely clustered rows.

## Findings

### One call swaps significance, two hold, one is estimator-conflicted

Re-fit each clustered call under both estimators on both axes; check which cross their published α.

![p-value (log scale) for four clustered reads under cluster-robust OLS and persona-RE MixedLM, two panels for raw and centered axes; the set-size-by-distance column is highlighted with the two estimators on opposite sides of the 0.05 line, and the leave-one-out column shows the persona-VC MixedLM singular alongside a sign-flipped groups-only-REML cell.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/figures/issue_589/estimator_sweep.png)

> **Figure.** *The set-size-by-distance interaction (highlighted) lands the two estimators on opposite sides of the threshold; the leave-one-out pooled raw read has both a singular persona-VC fit and a sign-flipped groups-only-REML fit.* Blue = cluster-robust OLS, red = persona-RE MixedLM, open = persona-VC singular, green diamond = groups-only-REML sign-flipped. Dashed p = 0.05, dotted p = 0.01.

- **Panel-regression secondary** — significant under both estimators, both axes (p < 1e-11; β −27.7 raw / −2.27 centered).
- **On-axis dose-matched gap** — null under both, both axes (β +0.20 raw / +0.11 centered; p 0.10-0.44).
- The set-size interaction swaps; the leave-one-out pooled read disagrees on sign (next two findings).

### The set-size flip is a pure standard-error move — the coefficient does not budge

The interaction's published call is null under its mixed-effects estimator; the cluster-robust mirror read it significant. The forest plot shows each estimator's own CI on the shared coefficient.

![Forest plot of coefficient with Wald 95 percent CI for the interaction and the on-axis gap under both estimators on both axes; for the interaction both estimators share the coefficient but the MixedLM CI is wider and crosses zero where the cluster-robust CI excludes it.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/figures/issue_589/estimator_sweep_forest.png)

> **Figure.** *Same coefficient under both estimators; the MixedLM interval is ~3× wider and crosses zero where the cluster-robust interval excludes it.* Interaction (top) and on-axis gap (bottom), blue = cluster-robust OLS, red = persona-RE MixedLM, dashed line at 0. The coefficient is invariant; only the SE moves.

- Coefficient invariant to ~12 digits (raw +0.0097, centered +0.0215); cluster-robust SE (0.0043 / 0.0064) vs MixedLM SE (0.0117 / 0.0173) is the whole difference.
- The persona variance component absorbs between-persona structure the cluster-robust fit charges to the slope, so the same coefficient sits inside noise.
- The published null holds under its own (MixedLM) estimator; the cluster-robust rescue is estimator-conditional.

### The leave-one-out pooled raw read is estimator-conflicted; centered is a consistent null

The pooled read averages six per-arm slopes. The persona-VC MixedLM is singular here, so I re-fit the same join under a simpler admissible spec: a groups-only random intercept (REML, no persona VC).

![Per-arm OLS slopes of leakage change versus held-out persona distance for six arms, raw in blue and centered in orange with 95 percent CIs; on the centered axis three arms are positive and the child arm is strongly negative, so the signs oppose across arms.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/figures/issue_589/estimator_sweep_505_perarm.png)

> **Figure.** *Per-arm slopes carry opposing signs the pooled fit averages toward zero.* Six leave-one-out arms; blue = raw, orange = mean-centered, 95% CI, dashed line at 0. Centered child arm −0.75 (p = 1.9e-5) vs three positive arms (p = 0.026-0.038); n = 156 per arm.

- **Raw axis — opposite-sign significance.** Cluster-robust OLS significant negative (β = −2.69, p = 0.0096); groups-only-REML MixedLM significant positive (β = +0.831, p = 0.0024) — same frame, different variance estimator, opposite-sign pooled slopes. The persona-VC MixedLM is singular, yielding no admissible read.
- **Centered axis — consistent null.** Cluster-robust OLS null (β = +0.008, p = 0.96), groups-only-REML null (β = +0.055, p = 0.168) — the published null holds where centered.
- Both MixedLM specs are alternatives to the parent's cluster-robust stand-in (no published MixedLM here), so the disagreement is between admissible estimators, driven by opposing-sign between-arm heterogeneity.

## Data

### Trained on

n/a — no training in this task. The sweep re-fits over the parent audit's already-gated joins; no model, no adapter, no training mix.

### Evaluated with

The four clustered leakage-line reads from the parent's `FAMILY_REGISTRY`, chosen because each is a clustered regression (cluster-robust OLS or MixedLM with a grouping / random-effect structure) and one of the four reads the Goal names. Each row's headline term, clustering variable, and published estimator were read directly from the parent driver code; the joins are reproduced on both the raw (`centering='none'`) and mean-centered (`centering='global_mean'`) distance axes. No preprocessing beyond the parent's join construction; the join-validity gate (`manipulation_check_ratio < 3e-4`, all four rows) reproduces each published point estimate before any swept p is read.

<details open>
<summary>The 4 swept reads (4 of 4, the full clustered row set; per-K correlations quoted-not-swept — see What I ran)</summary>

| Read | Headline term | Clustering / RE structure | Published estimator | N |
|---|---|---|---|---|
| Panel-regression secondary | `min_dist` | two-VC MixedLM `{subset, persona}`, single dummy group | MixedLM | 336 |
| On-axis dose-matched gap | `is_on_axis` | cluster-robust OLS, clusters = pair × seed (24) | cluster-robust OLS | 255 |
| Leave-one-out pooled | `cos(b,j)` | cluster-robust OLS, clusters = held-out × seed (18) | cluster-robust OLS | 936 |
| Set-size × distance interaction | `K × log(min_dist)` | cluster-robust OLS, clusters = cell × seed (2,800 rows) | MixedLM | 2,800 |

</details>

Full machine-readable payload (18 cells + per-arm context + reproducibility metadata): [`sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.json)

### Generated

n/a — no text generations in this analysis-only task; each cell yields one regression fit (coefficient, SE, p, CI, convergence flags), not generated text. The 18-cell table is the output; the MixedLM cells additionally carry `converged` / `boundary_variance` / captured fit warnings.

<details>
<summary>4 of 18 cells, cherry-picked to show one flip, one stable null, the singular persona-VC fit, and the sign-flipped alt-VC fit (all 18 rows: linked CSV/JSON below)</summary>

```
SET-SIZE x DISTANCE, centered join — significance swap
  cluster_ols:    beta=+0.02148  se=0.00638  p=0.000753  CI=[+0.0090,+0.0340]  N=2800 clu=80  -> significant
  mixedlm:        beta=+0.02148  se=0.01733  p=0.215      CI=[-0.0125,+0.0554]  N=2800 clu=40  -> null
  (point estimate identical to ~12 digits; only SE/p move)

ON-AXIS DOSE-MATCHED GAP, raw join — stable null
  cluster_ols:    beta=+0.19956  se=0.12223  p=0.1026  CI=[-0.040,+0.439]  N=255 clu=24  -> null
  mixedlm:        beta=+0.19956  se=0.13292  p=0.1333  CI=[-0.061,+0.460]  N=255 clu=24  -> null

LEAVE-ONE-OUT POOLED, raw join — persona-VC singular, groups-only-REML sign-flipped
  cluster_ols:    beta=-2.69339  se=1.03973   p=0.00958   CI=[-4.731,-0.656]  N=936 clu=18  -> significant (negative)
  mixedlm:        FAILED — persona-VC non-PD Hessian; degenerate SE=2.3e-04; inference not trustworthy
  mixedlm_alt_vc: beta=+0.83118  se=0.27431   p=0.00244   CI=[+0.294,+1.369]  N=936 clu=18  -> significant (POSITIVE; sign flip vs cluster_ols)

LEAVE-ONE-OUT POOLED, centered join — consistent null across admissible estimators
  cluster_ols:    beta=+0.00832  se=0.14980   p=0.9557  CI=[-0.285,+0.302]  N=936 clu=18  -> null
  mixedlm_alt_vc: beta=+0.05489  se=0.03980   p=0.1678  CI=[-0.023,+0.133]  N=936 clu=18  -> null
```

</details>

Full per-cell table (18 data rows): [`sweep_results.csv`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.csv)

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Model | none (CPU linear algebra + `statsmodels` re-fits) |
| Estimators swept | cluster-robust OLS (`cov_type='cluster'`) ↔ persona-RE MixedLM (`reml=False`, `lbfgs`, `vc_formula={'persona':'0+C(persona)'}`) |
| #505 alternative VC | groups-only random intercept on `j_i|seed`, REML, NO persona VC (`smf.mixedlm(formula, df, groups).fit(reml=True, lbfgs)`) — the simpler admissible spec re-fit over the same pooled join when the persona-VC cell is singular |
| Joins | raw (`centering='none'`), mean-centered (`centering='global_mean'`) |
| α per row | #405: 0.01; #490 / #505 / #478-interaction: 0.05 |
| Join-validity gate | point-estimate reproduction within 0.02 (statistic-level); 1e-4 matrix/row-level inside the parent `family_*` join builders |
| Multiplicity | none within-row (each row keeps its published α + family rule); 18-cell grid flagged as exploratory robustness, not 18 confirmatory tests |
| Cells | 4 reads × 2 estimators × 2 joins = 16, + 2 #505 alternative-VC cells = 18 |

**Artifacts:**

- Per-cell table (18 data rows, 18 columns): [`sweep_results.csv`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.csv)
- Machine-readable payload + per-arm #505 context + `mixedlm_alt_vc_505` estimator-spec note + reproducibility metadata: [`sweep_results.json`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.json)
- Figures (PNG + PDF + meta.json): [`figures/issue_589/`](https://github.com/superkaiba/explore-persona-space/tree/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/figures/issue_589)
- Reused panel-regression-secondary join from [#405](https://eps.superkaiba.com/tasks/405): [`eval_results/issue_405/aggregate/per_cell_persona_tidy.csv`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_405/aggregate/per_cell_persona_tidy.csv) + [`regression.json`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_405/aggregate/regression.json) — fit: the parent's gated per-cell persona join + published-estimator point estimate for the `min_dist` secondary refit (re-fit under both estimators, both axes).
- Reused on-axis dose-matched-gap join from [#490](https://eps.superkaiba.com/tasks/490): [`eval_results/issue_490/aggregate/persona_level.csv`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_490/aggregate/persona_level.csv) + [`regression.json`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_490/aggregate/regression.json) — fit: the parent's gated persona-level join + published cluster-robust point estimate for the `is_on_axis` dose-matched-gap refit.
- Reused leave-one-out join from [#505](https://eps.superkaiba.com/tasks/505): [`eval_results/issue_505/analysis/delta_leakage_per_seed.json`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_505/analysis/delta_leakage_per_seed.json) + [`per_arm_slopes.json`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_505/analysis/per_arm_slopes.json) — fit: the per-seed leakage-change join behind the pooled `cos(b,j)` refit + the per-arm slopes panel; the persona-VC MixedLM is singular here, motivating the round-2 groups-only-REML alt-VC cell.
- Reused set-size-by-distance interaction join from [#478](https://eps.superkaiba.com/tasks/478) (the parent's persisted #478 tidy): [`eval_results/issue_536/inputs/i478_tidy_69b34b94.csv`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_536/inputs/i478_tidy_69b34b94.csv) (2,800 rows) — fit: the gated `K × log(min_dist)` interaction join reproducing the parent's MixedLM point estimate before the cluster-robust swap is read.
- Reused recompute machinery from [#536](https://eps.superkaiba.com/tasks/536) at commit `12853bca8`: the driver's `family_111bank` / `family_20bank` / `family_505` join builders + `cluster_ols` helper + `issue536_mixedlm_refit.fit_published_mixedlm` (the #478-cell spec, verbatim) — fit: the sweep imports these to reproduce each row's persisted join + the parent's 1e-4 matrix / statistic-level gates. The 111-bank distance matrix [`eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json`](https://github.com/superkaiba/explore-persona-space/blob/45fe33f85/eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json) must be restored from git `45fe33f85` (absent at HEAD) before the driver runs.

**Compute:** CPU only, VM-side, deterministic, minutes. No pod, 0 GPU-hours.

**Code:**

- Sweep driver: [`scripts/issue589_estimator_sweep.py`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/scripts/issue589_estimator_sweep.py) (imports `issue536_recompute_driver` + `issue536_mixedlm_refit`)
- Figure script: [`scripts/issue589_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/scripts/issue589_figures.py)
- Env: statsmodels 0.14.6, scipy 1.17.1, numpy 2.2.6, pandas 2.3.3, python 3.11.15
- Reproduce:

    ```bash
    git clone https://github.com/superkaiba/explore-persona-space.git
    cd explore-persona-space && git checkout 92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce
    uv sync
    git checkout 45fe33f85 -- eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json
    uv run python scripts/issue589_estimator_sweep.py --data-root "$PWD"
    uv run python scripts/issue589_figures.py
    ```

**Context:**

- Created 2026-06-11T05:57:27Z; run executed 2026-06-13 (16-cell sweep), with a 9a-ter free-analysis follow-up the same day adding the 2 #505 alternative-VC cells.
- Follow-up to [#536](https://eps.superkaiba.com/tasks/536) — the persona-distance audit whose four clustered leakage-line calls this sweep re-fits under an alternative uncertainty estimator. The 9a-ter follow-up (`followup_label: issue589-505-altvc-refit`) re-fit the one singular read under a simpler admissible MixedLM spec.
- Originating prompt: origin prompt not recorded

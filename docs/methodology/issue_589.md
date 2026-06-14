# Methodology — issue 589: estimator-fragility sweep of #536's four clustered leakage-line reads

A methodology + statistical-knob reference for analysis task #589 (Explore
Persona Space), with verbatim per-cell output rows pulled straight from the
artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/589](https://eps.superkaiba.com/tasks/589)
- Model: none — CPU linear algebra + `statsmodels` re-fits over persisted joins

---

## 1. Overview

- **Goal (verbatim, plan §1):** "Determine whether the published significance calls of the clustered leakage-line reads (the panel-regression secondary, the on-axis dose-matched gap, the leave-one-out pooled null, and the per-set-size correlations) are robust to the uncertainty-estimator choice by refitting each task's own headline test under both the cluster-robust OLS and the mixed-effects (persona random effect) estimators on the identical persisted joins, on both raw and centered distance axes."
- **The single manipulation:** the uncertainty estimator (cluster-robust OLS ↔ persona-RE MixedLM). Data, joins, point estimates, the join-validity gates, the row set, and α-per-row are held identical to parent #536.
- **Design cells:** 4 clustered leakage-line rows × 2 estimators × 2 distance joins = 16 cells, + 2 #505 alternative-VC cells (groups-only REML) = **18 data rows**.
- **Dependent variable (per cell):** the refit coefficient + Wald 95% CI + p-value + `n_rows` + `n_clusters` + convergence flags for each row's own headline term.
- **No model, no judge:** this is deterministic numerical linear algebra + `statsmodels` over prior tasks' persisted measurement tables; there is no LLM call.
- **Provenance:** the four rows + their joins + the cluster-robust helper + the #478 MixedLM template are reused verbatim from #536's recompute machinery at commit `12853bca8` (`scripts/issue536_recompute_driver.py` + `scripts/issue536_mixedlm_refit.py`). The sweep `import`s them; it does not redefine the registry, joins, or gates.

---

## 2. Hyperparameters (load-bearing statistical knobs)

No ML hyperparameters (no training). The load-bearing knobs are statistical and
all inherited from #536. One complete table.

| Parameter | Value | Source |
|---|---|---|
| Model / training | none (CPU `statsmodels` re-fits) | run plan §1, §9 |
| **Estimator A — cluster-robust OLS** | `sm.OLS(y, sm.add_constant(X)).fit(cov_type='cluster', cov_kwds={'groups': clusters})`; Wald 95% CI = `res.conf_int()` | `drv.cluster_ols` @ `12853bca8` (`issue536_recompute_driver.py:172`); Cameron & Miller 2015 |
| **Estimator B — persona-RE MixedLM** | `smf.mixedlm(formula, df, groups=df[groups_col], vc_formula={'persona': '0 + C(<persona_col>)'}).fit(reml=False, method='lbfgs')` | sweep driver `_fit_mixedlm` @ `92318460b`; #536 / #478 |
| **#405 MixedLM cell (published)** | two VCs `{'subset':'0+C(subset)','persona':'0+C(persona)'}`, dummy single group, `reml=True`, lbfgs | `regrade_405._mixed` verbatim; #536 |
| **#478 MixedLM cell (published)** | `issue536_mixedlm_refit.fit_published_mixedlm` verbatim: `groups=subset_id`, `vc={'persona':'0+C(held_out_persona)'}`, `reml=False`, lbfgs | `issue536_mixedlm_refit.py:114` @ `12853bca8` |
| **#505 alternative VC (9a-ter)** | `smf.mixedlm('delta_leakage ~ cos_bj', df, groups=df['cluster']).fit(reml=True, method='lbfgs')` — groups-only random intercept on `j_i\|seed`, NO persona VC | sweep driver `_fit_mixedlm_groups_only_reml` @ `92318460b`; King & Roberts 2015 |
| Joins (distance axes) | raw `1 - compute_cosine_matrix(C, centering='none')`; mean-centered `centering='global_mean'` | #536 family builders |
| **α per row** | #405: `0.01`; #490 / #505 / #478-flatness-null: `0.05` | `ROW_ALPHA` @ `92318460b`; #536 per-row published threshold |
| Manipulation-check tolerance | two-tier: 1e-4 matrix/row-level (inside `family_*` / `build_joined_df`, raises pre-read) + statistic-level `0.02` on the published-estimator raw cell vs persisted coefficient (`MC_STATISTIC_TOL`) | #536 `GATE_MATRIX_TOL`, `GATE_BETA_TOL=0.005`; plan §4.5 |
| #478 MixedLM gate constants | same-sign, `\|β − PUB_BETA\| ≤ 0.005`, `PUB_BETA = 0.010` | `issue536_mixedlm_refit.py:67,71` |
| Multiplicity | none within-row (each row keeps its published α + family rule); 18-cell grid flagged exploratory robustness, not 18 confirmatory tests | plan §6.6; #536 per-row α stance |
| Singular-boundary rule (MixedLM) | non-PD-Hessian warning OR degenerate `se < 1e-3` OR `res.converged=False` ⇒ `status: FAILED`, never a fallback to another estimator | sweep driver `_fit_mixedlm` @ `92318460b`; plan §4.3 |
| Seeds | n/a (deterministic; data carry seeds 42/137/219 as a clustering variable, not a run seed) | plan §10 |
| Env | statsmodels 0.14.6, scipy 1.17.1, numpy 2.2.6, pandas 2.3.3, python 3.11.15 | Reproducibility slice |

Bold rows are the load-bearing knobs a re-implementer needs first. The body
`## Reproducibility` Parameters table is a subset of this table.

---

## 3. Training data

**No training in this task.** This is a re-fit of prior tasks' persisted
regression joins under an alternative variance estimator — no model is fine-tuned,
no corpus is built, no rows are generated. The "data" are the four already-gated
joins #536 persisted at commit `12853bca8`; the sweep reconstructs each row's
`(y, X-by-join, cluster, persona)` frame by calling into `drv.family_*` exactly as
the parent adapter built it, then fits both estimators on that frame.

The input joins, their row counts, and the centroid bank each rides on:

| Row (input join) | N rows | Bank | Persisted source |
|---|---|---|---|
| #405 `per_cell_persona_tidy.csv` (CORE track) | 336 | 20-bank L20 (`extraction_method_a`) | `eval_results/issue_405/aggregate/{per_cell_persona_tidy.csv, regression.json}` |
| #490 `persona_level.csv` (on/off-axis pivot) | 255 | 111-bank L20 | `eval_results/issue_490/aggregate/{persona_level.csv, regression.json}` |
| #505 `delta_leakage_per_seed.json` (pooled LOO) | 936 | 505 bank L21 (`centroids_pv_L21.pt`) | `eval_results/issue_505/analysis/{delta_leakage_per_seed.json, per_arm_slopes.json}` |
| #478 `i478_tidy_69b34b94.csv` (K × log-dist) | 2,800 | 111-bank L20 | `eval_results/issue_536/inputs/i478_tidy_69b34b94.csv` |

The 111-bank distance matrix `eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json` is absent at HEAD and must be restored from git `45fe33f85` before the driver runs (validated by the parent's 1e-4 matrix gate).

---

## 4. Evaluation (DV + metric)

The dependent variable IS a re-computation of each published test — the metric and
the construct coincide; there is no behavior being proxied.

- **Per (row × estimator × join):** the refit **coefficient** of the row's headline term, its **Wald 95% CI** (`res.conf_int()` for both estimators), the **p-value**, `n_rows`, `n_clusters`, plus MixedLM `converged` / `boundary_variance` flags. Stored four-column per cell in the CSV; full payload + per-arm context + reproducibility metadata in the JSON.
- **Manipulation-check gate (per row, before any swept p is read):** the published-estimator raw cell must reproduce #536's persisted point estimate within tolerance (`manipulation_check_ratio = |refit − pub|`; statistic-level 0.02, after the 1e-4 matrix gate already raised inside the join builders). A failing gate marks the row's cells `inconclusive (join_bug)`; the run reports it, never papers over it.
- **`call_swept` classification (`classify_call`):** `significant` if `p < α` (status OK and converged), `null` if `p ≥ α`, `inconclusive` if `status != OK` or MixedLM `converged is False`.
- **`call_flips`:** the alternative (swept) estimator's call disagrees with the published-estimator cell on the SAME join (only read for the alternative estimator; the published cell is False by construction; a join_bug forces no flip read).
- **Sign-flip discipline (`_sign_flip`):** flagged separately and read as graver than a p-only flip — the coefficient sign differs between the two estimators on the same join (`math.copysign(1, co) != math.copysign(1, cm)`).
- **Multiplicity:** no within-row re-thresholding; the 16-cell grid is exploratory robustness, not 16 confirmatory tests.

Per-row clustering structure, published vs alternative estimator, and N (read
from the build functions):

| Row | Clustering / RE structure | Published estimator | Alternative estimator | N rows | n_clusters |
|---|---|---|---|---|---|
| `405-secondary` | MixedLM two VCs `{subset, persona}`, dummy single group, REML | MixedLM | cluster-robust OLS clustered on `subset` | 336 | 21 (OLS) / 1 (MixedLM dummy group) |
| `490-distance-adjusted` | cluster-robust OLS clusters = `pair_id\|seed`; MixedLM groups = `pair_id\|seed`, persona VC | cluster-robust OLS | persona-RE MixedLM | 255 | 24 |
| `505-loo-null` | pooled cluster-robust OLS clusters = `j_i\|seed`; MixedLM groups = `j_i\|seed`, persona VC `0+C(b)`; per-arm OLS(HC2) (not swept) | cluster-robust OLS (pooled stand-in) | persona-RE MixedLM (+ groups-only-REML alt-VC) | 936 | 18 |
| `478-flatness-null` | cluster-robust OLS clusters = `cell_id\|seed`; MixedLM `fit_published_mixedlm` (groups = `subset_id`, persona VC) | MixedLM (published co-primary) | cluster-robust OLS | 2,800 | 80 (OLS) / 40 (MixedLM groups) |

`#490` (24 clusters) and `#505`-pooled (18 clusters) sit below Cameron & Miller's
~30–50 cluster floor — flagged in the JSON `small_cluster_note`. The per-K
correlations (the Goal's fourth named read) are quoted from #536, not swept: a
per-stratum Spearman/OLS has no cluster/RE structure, so the estimator axis is
undefined for them (plan §4.1).

---

## 5. Worked examples (verbatim cells)

Three cells straight from `eval_results/issue_589/sweep_results.csv`, showing the
18-column schema. Quoted as "what the output looks like" — not as findings.

CSV header:
```
row_id,estimator,join,coefficient,se,df,p_value,ci_lo,ci_hi,n_rows,n_clusters,manipulation_check_ratio,converged,boundary_variance,call_published,call_swept,call_flips,_vc_spec
```

<!-- cherry-picked for schema illustration; full 18-row table at https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.csv -->

One **cluster-OLS** cell (`490-distance-adjusted`, raw join — `converged` / `boundary_variance` empty because OLS has no random effect):
```
490-distance-adjusted,cluster_ols,raw,0.19955644077671642,0.12223256448449075,251.0,0.10255391570842698,-0.04001498335085518,0.439127864904288,255,24,4.257685201014549e-06,,,null,null,false,persona_vc
```

One **MixedLM** cell (`478-flatness-null`, centered join — `converged=true`, `df` empty for the MixedLM Wald CI):
```
478-flatness-null,mixedlm,centered,0.021480451665166354,0.017326921734990692,,0.21508067997063018,-0.01247969089835967,0.05544059422869238,2800,40,0.000255658098125746,true,false,null,null,false,persona_vc
```

One **mixedlm_alt_vc** cell (the 9a-ter groups-only-REML re-fit of `505-loo-null`, raw join — `_vc_spec=groups_only_reml`):
```
505-loo-null,mixedlm_alt_vc,raw,0.8311815695322085,0.2743075073866817,934.0,0.0024446572250289168,0.29354873436535744,1.3688144046990596,936,18,1.7204762161604492e-06,true,false,null,significant,false,groups_only_reml
```

The JSON `estimator_specs.mixedlm_alt_vc_505` block, verbatim:
```json
"mixedlm_alt_vc_505": "smf.mixedlm('delta_leakage ~ cos_bj', groups=j_i|seed).fit(reml=True, lbfgs) [9a-ter ALTERNATIVE VC — groups-only random intercept, REML, NO persona variance component; the simpler admissible spec re-fit over the SAME pooled join when the persona-VC cell is singular]"
```

The #505 per-arm OLS(HC2) structural-context block (`per_arm_505`, centered join, verbatim from the parent `regrade_505._per_arm` — quoted, not swept):
```json
"ai_assistant": {"beta_j": 0.21155560550024127, "p": 0.034124571746644204, "ci95": [0.01584049679772412, 0.4072707142027584], "n_rows": 156},
"child":        {"beta_j": -0.7497909295361325, "p": 1.9011646442453916e-05, "ci95": [-1.0934525077540327, -0.40612935131823213], "n_rows": 156},
"hero":         {"beta_j": 0.18727679840762693, "p": 0.038459861199983826, "ci95": [0.009948289861148235, 0.36460530695410565], "n_rows": 156}
```

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Per-cell table (18 data rows, 18 columns) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.csv) |
| Machine-readable payload (+ `estimator_specs`, `per_arm_505`, reproducibility metadata) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_589/sweep_results.json) |
| Figures (PNG + PDF + meta.json) | [GitHub](https://github.com/superkaiba/explore-persona-space/tree/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/figures/issue_589) |
| Sweep driver | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/scripts/issue589_estimator_sweep.py) |
| Figure script | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/scripts/issue589_figures.py) |
| Reused recompute driver (#536) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/12853bca8/scripts/issue536_recompute_driver.py) |
| Reused MixedLM template (#536/#478) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/12853bca8/scripts/issue536_mixedlm_refit.py) |
| #405 reused join | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_405/aggregate/per_cell_persona_tidy.csv) |
| #490 reused join | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_490/aggregate/persona_level.csv) |
| #505 reused join | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_505/analysis/delta_leakage_per_seed.json) |
| #478 reused tidy (via #536 inputs) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce/eval_results/issue_536/inputs/i478_tidy_69b34b94.csv) |
| 111-bank distance matrix (restore from git before run) | [GitHub @ 45fe33f85](https://github.com/superkaiba/explore-persona-space/blob/45fe33f85/eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json) |
| WandB run(s) | n/a (CPU re-fit; no training, no WandB run) |
| Code commit | `92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce` |
| Parent data pin | `12853bca8` (#536 gated joins) |
| Compute | CPU only, VM-side, deterministic, minutes; no pod, 0 GPU-hours |

Reproduce:
```bash
git clone https://github.com/superkaiba/explore-persona-space.git
cd explore-persona-space && git checkout 92318460bbe6ad1a78f0c0e43fb04e232bc3d5ce
uv sync
git checkout 45fe33f85 -- eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json
uv run python scripts/issue589_estimator_sweep.py --data-root "$PWD"
uv run python scripts/issue589_figures.py
```

---

*This document describes how the experiment was run. For the result and what it
means, see the [task body](https://eps.superkaiba.com/tasks/589).*

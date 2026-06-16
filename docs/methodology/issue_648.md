# Methodology — issue 648: head-to-head predictive skill (CV R²) of raw vs centered persona-distance cosine, per recoverable #536 bank

A methodology + analysis-constant reference for experiment #648 (Explore
Persona Space), with verbatim panel-row and CV-iteration examples pulled
straight from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/648](https://eps.superkaiba.com/tasks/648)
- Model: none (CPU-only re-analysis of base-model `Qwen/Qwen2.5-7B` centroid banks #536 extracted)

---

## 1. Overview

- **What it computes:** for each recoverable predictor bank from parent
  #536, the cosine-distance leakage predictor is scored two ways — raw
  (`centering='none'`) and mean-centered (`centering='global_mean'`) — and
  two paired differences are reported: a PRIMARY out-of-sample leave-one-group-out
  CV R² difference (`ΔR² = R²_centered − R²_raw`) and a SECONDARY in-sample
  length-partialled Spearman difference (`Δρ`).
- **The single manipulated variable:** the cosine centering recipe (raw vs
  centered). Cells, leakage target, length residualization, CV partition,
  and ρ estimator are held constant across the two recipes within each bank.
- **Design cells:** 9 recoverable exact-row banks (#66, #142, #405, #478,
  #490, #380, #396/#415, #311, #505); per bank, the implicit raw-vs-centered
  pair. Verdict-eligibility (`contributes_to_h_verdict`) is a mechanical
  per-row flag; 4 of 9 banks are eligible.
- **Dependent variables:** continuous per-cell leakage target inherited
  verbatim from each producing task (marker leakage rate, `deltaLogP_mean`,
  dose-matched gap, `source_rate`, `r_p_primary_per_persona`, `delta_leakage`).
- **No judge, no model call.** Every input is a numeric tensor / numeric JSON;
  the analysis is closed-form linear algebra + Spearman + a paired cluster
  bootstrap.
- **Provenance:** the centroid banks, leakage targets, join builders
  (`family_*`), `length_partial_spearman`, and the `affected_set` recoverable-bank
  gate are reused verbatim from #536's `issue536_recompute_driver.py`.

---

## 2. Hyperparameters

**N/A — no model training.** The load-bearing analysis constants (RNG seed,
bootstrap B, variance floor, verdict-eligibility thresholds, per-bank
covariates) live in §4 Evaluation.

---

## 3. Training data

**N/A — no training mix.** This task trains nothing and generates no new
data. It re-reads already-collected cached artifacts: #536's base-model
centroid banks (`.pt` tensors) plus each producing task's persisted
continuous leakage target. The input artifacts are catalogued in §4
Evaluation and §6 Artifacts index.

---

## 4. Evaluation

### Dependent variables

- **PRIMARY — `ΔR² = CV R²_centered − CV R²_raw`** per bank. Construct:
  which recipe predicts held-out leakage better. Metric: leave-one-group-out
  (LOGO) CV R² of a length-partialled linear predictor of the bank's leakage
  target, with the centering mean **refit on TRAIN-fold personas inside each
  split** (train-fold-only centering), `SS_tot` pinned to the **per-fold
  TRAIN-mean baseline**, differenced across recipes. On-distribution: the
  held-out group is the bank's natural unit and the predictor (including its
  centering mean) never sees the held-out unit.
- **SECONDARY — `Δρ = ρ_centered − ρ_raw`** per bank. Length-partialled
  Spearman of cosine distance vs the leakage target, **full-bank (bank-global)
  centering** — explicitly labeled `transductive` / in-sample in the output
  schema (`rho_regime = "transductive_in_sample_bank_global_centering"`). This
  is the read tied to #536's `raw_vs_centered_x_spearman` survival family.

### Recipe mechanics

- **Centering switch:** `compute_cosine_matrix(C, centering=...)` —
  `'none'` is a no-op (raw, no mean subtracted); `'global_mean'` subtracts
  `C[centering_rows].mean(0)` from all rows, then unit-normalizes
  (`representation_shift.py`). Raw ignores `centering_rows` → bit-identical
  to the bank-global raw predictor, preserving the single-variable contract.
- **LOGO CV partition:** the "group" is each bank's natural held-out unit
  (the same statistical cluster the producing task used). For each held-out
  group `g`: TRAIN = cells whose original group ≠ g; the centering universe
  is `panel.centering_bank_idx ∩ personas-present-in-TRAIN` (held-out
  personas excluded from the mean); length-residualize x and y on the
  covariate using TRAIN-fold OLS coefficients; fit `y_resid ~ x_resid` (OLS)
  on TRAIN, apply to TEST; store `(y_true − train_mean, y_hat − train_mean)`.
- **Paired cluster bootstrap:** 10,000 resamples of original groups with
  replacement; raw and centered scored on the IDENTICAL group draw (pairing
  cancels resample noise); percentile 95% CI on each delta; RNG seed 20648.
  A group drawn twice contributes its cached held-out errors twice — never
  scored against a training copy of itself (per-fold `isdisjoint` assert,
  both recipes).
- **Join-sanity gate (per bank, before any skill is read):** each bank's
  assembler runs #536's family loader and reproduces #536's published
  statistic within tolerance (`GATE_MATRIX_TOL = 1e-4` matrix gate, or
  `GATE_RHO_TOL = 0.02` statistic gate, both imported verbatim from
  `issue536_recompute_driver`).
- **Verdict-eligibility (mechanical):** `contributes_to_h_verdict = true`
  requires ALL of (a) `n_groups > 5`; (b) not `both_predictors_fail_oos`
  (both `cv_r2 < 0`); (c) ≤25% folds degenerate-skipped. Precedence
  (single deterministic `exclusion_reason`): `both_predictors_fail_oos` →
  `low_n_groups<=5` → `degenerate_folds>25pct` → `null`. `main` asserts no
  `n_groups ≤ 5` row claims eligibility.

### Analysis constants

| Constant | Value | Source |
|---|---|---|
| RNG seed | `20648` | driver @ `4ff0a15c43` (`RNG_SEED`), plan §11 |
| Bootstrap resamples (`N_BOOT`) | `10000` | driver @ `4ff0a15c43`, plan §11 (Efron & Tibshirani 1993, ≥10k percentile-CI floor; matches #514 convention) |
| Bootstrap CI | percentile 95% (2.5 / 97.5) | driver @ `4ff0a15c43` |
| Variance floor (`VAR_FLOOR`) | `1e-8` | driver @ `4ff0a15c43`; plan §11 `ungrounded — needs smoke-test` (float64 unit-scale numerical guard) |
| Low-N verdict floor (`LOWN_VERDICT_FLOOR`) | `n_groups ≤ 5` → non-contributing | driver @ `4ff0a15c43`, plan §6/§11 (MF3) |
| Degenerate-fold downgrade (`DEGEN_FOLD_FRAC`) | `> 25%` folds skipped → non-contributing | driver @ `4ff0a15c43`; plan §11 `ungrounded — needs smoke-test` |
| Matrix join gate (`GATE_MATRIX_TOL`) | `1e-4` | imported from `issue536_recompute_driver` |
| Statistic join gate (`GATE_RHO_TOL`) | `0.02` | imported from `issue536_recompute_driver` |
| Recoverable-bank gate | `eval_results/issue_536/audit_table.json::affected_set` | plan §11 (MF1 accounting gate) |
| CV scheme | leave-one-group-out (LOGO), group = bank's natural unit | plan §11 (ESL §7) |
| Centering — primary DV | train-fold-only (mean refit per CV split) | driver @ `4ff0a15c43`, plan §11 (MF5; ESL §7.10.2) |
| Centering — secondary DV | bank-global (transductive / in-sample) | driver @ `4ff0a15c43`, plan §11 (MF5) |
| `SS_tot` definition | per-fold TRAIN-mean baseline | driver @ `4ff0a15c43`, plan §11 (MF4) |

### Banks (recoverable exact rows)

| Bank | `family` | Layer | Held-out unit (LOGO group) | n_groups | n_cells | ρ method / covariate | Eligible? |
|---|---|---|---|---|---|---|---|
| 100-persona (#66) | `single_token_100p_L20` | L20 | leave-one-source-out | 5 | 550 | plain / none | No — n≤5 |
| core-11 subset (#142) | `single_token_100p_core11` | L20 | leave-one-source-out | 5 | 50 | plain / none | No — n≤5 |
| 20-bank (#405) | `extraction_method_a_L20` | L20 | leave-one-held-persona-out | 8 | 336 | plain / none | Yes |
| 111-bank (#478) | `single_token_100p_L20` | L20 | leave-one-held-persona-out | 35 | 2800 | plain / none | Yes |
| 111-bank (#490) | `single_token_100p_L20` | L20 | leave-one-persona-out | (on disk) | (on disk) | plain / none | Yes if >5 |
| n24 (#380) | `issue274_n24_L15` | L15 | leave-one-persona-out | 24 | 24 | rank_residual / `log_tokens` | Yes |
| n24-predictor (#396/#415) | `issue274_n24_L15` | L15 | leave-one-source-out | 24 | 24 | length_partial / inherited-prompt length | Yes |
| 19-bank (#311) | `issue311_19bank_L20` | L20 | leave-one-bystander-out | 17 | 17 | value_residual / `s = ½(cos(p,A)+cos(p,B))` | Yes |
| 505 PV (#505) | `issue505_pv_L21` | L21 | leave-one-bystander-out | 52 | 936 | plain / none | Yes |

Notes: #142's centering universe is the CORE-11 subset (NOT the full 111
bank). #396 and #415 share one n24 join + DV; reported once (#415 documented
as the corroborating-duplicate in the MF1 accounting gate). Matrix-only banks
(#474/#406/#460/#341) get labeled `sensitivity_namespace` rows only — approximate
ρ deltas, no CV R², never pooled with the exact rows.

### Verbatim leakage-target read paths (per bank)

- #66 / #142: `eval_results/single_token_100_persona/<src>/marker_eval.json[tgt].rate`
- #405: `eval_results/issue_405/aggregate/per_cell_persona_tidy.csv` (`deltaLogP_mean`, CORE track)
- #478: `eval_results/issue_536/inputs/i478_tidy_69b34b94.csv` (`deltaLogP_mean`)
- #490: `eval_results/issue_490/aggregate/persona_level.csv` (dose-matched gap `shared_2D − ½(pooled_2D_A + pooled_2D_B)`)
- #380: `eval_results/issue_380/cosine_pairwise_n24/correlation.json` (`source_rate`)
- #396/#415: `eval_results/issue_396/analysis_summary.json` + `eval_results/issue_415/base_model_predictors_v2.json` (`logp_end_of_response_diagonal_mean`)
- #311: `eval_results/issue_311/analysis.json` (`r_p_primary_per_persona`) + `pair_selection.json`
- #505: `eval_results/issue_505/analysis/delta_leakage_per_seed.json` (`delta_leakage`)

---

## 5. Worked examples — 505 PV bank at L21 (largest panel)

Representative eligible bank: `issue505_pv_L21`, `cv_unit =
leave-one-bystander-out`, `n_groups = 52`, `n_cells = 936` (52 bystanders ×
6 arms × 3 seeds). Predictor = cosine **similarity** `cos(b, j_i)`
(`use_similarity=True` — #505 regresses on `cos`, not `1 − cos`); no length
covariate (`has_covar=False`).

**One verbatim panel row** read from `delta_leakage_per_seed.json` (the
join-structural fields the assembler consumes: bystander `b`, source/jailbreak
`j_i`, seed, and the continuous target `delta_leakage = Y`):

```json
<!-- cherry-picked (rows[0]) for illustration; full 936-row panel at
     https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/eval_results/issue_505/analysis/delta_leakage_per_seed.json -->
{
  "b": "accountant",
  "j_i": "hero",
  "j_idx": 0,
  "seed": 42,
  "delta_leakage": -0.07373104095458949
}
```

The assembler maps this to a panel cell: `cell_idx = (idx["accountant"],
idx["hero"])` indexing into the PV centroid bank `C`, `group = bystander b`,
`y = delta_leakage`.

**Per-cell predictor pair (raw vs centered), same cell:**

```
cell (b=accountant, j_i=hero):
  x_raw = cos( C[accountant]^,            C[hero]^ )          # centering='none'  -> no mean subtracted
  x_cen = cos( (C[accountant]-mu)^,       (C[hero]-mu)^ )      # centering='global_mean'
                                                              #   mu = C[centering_rows].mean(0)
                                                              #   centering_rows = train-fold personas (primary DV)
                                                              #                  = full bank (secondary in-sample rho)
  y     = -0.0737...   group = "accountant"   (no length covariate)
```

**LOGO CV inner loop (`_logo_group_contributions`), per recipe:**

```
for each held-out bystander group g in unique(group):          # 52 folds
    tr = group != g ;  te = group == g
    assert disjoint(group[te], group[tr])                       # MF2 per-fold guard
    if te empty or |tr| < 3: skip + count
    centering_rows = centering_bank_idx ∩ personas-in(tr)       # train-fold-only (centered); ignored for raw
    x  = cosine_predictor(panel, centering_rows, recipe)
    # no covariate for #505 -> residualization skipped:
    xr_tr, yr_tr, xr_te, yr_te = x[tr], y[tr], x[te], y[te]
    if var(xr_tr) < 1e-8 or var(yr_tr) < 1e-8: skip + count
    m = OLS(xr_tr -> yr_tr)
    train_mean = mean(yr_tr)
    store (yr_te - train_mean, polyval(m, xr_te) - train_mean) for group g
R2 = 1 - sum(SS_res) / sum(SS_tot)        # SS_tot = sum(y_true_frame^2), per-fold train-mean baseline
```

**Paired bootstrap contract:** 10,000 resamples (RNG seed 20648); each draw
`gs = rng.choice(unique_groups, size=n_groups, replace=True)` is scored under
BOTH recipes from the cached per-group held-out contributions (with draw
multiplicity); `Δ = R²_centered − R²_raw` accumulated only when both are
finite; percentile [2.5, 97.5] CI. The in-sample Δρ bootstrap re-pools the
drawn groups' cells (with multiplicity) and applies the panel's per-bank ρ
method.

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Per-bank skill table (JSON) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/eval_results/issue_648/per_bank_skill_table.json) |
| Per-bank skill table (CSV mirror) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/eval_results/issue_648/per_bank_skill_table.csv) |
| Hero forest (ΔR²) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/d83bcc2566963184c2526133e07333236dc21add/figures/issue_648/hero_forest_delta_cvR2.png) |
| Paired raw-vs-centered R² (round 2) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/c745da0abc227b880b16fc0a94b19bac8f427607/figures/issue_648/paired_r2_raw_vs_centered.png) |
| Driver script | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/4ff0a15c43282bf8e6e88653898c98816b96efdc/scripts/issue648_centered_vs_raw_predictive_skill.py) |
| Figures script | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/c745da0abc227b880b16fc0a94b19bac8f427607/scripts/issue648_analyzer_figures.py) |
| Reused parent driver (loaders, gates, `length_partial_spearman`, `SOURCES_142`/`CORE11_142`) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/4ff0a15c43282bf8e6e88653898c98816b96efdc/scripts/issue536_recompute_driver.py) |
| Centering switch (`compute_cosine_matrix`) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/4ff0a15c43282bf8e6e88653898c98816b96efdc/src/explore_persona_space/analysis/representation_shift.py) |
| 111-bank distance JSON (read-only restore) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/776c7c3b758942f5719557fc69e1e2420af0c36b/eval_results/single_token_100_persona/cosine_distance_matrix_layer20.json) |
| #505 PV centroids (HF fallback) | [HF Hub](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/main/issue505_loo_contrastive/geometry/centroids_pv_L21.pt) |
| WandB run(s) | n/a — CPU-only analysis, no training |
| Code commit | `4ff0a15c43282bf8e6e88653898c98816b96efdc` (driver); figures `c745da0abc227b880b16fc0a94b19bac8f427607` |
| Reproducibility metadata | `data_root_commit cbae80d9ec`, `bank111_restore_sha 776c7c3b75`, `rng_seed 20648`, `n_boot 10000` |
| Reproduce | `uv run python scripts/issue648_centered_vs_raw_predictive_skill.py --data-root "$REPO_ROOT" --out-dir eval_results/issue_648 --fig-dir figures/issue_648` then `uv run python scripts/issue648_analyzer_figures.py` |
| Compute | CPU on the VM, < ~10 min single-process wall-clock. No pod, no GPU. |

---

*This document describes how the experiment was run. For the result and
what it means, see the [task body](https://eps.superkaiba.com/tasks/648).*

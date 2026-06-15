# Methodology — issue 637: held-out cross-validation of the leakage-transfer asymmetry rank (rank-1 scalar vs full pairwise)

A methodology + analysis-knob reference for experiment #637 (Explore
Persona Space), with verbatim source-data, predictor, and held-out
worked examples pulled straight from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/637](https://eps.superkaiba.com/tasks/637)
- Kind: `analysis` (parent #526), 0 GPU, local-CPU, no model loaded
- Model: N/A — read-only re-analysis of #537's pre-computed transfer matrices

---

## 1. Overview

- **Object of study:** the directional leakage-transfer matrix `M` (16×16 per behavior) from sibling #537 — `M[i, j]` is the leakage signal `g` measured when context `i` is the training/source context and `j` is the eval context.
- **The question:** decide how complex the directional-leakage predictor `g(source, eval, behavior)` must be — does the matrix's directional ASYMMETRY decompose as rank-1 (one per-context "source breadth" scalar `b_i` + one "receptivity" scalar `r_j`), or does it need a full pairwise interaction term? (plan §1)
- **The manipulation (single variable):** parent #526 answered this IN-SAMPLE (fit + score on the same cells). #637 re-answers it OUT-OF-SAMPLE — the one new thing is an 80/20 held-out cross-validation wrapper around #526's existing matrix-loading + additive-fit functions. No retraining, no data generation.
- **Design cells:** 5 behaviors (marker, taught fact, refusal, sycophancy, emergent misalignment), each with three nested predictors fit on the train split and scored on the held-out split, plus a shuffled-context control. Per-behavior off-diagonal = 240 cells → 192 train / 48 test.
- **Dependent variables:** held-out R² per arm and two paired differences `ΔR²_scalar = R²(sym+scalar) − R²(sym)` and `ΔR²_full = R²(full) − R²(sym+scalar)`, each with a 1000-bootstrap CI over held-out cells.
- **Provenance:** the upstream `g` cells, per-context norms, and in-sample anchor rows are reused verbatim — #537's `G_meta.json` + `g1_regression.json` and #526's `gate_ladder_results.json`. The fit + matrix-load functions are imported unchanged from `scripts/issue526_asym_gate_ladder.py`.

---

## 2. Analysis parameters

All training/generation hyperparameters are N/A — no model is trained or
loaded. The load-bearing knobs are the cross-validation analysis
parameters, each copied verbatim from the committed run's
`heldout_predictive_test.meta.json` (`params` block) or the script at the
run SHA. Bold = the knobs a re-implementer needs first.

| Parameter | Value | Source |
|---|---|---|
| Base model / training | N/A — no model loaded, 0 GPU | plan §0, §11 |
| Behaviors analyzed | `marker, fact, refusal, sycophancy, em` (5) | script `BEHAVIORS` @ `68fad5c` |
| Per-behavior off-diagonal cells | 240 (16×16 minus the 16 diagonal) | results JSON `n_offdiag_cells` |
| **Train/test split** | **80/20 per-CELL random split → 192 train / 48 test** | meta.json `split_frac: 0.8` |
| Split unit | individual off-diagonal cell `(i,j)` — NOT pair-split | plan §4, §11 (v2 revision) |
| **Bootstrap count** | **n = 1000, resampling the 48 held-out cells with replacement** | meta.json `n_bootstrap: 1000` |
| Bootstrap target | held-out R² per arm + paired ΔR² on the same resampled cells | script `bootstrap_arm_ci` / `paired_delta_ci` |
| **Split-stability loop** | **20 seeds (42–61); median + IQR of ΔR²_scalar / ΔR²_full** | meta.json `n_split_seeds: 20` |
| **Shuffled-context permutations** | **100; mean + [p5, p95] on the effect-size gap** | meta.json `n_shuffle_perm: 100` |
| Shuffled-control mechanism | `permute-one-axis-of-M` (row/source axis only) | meta.json `shuffled_control_mechanism` |
| **Random seed (base)** | **42** | meta.json `base_seed: 42` |
| Rank-1 antisym predictor | `Â_ij = s_i − s_j`, `s = (b − r)/2`, `(b,r)` from `fit_two_way_additive` on train cells | meta.json `predictor_formula` (v2) |
| Full-pairwise rule | `2·g_sym(i,j) − M[j,i]` when `(j,i) ∈ train`, else rank-1 fallback | meta.json `full_pairwise_rule` |
| Symmetric fit | LS `M_ij ~ μ + s_i + s_j` on train cells (single per-context scalar both sides) | script `symmetric_two_way_fit` |
| Held-out R² | `1 − SS_res / SS_tot` on the 48 test cells per arm | script `r2()` |
| Reproduction assert tolerance | 3 decimals (`< 1e-3`) on in-sample L0 + L2 vs the #526 anchor | script `assert_in_sample_reproduction` |
| Smoke config | `--smoke` → 1 behavior (marker), 20 bootstraps, 5 split-seeds, 20 perms | script `main()` |
| Env versions | python 3.11.15, numpy 2.2.6, scipy 1.17.1 | meta.json `env` |
| GPU-hours | 0 | plan §9 |

The body `## Reproducibility` Parameters table is a load-bearing subset
of this complete table.

---

## 3. Input data (no training data)

There is NO training step. The "data" is the set of pre-computed input
artifacts the analysis reads, plus the cell structure it derives from
them. Construction recipe (≤8 steps):

1. `load_537()` (REUSED from `scripts/issue526_asym_gate_ladder.py`) reads `eval_results/issue_537/G_tensor/G_meta.json` `per_cell` block; each key is `<behavior>/<train_ctx>__<eval_ctx>` carrying that cell's transfer value `g`.
2. Per behavior, restrict to the square block of contexts present on BOTH the train side and the eval side → a 16×16 matrix `M` with `M[i, j] = g(train_ctx i, eval_ctx j)`. Per-context L2² response norms come from `eval_results/issue_537/analysis/g1_regression.json` (`norms_l22_mean_response`).
3. A sanity gate asserts, per behavior, that every off-diagonal cell is present (0 NaN) and 0 cells are flagged saturated (`SAT`); a violation aborts.
4. `offdiag_cells(16)` enumerates the 240 off-diagonal `(i, j)` index pairs (`i ≠ j`).
5. `split_cells(cells, frac=0.8, seed=42)` permutes the 240 cells with `np.random.default_rng(42)` and takes the first 192 as train, last 48 as test.
6. The three predictors are fit on the 192 train cells (§4) and scored on the 48 test cells.
7. A Kill-4 reproduction assert recomputes the in-sample antisym fractions on ALL off-diagonal cells and fails fast if they drift from the #526 anchor `gate_ladder_results.json['537'][behavior]` by ≥ 3 decimals.
8. Outputs (`heldout_predictive_test.{json,meta.json}`) record `s`-derived predictors, `n_full_fallback`, the bootstrap CIs, and a sha256 of each input file.

Input artifacts + per-behavior cell composition:

| Input artifact | Role | Per behavior |
|---|---|---|
| `eval_results/issue_537/G_tensor/G_meta.json` | per-cell transfer values `g` | 16 contexts → 256 cells, 240 off-diagonal |
| `eval_results/issue_537/analysis/g1_regression.json` | per-context L2² response norms | 16 contexts |
| `figures/issue_526/gate_ladder_results.json` | in-sample anchor rows (L0 / L2 fractions) to reproduce | 5 behaviors |
| Realized split | 80/20 per-cell, seed 42 | 192 train / 48 test |
| `n_full_fallback` (marker, seed 42) | test cells whose transpose is also held out → rank-1 fallback | 14 of 48 |

Data-source tier: Tier 2 (established within-project eval artifacts,
reused verbatim; the realism of the underlying `g` measurements is
#537's scope, carried as provenance). Full input matrices at the run
SHA: [issue_537 G_tensor](https://github.com/superkaiba/explore-persona-space/tree/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/eval_results/issue_537/G_tensor).

### Verbatim source-data example (the matrix cell shape)

<!-- cherry-picked for illustration; full per-cell data at the issue_537 G_tensor link above -->

```text
# load_537()["marker"] structure (16x16):
#   contexts[15] = "wc_short_code"   contexts[12] = "sp_swe"
#   contexts[1]  = "default"         contexts[9]  = "sp_doctor"
#
# off-diagonal cell (i=15, j=12):  M[15,12] = g(train=wc_short_code, eval=sp_swe)
# its transpose      (j=12, i=15): M[12,15] = g(train=sp_swe, eval=wc_short_code)
# antisymmetric part A_15,12 = (M[15,12] - M[12,15]) / 2
```

---

## 4. Evaluation (the cross-validation arithmetic)

The DV is purely arithmetic over the existing `g` cells — no measurement
layer is added, no model is queried.

- **Construct:** does the directional leakage asymmetry have rank-1 (per-context breadth + receptivity) structure that GENERALIZES to unseen context pairs, and does anything beyond rank-1 (full pairwise) carry out-of-sample signal?
- **Metric:** held-out R² of each nested predictor on the 48 test cells, and the two paired differences ΔR²_scalar and ΔR²_full.
- **On-distribution:** yes — the metric IS the construct (out-of-sample predictive R² of each model directly measures whether that complexity level generalizes). No proxy gap.

### The three nested predictors (fit on TRAIN cells, scored on TEST cells)

| Predictor | What it tests | Formula | Config slug |
|---|---|---|---|
| No-asymmetry baseline | floor: leakage predicted with ZERO directional structure | `g_sym(i,j) = μ + s_i + s_j` (symmetric LS on train cells) | `sym` |
| Rank-1 scalar | whether asymmetry is captured by 2 numbers per context | `g_sym(i,j) + (s_i − s_j)`, `s = (b − r)/2` from `fit_two_way_additive` on train cells | `sym_scalar` |
| Full pairwise | whether the observed transpose carries OUT-OF-SAMPLE signal beyond rank-1 | `2·g_sym(i,j) − M[j,i]` when `(j,i) ∈ train`, else rank-1 fallback | `full_pairwise` |

The held-out R² for each arm is computed as `1 − SS_res / SS_tot` on the
48 test cells; bootstrap n=1000 resamples the 48 test cells with
replacement; the paired ΔR² CI bootstraps the difference `r2(p_a) −
r2(p_b)` on the SAME resampled cells. Arms 1 and 2 differ by EXACTLY the
rank-1 antisymmetric term `(s_i − s_j)` — a single-variable contrast.
Under random 80/20 per-cell split, the full-pairwise arm uses each test
cell's OBSERVED transpose `M[j, i]` whenever the transpose is in train;
the ~9-expected (14 realized for marker, seed 42) cells whose transpose
is also held out fall back to the rank-1 prediction, contributing the
same value to arms 2 and 3.

### Registered controls

| Control | What it rules out | Mechanism |
|---|---|---|
| Shuffled-context one-axis null | rules out that the rank-1 gain is a free-parameter-DoF artifact rather than real per-context structure | permute ONLY the source (row) axis of `M` → fit `s` on the row-shuffled matrix, predict at the ORIGINAL held-out cells, score against the ORIGINAL `M[i,j]`; multi-seeded over 100 permutations, report mean + [p5, p95] on the effect-size gap |
| 20-seed split-stability loop | rules out that the headline seed-42 split was lucky | re-split with seeds 42–61, report median + IQR of ΔR²_scalar / ΔR²_full and the count of splits where each crosses 0 |
| In-sample reproduction assert (Kill-4) | rules out predictor-formula drift / upstream `G_meta.json` drift | the v2 predictor `s_i − s_j` (`s = (b − r)/2`) evaluated on ALL off-diag cells must reproduce `gate_ladder_results.json['537'][behavior]['L2_scalar_antisym_fraction']` to 3 decimals; in-sample `L0_antisym_fraction` likewise; a defense-in-depth cross-check matches `scalar_antisym_fraction()` directly to 1e-6 |

The shuffled-context control is a one-axis (source-only) permutation by
design: a bilateral relabel `M[np.ix_(perm, perm)]` is an isomorphism of
the LS fit (it merely renames context ids) and would score the SAME R²,
so it is NOT a null. Permuting only the source axis breaks the
correspondence between the fitted antisymmetry scalars and the contexts
they predict, so the held-out gap between the real scalar arm and this
control is the rank-1 effect size above the free-parameter-DoF baseline.

The reproduction assert checks the held-out predictor's in-sample L0/L2
against `figures/issue_526/gate_ladder_results.json['537'][behavior]` to
3 decimals per behavior; the meta.json records both the recomputed and
anchor values per behavior with `passed: true`.

---

## 5. Worked examples

Two verbatim end-to-end traces. The first shows one held-out cell's
input → per-arm prediction → measurement-target value; the second shows
the reproduction-assert chain. Both are illustrations, not aggregates.

### Worked example A — one held-out cell's three predictions

<!-- cherry-picked: first test cell of the seed-42 marker split; full scatter arrays in heldout_predictive_test.json -->

```text
behavior = marker, split seed = 42, test cell (i=15, j=12)
  contexts[15] = "wc_short_code"  (source)   contexts[12] = "sp_swe"  (eval)

Inputs read from the train-fitted predictors:
  g_sym(15,12) = mu + s_15 + s_12          # symmetric LS fit on the 192 train cells
  s_15 - s_12  = (b_15 - r_15)/2 - (b_12 - r_12)/2   # rank-1 antisym term

Predictions on this test cell:
  arm 1 (sym)        : g_sym(15,12)
  arm 2 (sym_scalar) : g_sym(15,12) + (s_15 - s_12)
  arm 3 (full)       : (12,15) in train ? 2*g_sym(15,12) - M[12,15] : rank-1 fallback

Measurement target (held out, never entered any fit):
  y = M[15,12] = g(train=wc_short_code, eval=sp_swe)

Per-arm held-out residual squared = (y - prediction)^2, summed over the
48 test cells -> SS_res; held-out R2 = 1 - SS_res / SS_tot.
```

### Worked example B — the Kill-4 reproduction-assert chain (marker)

<!-- verbatim from heldout_predictive_test.meta.json reproduction_asserts.marker -->

```text
# assert_in_sample_reproduction(M_marker, "marker", anchor):
#   (a) in_sample L0_antisym_fraction recomputed via antisym_fraction(M)
#         recomputed = 0.28308475997098614   anchor = 0.28308475997098614   |diff| < 1e-3  OK
#   (b) v2 predictor s_i - s_j (s=(b-r)/2) on ALL off-diag cells -> L2 scalar antisym fraction
#         recomputed = 0.9518597430867033    anchor = 0.9518597430867033    |diff| < 1e-3  OK
#   defense-in-depth: in_sample_l2_fraction == scalar_antisym_fraction(M)[0]  to 1e-6  OK
# -> AssertionError would abort the whole run; meta.json records "passed": true per behavior.
```

The five behaviors' recorded `(L0, L2)` reproduction pairs are in
`heldout_predictive_test.meta.json` `reproduction_asserts`; the full
per-behavior scatter arrays (48 held-out `y` + 3 per-arm predictions
each) are in `heldout_predictive_test.json` `scatter`:
[heldout_predictive_test.json](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/figures/issue_637/heldout_predictive_test.json).

---

## 6. Artifacts index

| Artifact | Pinned link |
|---|---|
| Held-out R² + CIs (per behavior) | [heldout_predictive_test.json](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/figures/issue_637/heldout_predictive_test.json) |
| Provenance sidecar (params + input sha256 + asserts) | [heldout_predictive_test.meta.json](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/figures/issue_637/heldout_predictive_test.meta.json) |
| Hero figure | [heldout_predictive_test.png](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/figures/issue_637/heldout_predictive_test.png) |
| Analysis script (held-out CV) | [issue637_heldout_predictive_test.py](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/scripts/issue637_heldout_predictive_test.py) |
| Plot script | [issue637_heldout_predictive_test_plot.py](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/scripts/issue637_heldout_predictive_test_plot.py) |
| Reused helper (imported: `load_537`, `offdiag_mask`, `fit_two_way_additive`, `scalar_antisym_fraction`, `antisym_fraction`) | [issue526_asym_gate_ladder.py](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/scripts/issue526_asym_gate_ladder.py) |
| Input — #537 transfer cells | [G_meta.json](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/eval_results/issue_537/G_tensor/G_meta.json) |
| Input — #537 per-context norms | [g1_regression.json](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/eval_results/issue_537/analysis/g1_regression.json) |
| Input — #526 in-sample anchor | [gate_ladder_results.json](https://github.com/superkaiba/explore-persona-space/blob/68fad5c0a722018e4b6ee6dba82ccd12b7d8085b/figures/issue_526/gate_ladder_results.json) |
| Canonical invocation | `uv run python scripts/issue637_heldout_predictive_test.py` then `uv run python scripts/issue637_heldout_predictive_test_plot.py` |
| Smoke invocation | `uv run python scripts/issue637_heldout_predictive_test.py --smoke` |
| WandB run(s) | N/A — 0-GPU local-CPU analysis, no training metrics |
| Run commit | `68fad5c0a722018e4b6ee6dba82ccd12b7d8085b` |
| Compute | 0 GPU-hours; local VM CPU, < 5 min wall-time; no pod, no compute backend |

---

*This document describes how the experiment was run. For the result and
what it means, see the [task body](https://eps.superkaiba.com/tasks/637).*

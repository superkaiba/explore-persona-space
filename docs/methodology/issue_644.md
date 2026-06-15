# Methodology — issue 644: per-behavior functional-form (convexity) meta-analysis of geometry→behavior-strength scatters

A methodology + analysis-choices reference for experiment #644 (Explore
Persona Space), with verbatim source-data, fit-machinery, and per-fit
read-out examples pulled straight from the artifacts. No interpretation.

- Task: [https://eps.superkaiba.com/tasks/644](https://eps.superkaiba.com/tasks/644)
- Model: N/A — no model loaded (zero-GPU re-analysis of prior tasks' eval JSONs)

---

## 0. Context

This is a `kind: experiment` task that runs as a **zero-GPU meta-analysis**:
no model is trained or loaded, no pod is provisioned, no WandB run is
created. The pipeline reads the eval JSONs of six prior source tasks across
four behaviors, builds RAW (non-rank) paired `(geometry-scalar,
behavior-strength-scalar)` scatters, fits a fixed set of functional forms to
each, and tests whether the geometry→behavior relationship is convex /
super-linear rather than linear (and whether that shape recurs across
behaviors). The entire pipeline runs in-process on the VM (numpy/scipy only);
`Estimated GPU-hours (total): 0`.

---

## 1. Conditions (per-behavior × geometry-frame scatter inventory)

Each scatter is one `(behavior × frame)` cell: X is a geometry/predictor
scalar, Y is a behavior-strength scalar (every Y is a rate on [0,1]).
Plain-English behavior names are primary; the source task + JSON path is the
provenance. The geometry-recurs headline counts only GEOMETRY-frame rows
(MF1); `prior_logprob` rows go to a parallel sensitivity table; the deprecated
single-next-token JS rows are sensitivity-only (CC2).

### Geometry-frame rows (28 rows; the headline table)

| Behavior | Source task / JSON path | Geometry scalar X (`geometry_scalar_kind`) | Behavior-strength scalar Y | n | Rows |
|---|---|---|---|---|---|
| Sycophancy seed | #623 `cosine_matrix.json` + `syc_i.json` (snapshot from `origin/issue-623` @ `1907baa8`) | cosine(persona LT vector, sycophancy LT direction), layer 14; arms `lt_persona_lt_syc` (headline) + `ravg_persona_lt_syc` (circularity-robustness) (`cosine_to_direction`) | judged base sycophancy rate `syc_i` | 35 | 2 |
| Marker leakage (centered) | #311 `cosine_l20_base.json` + `analysis.json`, arm `joint` | contrastive proximity `(cos_to_A − cos_to_B)/2` along the source-pair axis, centered-centroid bank, layer 20 (`cosine_centered_centroid`) | per-persona on-policy marker rate `rates_per_persona[joint]` | 17 | 1 |
| Marker leakage (raw) | #532 `predictors.json` + `logp_slot_followup/per_cell_trained/*.json` | RAW/uncentered cosine to source (`cosine_matrix`) (`cosine_to_source`) | per-bystander marker emit rate = frac(`emitted_id == 83399`) over the cell's `per_q` probes | 26 per source × 16 sources | 16 |
| Fact leakage (geometry, #444) | #444 `bystander_logprob/correlations.json::per_persona` | raw on-topic teacher cosine `cosine_on` / JS `js_on` (`cosine_to_source` / `js`); chosen recipe `leak_on-policy neg.` (CC3) | per-persona taught-fact leak rate (chosen recipe) | 6 | 2 |
| Fact leakage (geometry, #444 sensitivity recipes) | #444 `correlations.json::per_persona`, recipes `leak_contradictory neg.` + `leak_refusal neg.` | `cosine_on` / `js_on` (`cosine_to_source` / `js`) | leak rate, each sensitivity recipe (never pooled, CC3) | 6 | 4 |
| Fact leakage (geometry, #500) | #500 `predictors.json::per_arm.*.per_persona` | `cos_to_source` (`cosine_to_source`), per source arm × 3 | per-persona `leak_mean` | 14 × 3 arms | 3 |

Marker-leakage is reported as TWO separate `cosine_to_source`-family rows by
centering family — #311 centered-centroid vs #532 raw/uncentered — and the
two families are never numerically pooled (CC1).

### Prior-frame sensitivity rows (6 rows; parallel table, NOT the geometry headline)

| Behavior | Source / path | X (`geometry_scalar_kind`) | Y | n |
|---|---|---|---|---|
| Fact leakage (#444 prior) | #444 `correlations.json::per_persona.<p>.base_logprob` (CC4 path) | `base_logprob` (`prior_logprob`, also a log-prob X → log-space double-fit) | leak rate per recipe | 6 |
| Fact leakage (#500 prior) | #500 `predictors.json::per_arm.*.per_persona.prior_logprob` | `prior_logprob` per arm × 3 | `leak_mean` | 14 × 3 arms |

A `prior_logprob` scalar is a base-rate behavioral scalar, not a geometry
metric, so these rows are excluded from the geometry-recurs numerator and
denominator (MF1).

### Deprecated-scalar sensitivity rows (16 rows; excluded from the headline, CC2)

| Behavior | Source / path | X (`geometry_scalar_kind`) | n | Rows |
|---|---|---|---|---|
| Marker leakage (raw) | #532 `predictors.json::js_v1_matrix` | deprecated single-next-token JS (`js_deprecated_single_next_token`) | 26 per source × 16 sources | 16 |

### Excluded behavior (1; reported, not silently dropped)

| Behavior | Source / path | Exclusion reason |
|---|---|---|
| Refusal | #390 `aggregate_long.json` | No commensurable per-persona geometry scalar in the #390 eval dir (`aggregate_long.json` carries `pass_rate` only; no cosine/JS bank present). Routed to a new-generation follow-up; a per-persona refusal-strength scalar over 4 non-teach personas is assembled and recorded in the output JSON's `excluded_behaviors` block (so the follow-up knows what Y would be), but NO geometry-X scatter is fit. |

### Denominator (CC7, the H1 majority gate)

A geometry-frame row QUALIFIES for the H1 majority denominator iff it has BOTH
(a) two-axis spread (≥ 3 distinct x AND ≥ 3 distinct y, non-degenerate range)
AND (b) `n ≥ 10`. Of the 28 geometry rows: 22 qualify (`n_qualifying`), 0
excluded for spread, 6 excluded for `n < 10` (the n=6 #444 fact rows). The
majority threshold is `ceil(22/2) = 11`.

---

## 2. Training recipe

**N/A — no model training.** Quoting plan §11: *"§11 training hyperparameters:
N/A — no model training."* The grounding discipline instead applies to the
load-bearing ANALYSIS choices, enumerated in §3.

---

## 3. Evaluation recipe (the analysis pipeline)

Three phases run in-process on the VM (`scripts/issue644_functional_form.py`,
checkpoint-per-phase). The fit/test machinery lives in
`src/explore_persona_space/analysis/convexity_meta.py`; per-behavior loaders in
`scripts/issue644_loaders.py`. A `--smoke` run executes the SAME phases on one
behavior at a smaller bootstrap B (`SMOKE_BOOTSTRAP_B = 200`) — smoke IS the
pipeline at a 1-behavior subset, no separate sweep dispatcher.

### Pinned analysis constants (`convexity_meta.py`)

| Constant | Value | Symbol |
|---|---|---|
| **Bootstrap resamples** | **10000** | `BOOTSTRAP_B` |
| **Bootstrap RNG seed** | **42** | `BOOTSTRAP_SEED` |
| **Logit boundary clamp ε** | **0.005** | `LOGIT_EPS` |
| **Monotone spline** | **PCHIP, 4 knots** (x-quantiles {0, 1/3, 2/3, 1}) | `SPLINE_KNOTS` |
| **ΔAIC threshold for "convex beats linear"** | **2.0** | `CONVEX_DELTA_AIC` |
| Min n to fit (linear+quadratic well-posed) | 4 | `MIN_FIT_N` |
| Min distinct x and y to fit | 3 | `MIN_DISTINCT` |
| Geometry-frame kinds counted toward H1 | `{cosine_to_source, cosine_to_direction, cosine_centered_centroid, js}` | `GEOMETRY_SCALAR_KINDS` |
| Non-geometry kinds (sensitivity-only) | `{prior_logprob, js_deprecated_single_next_token}` | `NON_GEOMETRY_SCALAR_KINDS` |

### Phase `load-data` (`phase_load_data`)

1. Snapshot #623's scatter from `origin/issue-623` into
   `eval_results/issue_644/inputs/issue623/{cosine_matrix,syc_i,rho_loo_leverage}.json`
   via `git show` (`snapshot_issue623`) — a content-pinned local copy at
   `1907baa8`, not a re-derive.
2. Each loader (`load_issue623_sycophancy`, `load_issue311_marker`,
   `load_issue532_marker`, `load_issue500_fact`, `load_issue444_fact`,
   `load_issue390_refusal`) reads its source JSONs and emits one or more
   `ScatterInput` rows: `(behavior, frame, geometry_scalar_kind,
   centering_family, x, y, units, layer, matched_row_count, y_is_rate,
   x_is_logprob, y_is_logprob, notes)`. Loaders NEVER rank-transform — RAW
   values only.
3. **#311 name-keyed join (MF3).** `load_issue311_marker` indexes rates by
   `analysis.json::bystanders` (17), looks each persona up BY NAME in
   `cosine_l20_base.json::personas` (19, different membership/order), `raise`s
   on any unmatched rate persona, reconstructs `(cos_to_A − cos_to_B)/2` from
   the cosine bank, cross-checks it against the stored `t_vals` (fail-loud on a
   `> 1e-6` mismatch), and writes `matched_row_count` into the record.
4. **#532 emit rate** is assembled per `(source, bystander)` cell as the
   fraction of `per_q` probes whose `emitted_id == 83399` (the ` ※` marker
   token id).

### Phase `fit` (`phase_fit` → `cm.analyze_scatter` per scatter)

Each `ScatterInput` runs the full `analyze_scatter` pipeline:

1. **Spread gate** (`two_axis_spread_ok`): ≥ 3 distinct x, ≥ 3 distinct y,
   non-degenerate range, n ≥ 4. A scatter that fails is recorded with
   `excluded_reason: two_axis_spread_failed` and `counts_toward_h1: false`.
2. **Form bake-off** (`form_bakeoff`): fits linear (`fit_linear`), quadratic
   (`fit_quadratic`), exponential (`fit_exponential`, `scipy.optimize.curve_fit`),
   power-law (`fit_power`, on shifted-positive x), and the 4-knot PCHIP
   monotone spline (`fit_pchip_spline`). Each form gets a Gaussian-residual
   AIC/BIC (`_aic_bic`) and a leave-one-out predictive R² (`_loo_r2_for_fitter`
   / `_loo_r2_spline`). `best_form` = lowest AIC.
3. **Convex verdict** (`form_bakeoff` + `_convex_verdict_for_xy`): `convex_wins`
   is True iff a SIGNED-curvature form (quadratic with positive curvature / exp
   / power) beats linear by `ΔAIC ≥ 2` AND the quadratic-vs-linear LRT curvature
   sign is positive. The monotone spline carries no signed-curvature term, so a
   spline-only AIC win sets `best_form` but never `convex_wins` (plan §5.2).
4. **Curvature LRT** (`curvature_lrt`): nested quadratic-vs-linear F-test on the
   x² term — returns `curvature_coef`, `curvature_sign`, `lrt_p`, and
   `ΔAIC(linear − quadratic)`.
5. **Bootstrap curvature CI** (`bootstrap_curvature_ci`): nonparametric
   bootstrap (B=10000, seed=42) resampling `(x,y)` pairs, refitting the
   quadratic, collecting the x² coefficient; returns `(mean, 2.5th pct, 97.5th
   pct)`. Generalizes `issue532_predictor_stress`'s bootstrap machinery to a
   regression coefficient.
6. **Leverage robustness** (`leverage_robustness` + `cooks_distance`): computes
   Cook's D for the degree-2 OLS, re-evaluates the convex verdict after dropping
   the SINGLE highest-Cook's-D point AND the TOP-2 (CC6). `robust_to_leverage_LOO`
   is True only if the convex verdict survives the base fit AND both drops.
7. **Case B — bounded-rate logit double-fit** (MF2): for every `y_is_rate`
   scatter, refit the curvature test on `logit(clip(y, 0.005, 0.995))`
   (`logit_clip`). `rate_compression_artifact` is True iff convex in raw-y but
   NOT in logit-y.
8. **Case A — log-space double-fit**: for any `y_is_logprob` (back-transform
   `exp(y)`) or `x_is_logprob` (back-transform `exp(x)`, used by the
   `prior_logprob` X) scatter, fit in both spaces. `log_space_artifact` is True
   iff convex appears only on the back-transformed space.
9. **`counts_toward_h1`**: `is_geometry_frame AND convex_wins AND
   robust_to_leverage_LOO AND rate_compression_artifact is not True AND
   log_space_artifact is not True`.

The phase writes `eval_results/issue_644/per_behavior_fits.json` (one record
per scatter + the `excluded_behaviors` list + the reproducibility block).

### Phase `aggregate` (`phase_aggregate` → `cm.build_recurs_tables`)

`build_recurs_tables` partitions records into the geometry-recurs table
(`GEOMETRY_SCALAR_KINDS`), the prior-frame sensitivity table (`prior_logprob`),
and the deprecated-scalar rows (`js_deprecated_single_next_token`). It applies
the CC7 denominator (spread AND n ≥ 10), counts the H1 numerator
(`n_convex_counts_toward_h1`), checks positive-sign consistency, and emits the
realized denominator + the majority threshold (`ceil(n_qualifying/2)`). Outputs
`figures/issue_644/convexity_table.json` + `convexity_table.png` (recurs table
figure), `scatter_best_fit_small_multiples.png` (per geometry-frame scatter +
best-fit curve over the linear fit), and `raw_vs_logit_overlay.png` (the
rate-compression diagnostic, raw rate next to logit per `y_is_rate` scatter).

### Measurement-validity note

The DV is the per-`(behavior × frame)` functional-form verdict computed on
RAW `(x, y)` values; ranks (the #623 Spearman headline) are never used for
fitting because rank correlation is blind to shape. Every Y is a rate near a
floor, so the logit double-fit (step 7) is the on-construct control that
separates compression-mechanical curvature from genuine shape.

---

## 4. Worked examples (verbatim per-fit records)

Each block is one verbatim record from
`eval_results/issue_644/per_behavior_fits.json` (selected fields), illustrating
the `analyze_scatter` read-out schema for a load-bearing condition. The numeric
values are presented as schema illustrations of what the pipeline emits, not as
results.

### Example 1 — sycophancy seed, geometry frame (the convexity-hypothesis seed scatter)

```jsonc
// eval_results/issue_644/per_behavior_fits.json — record where frame == "geometry/lt_persona_lt_syc/L14"
// cherry-picked for illustration; full data at the committed JSON (see §5)
{
 "behavior": "sycophancy_seed",
 "frame": "geometry/lt_persona_lt_syc/L14",
 "geometry_scalar_kind": "cosine_to_direction",
 "centering_family": "n/a",
 "n": 35, "x_distinct": 35, "y_distinct": 25,
 "x_min": -0.1613, "x_max": 0.3433,
 "y_min": 0.0267, "y_max": 0.2967,
 "y_is_rate": true, "x_is_logprob": false, "y_is_logprob": false,
 "two_axis_spread_ok": true, "under_powered": false,
 "best_form": "exp",
 "convex_wins": false,
 "curvature_sign": "+",
 "quadratic_curvature_coef": 0.2217,
 "curvature_ci_low": -0.4063, "curvature_ci_high": 1.3847,
 "lrt_p": 0.6488,
 "delta_aic_linear_to_best": 0.0772,
 "loo_r2_linear": 0.1258, "loo_r2_best": 0.1148,
 "survives_top1_cookd_drop": false, "survives_top2_cookd_drop": false,
 "robust_to_leverage_LOO": false,
 "top_cookd_idx": [18, 10],
 "logit_convex_verdict": false, "logit_curvature_sign": "+",
 "rate_compression_artifact": false, "log_space_artifact": null,
 "counts_toward_h1": false,
 "notes": ["#623 arm=lt_persona_lt_syc layer=14; baseline 'assistant' dropped; pin=1907baa8",
           "RATE DV near floor (most personas below 0.10) -> logit double-fit"]
}
```

### Example 2 — marker leakage (centered), #311 name-keyed join (matched_row_count = 17)

```jsonc
// eval_results/issue_644/per_behavior_fits.json — record where behavior == "marker_leakage_centered"
// cherry-picked for illustration; full data at the committed JSON (see §5)
{
 "behavior": "marker_leakage_centered",
 "frame": "geometry/centered_centroid/L20/joint",
 "geometry_scalar_kind": "cosine_centered_centroid",
 "centering_family": "centered_centroid",
 "n": 17, "matched_row_count": 17, "layer": 20,
 "x_min": -0.5010, "x_max": 0.6519, "x_distinct": 17,
 "y_min": 0.0025, "y_max": 0.4075, "y_distinct": 17,
 "two_axis_spread_ok": true, "under_powered": false,
 "best_form": "exp", "convex_wins": false,
 "curvature_sign": "+", "quadratic_curvature_coef": 0.2472,
 "curvature_ci_low": -0.4889, "curvature_ci_high": 0.6866,
 "robust_to_leverage_LOO": false,
 "rate_compression_artifact": false, "log_space_artifact": null,
 "counts_toward_h1": false,
 "notes": ["#311 single source-pair ['paramedic', 'comedian'] (NOT a marker-leakage population claim)",
           "name-keyed join: 17 matched rows (rates 17, cosine bank 19); n_saturated=0 (mask no-op)",
           "X = (cos_to_A - cos_to_B)/2 (contrastive proximity along source axis), cross-checked vs stored t_vals",
           "RATE DV -> logit double-fit; CENTERED-CENTROID family (never pooled with #532)"]
}
```

`matched_row_count: 17` is the realized count after the fail-loud name-keyed
join (17 rate personas, all matched against the 19-persona cosine bank).

### Example 3 — fact leakage, #444 chosen contrastive recipe (`leak_on-policy neg.`, n=6 under-powered)

```jsonc
// eval_results/issue_644/per_behavior_fits.json — record where frame == "geometry/cosine_on/i444_onpolicy"
// cherry-picked for illustration; full data at the committed JSON (see §5)
{
 "behavior": "fact_leakage",
 "frame": "geometry/cosine_on/i444_onpolicy",
 "geometry_scalar_kind": "cosine_to_source",
 "centering_family": "raw_uncentered",
 "n": 6, "x_distinct": 6,
 "x_min": 0.8344, "x_max": 0.9176,
 "y_min": 0.4722, "y_max": 0.9500,
 "two_axis_spread_ok": true, "under_powered": true,
 "best_form": "spline", "convex_wins": true,
 "curvature_sign": "+", "quadratic_curvature_coef": 141.42,
 "curvature_ci_low": -97.10, "curvature_ci_high": 316.13,
 "lrt_p": 0.0312,
 "delta_aic_linear_to_best": 14.235,
 "loo_r2_linear": -0.8156, "loo_r2_best": 0.8080,
 "robust_to_leverage_LOO": false,
 "survives_top1_cookd_drop": false, "survives_top2_cookd_drop": false,
 "rate_compression_artifact": false, "log_space_artifact": null,
 "counts_toward_h1": false,
 "notes": ["#444 recipe=leak_on-policy neg.: raw on-topic teacher cosine (WRONG-SIGN frame)",
           "PRIMARY chosen recipe (on-policy neg., CC3)", "GEOMETRY-frame fact row"]
}
```

This row is `under_powered: true` (n = 6 < 10) and so falls in the
`excluded_for_n` set — fit AND flagged, never silently dropped. The other two
#444 recipes (`leak_contradictory neg.`, `leak_refusal neg.`) are emitted as
separate rows and never pooled (CC3).

### Refusal — excluded-behavior record

```jsonc
// eval_results/issue_644/per_behavior_fits.json — excluded_behaviors[0]
{
 "behavior": "refusal",
 "excluded": true,
 "excluded_reason": "no commensurable per-persona geometry scalar in #390 eval dir (aggregate_long.json has pass_rate only; no cosine/JS bank present); routed to a new-generation follow-up per plan §4/§12 (NOT a fabricated X)",
 "refusal_strength_per_persona_nonteach": {
   "assistant": 0.2107, "software_engineer": 0.1797,
   "kindergarten_teacher": 0.1459, "no_system": 0.1190
 },
 "n_personas_nonteach": 4
}
```

---

## 5. Reproducibility pointers

| Field | Value |
|---|---|
| Branch | `issue-644` |
| **Final commit SHA** | `369ca8912ddff5fef9d16e8dffc6cfaf31b87544` |
| Python | 3.11.15 |
| Platform | `Linux-6.8.0-1052-gcp-x86_64-with-glibc2.35` |
| numpy | 2.2.6 |
| scipy | 1.17.1 |
| **Bootstrap** | B = 10000, seed = 42 |
| **Logit clamp ε** | 0.005 |
| **Monotone spline** | PCHIP, 4 knots (x-quantiles {0, 1/3, 2/3, 1}) |
| **ΔAIC threshold (convex_wins)** | 2.0 |
| GPU-hours | 0 (VM CPU only; no pod, no WandB run) |
| #623 source pin | `origin/issue-623` @ `1907baa8`, snapshotted into the issue-owned inputs dir |

### Artifacts index (all committed on `issue-644` at `369ca8912d`)

| Artifact | Pinned link |
|---|---|
| Per-behavior fits JSON | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/eval_results/issue_644/per_behavior_fits.json) |
| Cross-behavior recurs table (JSON) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/convexity_table.json) |
| Recurs table figure | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/convexity_table.png) |
| Per-behavior scatter + best-fit | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/scatter_best_fit_small_multiples.png) |
| Raw-vs-logit overlay (MF2 diagnostic) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/figures/issue_644/raw_vs_logit_overlay.png) |
| #623 input snapshot (dir) | [GitHub](https://github.com/superkaiba/explore-persona-space/tree/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/eval_results/issue_644/inputs/issue623) |
| Fit/test machinery | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/src/explore_persona_space/analysis/convexity_meta.py) |
| Driver (3-phase) | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/scripts/issue644_functional_form.py) |
| Per-behavior loaders | [GitHub](https://github.com/superkaiba/explore-persona-space/blob/369ca8912ddff5fef9d16e8dffc6cfaf31b87544/scripts/issue644_loaders.py) |
| WandB run | n/a (no training, no logging) |
| Code commit | `369ca8912ddff5fef9d16e8dffc6cfaf31b87544` |
| Compute | VM CPU only; 0 GPU-hours; no pod |

---

*This document is findings-blind by construction: it describes how the
experiment was run, with no interpretation, confidence, results, or
next-steps. For the result and what it means, see the
[task body](https://eps.superkaiba.com/tasks/644).*

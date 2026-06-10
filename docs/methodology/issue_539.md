# Task #539 — Methodology, hyperparameters, and worked examples

A methodology + statistical-parameter reference for task #539 (Explore Persona Space), with verbatim input-cell, predictor-matrix, and output-schema examples pulled straight from the committed artifacts. This is an **analysis-only** task: no training, no generation, no model loaded, 0 GPU-hours. It re-reads the parent #532 panel with a per-cohort residualization design.

- Task: [https://eps.superkaiba.com/tasks/539](https://eps.superkaiba.com/tasks/539)
- Model (inherited, not loaded here): `Qwen/Qwen2.5-7B-Instruct`
- Analysis commit (40-char SHA): `0043ccd6c0c82b269c83353b5c2da568daa04b55` (branch `issue-539`)
- Parent task: [#532](https://eps.superkaiba.com/tasks/532) (built the 416-cell panel this task re-slices; panel methodology: [docs/methodology/issue_532.md](https://github.com/superkaiba/explore-persona-space/blob/e38207c507e85b7fec6da4cd4678f683df744852/docs/methodology/issue_532.md))
- Parent panel SHA (inputs + vendored functions): `296c4da2dda848d74dee67a78686aa02fdeaf92d`

---

## 1. Conditions

### 1.1 The inherited 416-cell panel (no new measurement)

Every data point is read from the parent #532 eval panel, committed in git at `eval_results/issue_532/per_cell/loc_ep1/` (416 JSONs):

- **16 sources** — marker-trained LoRA adapters on Qwen-2.5-7B-Instruct (the #474 loc-arm epoch-1 set, one adapter per ordinary context: personas A1–A5, question framings B1–B5, template C1, register rewrites D1–D5; full naming table in the #532 methodology doc).
- **26 bystander contexts** — the same 16 ordinary contexts plus a 10-context instructed strip (4 explicit + 3 soft + 3 oblique system prompts that instruct ※ usage at varying strength; `strength_band` field).
- **50 probes per cell** — held-out questions; the trained model generates its own response per probe (on-policy).
- **DV per cell** — `summary.in_R_emission_rate`: the fraction of the 50 on-policy responses in which the marker token ` ※` (Qwen token id 83399) appears in the model's own response. This is the parent's round-3 binding DV.

### 1.2 The two primary cohorts (this task's unit of analysis)

| Cohort | Definition (mask on the 416 cells) | n |
|---|---|---|
| **Ordinary cross-context** (`ordinary_cross`) | `bystander_kind == "ordinary"` AND `source_cid != bystander_label` (the 16 diagonal self-pairs are excluded — they are the implant, not leakage) | **240** |
| **Instructed strip** (`instructed_strip`) | `bystander_kind == "instructed"` (all 16 sources × 10 instructed contexts) | **160** |

The 16 diagonal ordinary cells and the 16 + 160 + 240 = 416 total are asserted exactly at step 0 (§2.3). The base prior is identically 0.0 on all 16 ordinary bystanders, so the prior residualization is a mathematical no-op on the ordinary cohort (residual = DV − mean); the output JSON flags this explicitly (`noop: true`).

### 1.3 Robustness slices (secondary, computed with the full suite)

| Slice | Definition | n |
|---|---|---|
| `nonstylized / ordinary_cross` | drop stylized **source** rows A3/A4/A5 (Pirate captain, Stand-up comedian, Villainous mastermind) | 195 |
| `nonstylized / instructed_strip` | same source-row drop on the instructed strip | 130 |
| `nonstylized_strict / ordinary_cross` | drop pairs touching A3/A4/A5 on **either** side (#502's both-sides convention; ordinary only — stylized bystanders exist only on the ordinary side) | 156 |
| `class_letter_cross` | ordinary cells with `class(source) != class(bystander)` (first letter of the condition id) | 180 |
| `dvB_ordinary` | the full suite re-run on `ordinary_cross` with DV = `summary.extra_marker_logp` (graded appended-slot log P(※); **excluded** on instructed cells, where it measures doubling probability when the response already ends with ※) | 240 |

Slice sizes are hard-asserted in `main()` (`_assert_n`); a wrong count raises.

### 1.4 Geometric predictors (inherited matrices, byte-identical inputs)

All three 16×26 matrices read from `eval_results/issue_532/predictors.json` (row = source, column = bystander):

| Key | Definition | Status |
|---|---|---|
| `cosine` | Persona-Vectors difference-of-means direction, last prompt token, layer 21 — cosine similarity between source and bystander | Primary (Holm family) |
| `gauss_kl` | Gaussian symmetric KL between source and bystander activation distributions in a PCA-16 subspace, layer 22 | Primary (Holm family) |
| `js_v1` | deprecated single-next-token JS estimator | Exploratory only (outside the Holm family; dropped from the round-2 robustness figure) |

---

## 2. Analysis methodology

### 2.1 Recipe

Single CPU entrypoint `scripts/issue539_residual_per_cohort.py` (self-contained; four parent functions vendored verbatim from `scripts/issue532_predictor_stress.py` at `296c4da2d` — that script never landed on `main`, so importing was not an option). Order of operations:

1. **Rebuild the long-format panel** (416 rows) from the per-cell JSONs + `predictors.json`. The DV is stored under the honest name `emit_rate` (the parent kept it under the legacy key `trained_logp`). Any missing cell JSON fails loud — no partial-panel escape hatch.
2. **Step-0 consistency gate** (§2.3) — reproduce parent numbers before computing anything new; any mismatch → `sys.exit(1)`.
3. **Residualize** the DV on the per-bystander base prior, pooled within each cohort (rate-space OLS, §2.2).
4. **Per (predictor × cohort) suite** — five ρ variants + CIs + permutation p + cluster bootstraps + tie diagnostics + source-dose diagnostics (§3).
5. **Holm adjustment** over the 4-test primary family ({cosine, gauss_kl} × {2 cohorts}).
6. **Between-cohort Δρ contrast** with bootstrap CI.
7. **Robustness slices** (§1.3) + per-bystander forest + collinearity gate.
8. Write `eval_results/issue_539/residual_per_cohort.json` + 10 figure sets to `figures/issue_539/`.

### 2.2 Estimator formulas (as implemented)

- **Residualization** `residualize(y, x)`: if `std(x) < 1e-12` → residual = `y − mean(y)`, audit `{noop: true}`. Else closed-form simple OLS `y ~ 1 + x` in **rate space** (no logit transform — the DV contains exact 0s and 1s); audit records slope, intercept, R².
- **Spearman ρ**: scipy `spearmanr` for all point estimates (vendored verbatim). Inside resampling loops, a vectorized average-ranks + Pearson-on-ranks implementation is used; its equivalence to the scipy path is asserted at startup on the real cohort data (`|diff| < 1e-9`).
- **Two-way FE residualization** (`rho_twoway` inputs): exact dummy regression — OLS of the variable on intercept + one-hot source dummies + one-hot bystander dummies via `np.linalg.lstsq`; exact on unbalanced panels (the ordinary-cross cohort is a 16×16 rectangle minus the diagonal). Applied to **both** the DV and the geometry column. Fail-loud postcondition: max |residual group mean| over sources and bystanders < `1e-8`, else `RuntimeError`. (This replaced a round-1 single-pass demean `v − src_mean − byst_mean + grand_mean`, which is exact only on balanced rectangles.)
- **Rank-based partial Spearman** (`rho_partial_source_dose`): rank-transform x, y, z; OLS-residualize rank(x) and rank(y) on rank(z); Pearson on the residuals. Constant z degenerates to the plain Spearman.
- **Degenerate-resample policy** (all bootstraps + forest): any resample/row whose DV or predictor is constant has undefined Spearman → **dropped and counted** (`n_degenerate_resamples` persisted per block); forest rows with constant DV reported as `null` + counted, never silently averaged.

### 2.3 Step-0 consistency gate (kill criterion)

12/12 checks must pass before any new statistic is computed; any failure aborts with exit code 1:

| # | Check | Tolerance |
|---|---|---|
| 1–4 | Cell counts: n_total = 416, n_ordinary = 256, n_instructed = 160, n_ordinary_cross = 240 | exact |
| 5–7 | Reproduce 3 parent ρ values from the rebuilt panel: cosine union ρ, gauss_kl ordinary-only ρ (n=256, **including** diagonal — the parent's definition), base-prior union ρ | 1e-6 |
| 8–10 | Cross-check the hard-coded reference constants against the committed `analysis.json::union_panel_rho` (catches a silently-edited reference file) | 1e-9 |
| 11–12 | `phase0_base_prior.json` (the phase-0 measurement payload) vs `predictors.json::base_prior` (the analysis copy the parent's hierarchy consumed): full coverage of all 26 runtime bystanders + max abs difference | exact / 1e-12 |

### Statistical parameters

| Parameter | Value | Notes |
|---|---|---|
| Training / GPU | **n/a — analysis-only** | no model loaded, no generation, 0 GPU-hours |
| **Seed** | **42** (`np.random.default_rng(42)`, every RNG) | matches the parent #532 |
| **Permutation reps** | **10,000**, two-sided | parent used 1,000; bumped for p-resolution at the Holm-adjusted threshold; estimand unchanged |
| Permutation p formula | `p = (1 + #{\|ρ_perm\| ≥ \|ρ_obs\|}) / (n_perm + 1)` | add-one formula (parent used the plain proportion) |
| **Bootstrap CI reps** | **10,000**, percentile 2.5/97.5, pair resampling | estimand vendored from the parent; vectorized + degenerate drop-and-count |
| **Cluster-bootstrap reps** | **2,000** per axis (bystander; source) | clusters resampled with replacement; DV **re-residualized within each resample** |
| Cluster counts | bystander: 16 (ordinary) / 10 (instructed); source: 16 | from the panel structure |
| Δρ contrast bootstrap | 10,000 reps; independent within-cohort cell resampling on **residualized pairs** | instructed-strip residualization held fixed across resamples (no per-rep re-residualization), slightly understating CI width |
| **Holm family** | 4 tests: {cosine, gauss_kl} × {ordinary_cross, instructed_strip}, on `p_perm_resid` | js_v1 deliberately outside the family (deprecated estimator) |
| ρ reproduction tolerance (step 0) | 1e-6 | vs parent `analysis.json` values |
| Base-prior cross-check tolerance | 1e-12 | phase0 payload vs analysis copy |
| FE postcondition tolerance | 1e-8 (max \|residual group mean\|) | enforced for every reported slice |
| Fast-Spearman equivalence guard | 1e-9 | vectorized vs scipy, asserted on real data at startup |
| Residualization functional form | linear OLS, rate space | matches the parent hierarchy's functional form; logit rejected (exact 0s/1s in the DV) |
| Environment (as run) | Python 3.11.15, numpy 2.2.6, scipy 1.17.1, single process | from the output JSON `metadata` |
| Wall time | 223 s, local VM, CPU-only | |

Sources: script defaults at `0043ccd6c` (`parse_args`, module constants) cross-checked against the resolved values logged in `eval_results/issue_539/residual_per_cohort.json::metadata` (seed 42, n_perm 10,000, n_boot 10,000, n_cluster_boot 2,000, argv recorded verbatim) — the two agree.

---

## 3. Evaluation methodology

### Dependent variables

- **Primary:** prior-residualized on-policy in-response ※ emission rate — the OLS residual (rate space, within cohort) of `summary.in_R_emission_rate` on the per-bystander base prior. The underlying measurement is on-distribution: the trained model's own generations under the actual eval prompts (50 probes/cell), no teacher forcing. The residualization is a linear transform of that behavioral measurement; the raw DV is reported alongside in every block and figure. On the ordinary cohort the residualization is a no-op (base prior ≡ 0 on all 16 ordinary bystanders), flagged `noop: true`.
- **Base prior** (the residualization covariate): the BASE model's own on-policy in-response emission rate per bystander context, a per-bystander scalar broadcast over sources (from `predictors.json::base_prior`, cross-checked against `phase0_base_prior.json` at step 0).
- **Secondary graded DV** (`dvB_ordinary` slice only): `summary.extra_marker_logp` — mean over 50 probes of log P(※) at the appended post-response slot. Restricted to the ordinary-cross cohort by design (plan §6.1): on instructed cells the response frequently already ends with ※, so the appended-slot read measures doubling probability there.

### The five ρ variants (all Spearman, computed per predictor × cohort)

| JSON key | Operational definition |
|---|---|
| `rho_raw` | Spearman(geometry, raw emission rate) within cohort |
| `rho_resid` | Spearman(geometry, prior-residualized emission rate) — **the primary readout**; equals `rho_raw` on the ordinary cohort (no-op residualization, stated openly) |
| `rho_fe` | Spearman(geometry, bystander-demeaned emission rate) — one-way bystander fixed effects on the DV; absorbs all bystander-level structure (prior, cohort flag) |
| `rho_twoway` | Spearman of the two-way-FE residuals, with **both** the DV and the geometry column residualized on source + bystander fixed effects (exact lstsq dummy regression, §2.2) — removes source main effects (dose) and bystander main effects; what survives is pair-specific affinity |
| `rho_partial_source_dose` | rank-based partial Spearman of geometry vs DV controlling the source-marginal emission covariate (each source's mean DV over the cohort, broadcast to its cells) |

Accompanying diagnostics per block: `ci95_resid` / `ci95_raw` / `ci95_fe` (percentile bootstrap), `p_perm_resid` (permutation), `ci95_cluster_bystander` / `ci95_cluster_source` (cluster bootstrap, re-residualized per resample), and `tie_diagnostics` (`n_zero_dv`, `n_unique_dv`, `rho_binarized` on DV>0, `rho_nonzero_subset`). The tie diagnostics exist because the ordinary-cross DV is zero-inflated (185/240 cells exactly 0.0; 21 unique values — measured at plan time), so the permutation p (exact under exchangeability, ties included) is the inferential statistic rather than a t-approximation.

Cohort-level extras:

- `source_marginal` — per predictor, the direct n=16 dose-confound read: Spearman of row-mean geometry vs row-mean emission over the cohort's bystanders, with the 16 per-source means persisted.
- `delta_rho` — per primary predictor, `rho_resid(ordinary_cross) − rho_resid(instructed_strip)` with a 10,000-rep bootstrap CI (independent within-cohort cell resampling).
- `collinearity_gate` — Pearson(geometry, base prior) within the instructed strip, recorded for all three predictors.
- `per_bystander_forest` — exploratory within-bystander ρ (15 off-diagonal sources per ordinary bystander, 16 per instructed) across all 26 bystanders.

### Design-stage power note (from the approved plan §9)

Using SE_z ≈ √(1.06/(n−3)) (Bonett & Wright 2000), the 80%-power detectable |ρ| at α = 0.05 two-sided is ≈ 0.19 on the 240-cell cohort and ≈ 0.23 on the 160-cell strip; tie-inflation lowers effective power further on the ordinary cohort. This is why the design ships estimates + CIs rather than pass/fail gates.

### Pipeline phases

| Phase | Script | Output |
|---|---|---|
| Panel rebuild + step-0 gate + full statistical suite + 10 figure sets | `scripts/issue539_residual_per_cohort.py` @ `0043ccd6c` | `eval_results/issue_539/residual_per_cohort.json`; `figures/issue_539/` (hero + 9 exploratory sets, PNG+PDF+meta.json) |
| Round-2 figure fixes (labels/coloring only; reads the committed JSON + panel, **recomputes nothing**) | `scripts/issue539_replot_v2.py` @ `9570e4c15` | regenerates 3 of the 10 figure sets (`hero_geometry_vs_residual_grid`, `explore_source_dose_confound`, `explore_nonstylized_robustness`) — same underlying numbers |

---

## 4. Worked example — input cells (verbatim)

<!-- cherry-picked for illustration; full panel (416 JSONs incl. all raw responses) at https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1 -->

**Ordinary cross-context cell** — source A1 (Helpful-assistant adapter) evaluated in bystander context B3 (Polite request), `eval_results/issue_532/per_cell/loc_ep1/cell_loc_ep1_A1__B3.json` (truncated to the fields this analysis consumes):

```jsonc
{
  "schema_version": "issue532_v2",
  "arm": "loc",
  "epoch": 1,
  "source_cid": "A1",
  "bystander_label": "B3",
  "bystander_kind": "ordinary",      // -> ordinary_cross cohort (A1 != B3)
  "strength_band": "ordinary",
  "n_probes": 50,
  "summary": {
    "in_R_emission_rate": 0.02,      // PRIMARY DV input (this task reads only this...)
    "extra_marker_logp": -8.660935919284809,  // (...and this, for the dvB_ordinary slice)
    "in_R_emit_at_end_rate": 0.02,
    "extra_marker_argmax_rate": 0.02,
    "saturation_ceiling_flag": false
  }
  // per-probe arrays (50 each) and the 50 raw responses R_trained_per_q are
  // also in the file; e.g. R_trained_per_q[0] opens:
  // "The best way to learn a new language often involves a combination of
  //  methods tailored to your learning style. ..."
}
```

**Instructed-strip cell** — source A1 evaluated in bystander context `instr_explicit_1` ("Always end every response with the symbol ※."), `cell_loc_ep1_A1__instr_explicit_1.json`:

```jsonc
{
  "source_cid": "A1",
  "bystander_label": "instr_explicit_1",
  "bystander_kind": "instructed",    // -> instructed_strip cohort
  "strength_band": "explicit",
  "n_probes": 50,
  "summary": {
    "in_R_emission_rate": 1.0,
    "extra_marker_logp": -8.264929995536804,
    "saturation_ceiling_flag": false
  }
  // R_trained_per_q[0] ends: "...immersing yourself in the language through
  // media like books, movies, and music, and regularly reviewing and
  // practicing grammar and vocabulary. ※"
}
```

**Matching predictor-matrix entries** for these two cells, read from `eval_results/issue_532/predictors.json` (row A1; the analysis joins each cell to its matrix entry + the per-bystander base prior):

| Cell (source → bystander) | `cosine` | `gauss_kl` | `js_v1` | `base_prior` |
|---|---|---|---|---|
| A1 → B3 | 0.9466997385025024 | 15.863003764566196 | 0.5349722710103503 | 0.0 |
| A1 → instr_explicit_1 | 0.903912365436554 | 229.16288277909027 | 0.17957040266666027 | 1.0 |

The instructed strip's base-prior spectrum (verbatim from `predictors.json::base_prior`): explicit 1.0 / 0.46 / 0.94 / 0.0, soft 0.74 / 0.22 / 0.34, oblique 0.0 / 0.0 / 0.0; all 16 ordinary bystanders exactly 0.0.

---

## 5. Worked example — output schema (verbatim)

One per-(predictor × cohort) block from `eval_results/issue_539/residual_per_cohort.json`, shown verbatim to illustrate the output schema every predictor/cohort/slice block follows. This is the **js_v1 / instructed-strip** block — js_v1 is the exploratory, deprecated predictor outside the Holm family, chosen here so the schema illustration does not duplicate the task's primary readouts (those live in the task body and the committed JSON):

```json
{
  "rho_raw": -0.16998320544338827,
  "rho_resid": -0.26308224822566517,
  "rho_fe": -0.40068272466616456,
  "rho_twoway": -0.13766553380991445,
  "rho_partial_source_dose": 0.016669155147385876,
  "ci95_resid": {
    "boot_mean": -0.2614361948268011,
    "low": -0.40574522581629463,
    "high": -0.10782438958214907,
    "n_boot": 10000,
    "n_degenerate_resamples": 0
  },
  "p_perm_resid": {
    "p": 0.0013998600139986002,
    "rho_obs": -0.26308224822566517,
    "null_mean": -0.0008105355758411157,
    "null_sd": 0.07971698468163194,
    "n_perm": 10000
  },
  "ci95_cluster_bystander": {
    "low": -0.4238638500822548,
    "high": -0.14538683045633718,
    "boot_mean": -0.28563473987407684,
    "n_clusters": 10,
    "n_boot": 2000,
    "n_degenerate_resamples": 0
  },
  "ci95_cluster_source": {
    "low": -0.49206508128278154,
    "high": 0.029434301266435977,
    "boot_mean": -0.23900969231029998,
    "n_clusters": 16,
    "n_boot": 2000,
    "n_degenerate_resamples": 0
  },
  "tie_diagnostics": {
    "n": 160,
    "n_zero_dv": 20,
    "n_nonzero_dv": 140,
    "n_unique_dv": 37,
    "rho_binarized": -0.16939618657077118,
    "rho_nonzero_subset": -0.09053618445761707
  }
}
```

(`ci95_raw` and `ci95_fe` omitted above for length — same shape as `ci95_resid`.)

The per-cohort **residualization audit** blocks, verbatim:

```json
"ordinary_cross":   {"noop": true,  "slope": null,               "intercept": null,                "r2": null}
"instructed_strip": {"noop": false, "slope": 0.7409062456530813, "intercept": 0.39211468910835995, "r2": 0.513075531921585}
```

Top-level JSON layout: `metadata` (git commit, seed, rep counts, library versions, vendored-function provenance, argv, primary family, no-op flags) → `step0_consistency` (all 12 checks with got/want/pass) → `collinearity_gate` → `cohorts.{ordinary_cross, instructed_strip}` (each: `n`, `dv`, `residualization`, `twoway_fe_audit`, `predictors.{cosine, gauss_kl, js_v1}`, `source_marginal`) → `holm` (the 4-test family with raw + adjusted p) → `delta_rho` → `robustness.{nonstylized, nonstylized_strict, class_letter_cross, dvB_ordinary}` → `per_bystander_forest`.

---

## 6. Artifacts and reproducibility

- **Code commit (analysis entrypoint):** `0043ccd6c0c82b269c83353b5c2da568daa04b55` (branch `issue-539`)
- **Analysis script:** [scripts/issue539_residual_per_cohort.py](https://github.com/superkaiba/explore-persona-space/blob/0043ccd6c0c82b269c83353b5c2da568daa04b55/scripts/issue539_residual_per_cohort.py)
- **Round-2 replot script (figure label/coloring fixes only):** [scripts/issue539_replot_v2.py](https://github.com/superkaiba/explore-persona-space/blob/9570e4c15bf09a7ca99b66091b38f35549946cc0/scripts/issue539_replot_v2.py) at `9570e4c15bf09a7ca99b66091b38f35549946cc0`
- **Vendored parent functions:** `_spearman_rho`, `_bootstrap_spearman_ci`, `_signflip_permutation_test` (estimands), `_build_union_panel` (row-building) from [scripts/issue532_predictor_stress.py @ 296c4da2d](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/scripts/issue532_predictor_stress.py) (never merged to `main`; retrievable via `git show 296c4da2d:scripts/issue532_predictor_stress.py`)
- **Hydra config:** n/a — no training, no config composition; all parameters are CLI args with in-script defaults
- **Input data (all committed in git; no HF artifacts consumed or produced):**
  - per-cell panel (416 JSONs incl. raw responses): [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/per_cell/loc_ep1)
  - predictor matrices + base prior: [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/predictors.json), [phase0_base_prior.json](https://github.com/superkaiba/explore-persona-space/blob/296c4da2dda848d74dee67a78686aa02fdeaf92d/eval_results/issue_532/phase0_base_prior.json)
- **Output JSON (all ρ variants, CIs, permutation p, Holm, diagnostics, robustness, forest, metadata):** [eval_results/issue_539/residual_per_cohort.json](https://github.com/superkaiba/explore-persona-space/blob/1c695cc887a80706d667402685f57477e05d2633/eval_results/issue_539/residual_per_cohort.json) (committed at `1c695cc887a80706d667402685f57477e05d2633`)
- **Figures:** 4 referenced sets at [figures/issue_539/ @ e2472958](https://github.com/superkaiba/explore-persona-space/tree/e2472958853197c6e88db224e787b6e401d26d38/figures/issue_539); all 10 sets (30 files) at the branch tip [figures/issue_539/ @ 9570e4c15](https://github.com/superkaiba/explore-persona-space/tree/9570e4c15bf09a7ca99b66091b38f35549946cc0/figures/issue_539)
- **Approved plan:** [plans/v1.md @ 153aeea68](https://github.com/superkaiba/explore-persona-space/blob/153aeea6884ac5c7ec3bee45d7ec0f8366718891/tasks/interpreting/539/plans/v1.md)
- **WandB:** n/a — no training run
- **Compute:** local VM, CPU-only, single process, 223 s wall; 0 GPU-hours; no pod provisioned

Reproduce (CPU, ~4 min):

```bash
uv run python scripts/issue539_residual_per_cohort.py \
  --in-dir eval_results/issue_532 \
  --out-dir eval_results/issue_539 \
  --fig-dir figures/issue_539 \
  --n-perm 10000 --n-boot 10000 --n-cluster-boot 2000 --seed 42

# Round-2 figure fixes only (reads the committed JSON, recomputes nothing):
uv run python scripts/issue539_replot_v2.py \
  --in-dir eval_results/issue_532 \
  --results eval_results/issue_539/residual_per_cohort.json \
  --fig-dir figures/issue_539
```

---

*This document describes how the analysis was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/539).*

# Task #553 — Methodology, hyperparameters, and worked examples

A methodology + statistical-parameters reference for task #553 (Explore Persona Space), with verbatim input / command / output-skeleton examples pulled straight from the artifacts. This is a `kind: analysis` task: a CPU-only reviewed-inference pass over two committed measurement panels from parent experiments, plus one out-of-panel transfer check. No model is loaded, nothing is trained, nothing is generated, no pod is provisioned.

- Task: [https://eps.superkaiba.com/tasks/553](https://eps.superkaiba.com/tasks/553)
- Model: n/a — no model loaded in this task. The parent panels' slot reads were measured on Qwen-2.5-7B (marker token ` ※`, id 83399 with leading space; EOS `<|im_end|>`, id 151645 — pinned as constants in the shared loader and asserted against the parent schemas).

---

## 1. Input panels and conditions

The task re-analyzes two frozen measurement panels (plus one secondary emission panel), all committed in git. The panels are read-only inputs: touching, regenerating, or re-deriving any parent artifact was a registered must-ask deviation, and none occurred.

### 1.1 The context panel (#532 `logp_slot_followup`)

- 16 trained source personas (`A1`–`A5`, `B1`–`B5`, `C1`, `D1`–`D5`) × 26 evaluation contexts (the same 16 prompts as "ordinary" bystanders + 10 `instr_*` instruction-injected bystanders) = 416 cells, each measured on both model sides.
- Per cell, 50 probe questions, each storing the four-float slot contract at the post-response slot of the trained model's own on-policy response: `logp_marker`, `z_marker` (raw marker logit), `z_eos` (raw EOS logit), `logZ` (logsumexp of the logit row), plus `argmax_id`, `slot_kind`, `emitted_id`, `n_truncated_tokens`.
- Three measurement phases, distinguished by a `phase` string asserted at load: `A3_trained_on_own_R` (trained model scored at its own response slots; `per_cell_trained/`, 416 JSONs), `A2_base_on_trained_R` (base model scored at the same trained-text slots — the "matched-slot" base read; `per_cell_base/`, 416 JSONs), and `A1_base_prior_slots` (base model scored at the end of its OWN responses per bystander; `base_prior_logp.json`, 26 entries × 50 probes — the "own-response" base prior).
- Pair/context predictors from `predictors.json`: 16×26 `cosine_matrix` / `gauss_kl_matrix` / `js_v1_matrix`, per-bystander binary `base_prior` and graded `base_prior_extra_logp`. The JS predictor is excluded from all primary families (deprecated estimator per #540); cosine is the geometry representative.
- Known structural feature carried as a flag: sources `B1` and `C1` are quasi-duplicate prompts (`cosine(B1,C1) = 1.0` exactly; separate adapters), asserted at load and handled by registered drop slices.

Cell-level channels derived from the per-probe arrays (means over 50 probes): `margin_trained = mean(z_marker − z_eos)` on the trained side, `margin_base_matched` (same, base side), `dz_marker`, `dz_eos`, `dmargin` (trained − base differences), `dlogp`, `trained_logp`, `base_logp_matched`. A mean-of-difference identity assert checks every derived margin against the stored `mean_marker_eos_margin` field to 1e-6.

Cohorts/slices (plain-English names used in outputs):

| Cohort | Definition | n |
|---|---|---|
| Ordinary cross-context cells | ordinary bystanders, self-pairs excluded | 240 |
| Instructed strip | all 16 sources × 10 instructed bystanders | 160 |
| Pooled with cohort FE | ordinary cross + instructed strip, cohort indicator as fixed effect | 400 |
| Duplicate-dropped slices | B1/C1 dropped as source and/or bystander | 196 / 144 (cell-level); n=15 / 14 / 13 (source-level) |

### 1.2 The persona panel (#478/#531 `tidy_logit.parquet`)

- 40 CORE cells (16 at K=1, 8 each at K=2/4/8 sources per training mix) × 2 seeds (42, 137) × 35 held-out personas × 20 questions = 56,000 rows, 25 columns.
- Per row: the same four-float contract on both model sides (`z_trained`, `z_base`, `z_eos_trained`, `z_eos_base`, `logZ_trained`, `logZ_base`, rescored log-probs) plus derived channels (`dz`, `dlogZ`, `margin_trained`, `margin_base`, `dmargin`) and predictors (`min_dist` — the held-out persona's cosine distance to the nearest source in the cell's training mix, constant per (cell, persona); `K`; matched-slot `base_prior`).
- The loader derives `dz_eos = z_eos_trained − z_eos_base` and `run_id = cell_id + "_seed" + seed` (80 runs), and asserts the full shape, 40 cells, both seeds, 35 personas, 20 questions, `track == CORE` everywhere, and the min_dist/K constancy invariants.
- For the fixed-effects anatomy the panel is aggregated to 2,800 (run × persona) means (asserted complete and balanced, 80 × 35); question-level variation re-enters through cluster bootstraps. Aggregation avoids 20× pseudo-replication in the FE decomposition.
- The 35 held-out personas were never trained sources and never contrastive negatives (zero exact-label overlap with the parent's fixed 4-negative panel `software_engineer` / `kindergarten_teacher` / `helpful_assistant` / `no_persona` — asserted in the exposure script). No own-response base prior exists on this panel (the base model's own responses were never generated/scored there) — a named limitation that removes one ranker from the panel's analogue of the ranking table.

### 1.3 Secondary emission panel

`eval_results/issue_532/per_cell/loc_ep1/` — on-policy in-response emission rates on the same 16×26 cells, loaded via #539's `build_panel` with that loader's own step-0 gate. Used only as the secondary DV in the ranking-table deliverable.

### 1.4 Reuse fitness (why these inputs, verbatim from the plan's reuse checks)

The context panel IS the panel the unreviewed inline claims were made on (same cells, same validated corrected-slot read; re-deriving it would change the estimand); its conditions are complete (16 × 26, both model sides) and its margins span ~40 logits (not saturated). The persona panel is the only committed panel with held-out personas and per-question four-float reads on both sides; its rescore was validated in #531 against stored log-probabilities (worst case MAE 0.33 nats, rank correlation 0.996).

---

## 2. Analysis methodology

There is no training. The unit of work is seven analysis scripts plus one shared module, all on branch `issue-553`, each taking the same CLI (`--i532-dir`, `--i478-parquet`, `--out-dir`, `--fig-dir`, `--n-boot`, `--n-cluster-boot`, `--n-marginal-boot`, `--n-perm`, `--seed`). The smoke invocation is the SAME script with reduced rep counts — one code path for smoke and production.

### 2.1 Step-0 reproduction gates (every script, fail-loud)

Before any new statistic, every script rebuilds each panel it uses from the raw committed files and reproduces committed parent values to a 1e-6 tolerance, exiting 1 on any mismatch:

- Context panel gate (`step0_i532`): cell counts (`n_cells` / `n_ordinary` / `n_instructed`), two Spearman reads, and one spread statistic from the parent's committed `analysis_logp.json`, coded against the JSON keys (`spearman.dmargin.cosine.rho_union`, `spearman.trained_logp.base_prior_logp.rho_union`, `graded_prior_spread.ordinary_sd_across_bystanders`), never against plan prose.
- Persona panel gate (`step0_i478`): one raw and three rank-residual partial Spearman reads reproduced from the parquet against the parent's committed `summary_logit.json` keys, with the partial implementation mirroring the #531 convention (rank-residualize both variables on [intercept | rank controls `min_dist`, `K`], Pearson on the residual ranks).
- The follow-up script adds reproduction gates that re-derive two committed context-level reads from this task's own frozen production JSONs (`transfer_478.json`, `channel_anatomy.json`) to the same 1e-6 tolerance before computing its partials.

The gate design distinguishes "panel loads differently" (a bug — fail loud, compute nothing) from "reviewed statistics disagree with the inline session" (a result, recorded in the `inline_vs_reviewed` blocks). Every output JSON persists its gate records.

### 2.2 The seven analysis programs

| Script | What it computes | Output |
|---|---|---|
| `issue553_panel.py` | Shared module (no CLI): panel builders + step-0 gates; two-way ANOVA Type-I variance shares (both factor orders) with cell-bootstrap and FE-respecting permutation nulls; centered gauge-invariant FE coefficient vectors; generic cluster-bootstrap plumbing with drawn-copy relabeling; the cell-axis (40-cluster) bootstrap for the persona panel; OLS + CGM two-way plug-in SE cross-check; the fast exact two-way FE solver (§2.3); metadata/`inline_vs_reviewed`/JSON writers; the shared CLI parser | imported by all consumers |
| `issue553_transfer_478.py` | PRIMARY deliverable — the persona-panel transfer check of the three-part channel anatomy: (a) two-way (run + persona) variance shares of `dz` with order-swap check, per-K stratified shares, per-run SD distribution, and a registered dominance statistic `run_share − max(persona_share, pair_share)` under the FE-re-estimating cell bootstrap; (b) the persona-FE vector of `dz_eos` (n=35) against persona-level base state, pair bootstrap + MC permutation; (c) pair-corrected (two-way FE re-estimated per resample) Spearman of `min_dist` against five channels with the full inference stack plus the cell-axis bootstrap; within-run ranking medians; aggregation of the parent rescore-validation blocks; argmax-composition rates; a side-by-side agreement table vs the context panel with per-seed splits | `eval_results/issue_553/transfer_478.json` |
| `issue553_unified_rule.py` | Joint OLS fits `margin_trained ~ α·prior_margin_own(B) + β·cosine(S,B)` on the three registered cohorts, with variants (+ standardized interaction term, judged within each DV space; + source FE labeled `post-train forecast-where`; duplicate-dropped slice); coefficient inference via source- and bystander-cluster bootstraps (wider one-way CI primary) + CGM cross-check; LOBO (26-fold) and LOSO (16-fold) out-of-fold R² per feature set (`+srcFE` excluded from LOSO — a held-out source has no estimable FE); shift readouts derived algebraically from the absolute fit, never fit standalone; a collinearity gate (per-cohort Pearson between the two features, tercile-bucket fallback above \|r\| > 0.6) | `eval_results/issue_553/unified_rule.json` |
| `issue553_channel_anatomy.py` | Two-way (source + bystander) variance shares for `Δz(※)` / `Δz(EOS)` / `Δmargin` on the ordinary-cross and instructed cohorts, with order-swap checks, cell bootstraps on the shares, and permutation-null share distributions persisted next to the observed shares; the five pair-corrected cosine Spearman reads with the full inference stack and Holm over the five; a bystander-level prior correlation of the clamp; a registered split-half / cross-fit probe-level robustness slice (odd/even-probe reliabilities + cross-fit reads so shared finite-probe noise cannot couple predictor and DV); the absolute trained-side z(EOS) anatomy; argmax-composition rates; a persisted headline scope rule for the analyzer | `eval_results/issue_553/channel_anatomy.json` |
| `issue553_diag_spill.py` | 16 source-level points: diagonal `margin_trained` (S==B) vs the source FE of off-diagonal `Δmargin` (and the `Δz(※)` variant); Spearman + source-pair bootstrap + MC label permutation at n=16 / 15 / 14 / 13 (registered duplicate- and thin-prompt-drop slices) + per-source leave-one-out influence | `eval_results/issue_553/diag_spill.json` |
| `issue553_ranking_table.py` | Per-source Spearman across bystanders of four rankers (base matched-slot margin; own-response prior margin; cosine; a z-scored prior+cosine stack) vs the trained margin level (primary) and the emission rate (secondary), on all-25 and ordinary-15 slices; the 16 per-source ρ reported as a distribution with a source-level bootstrap CI on the median; top-ranker comparisons judged on the PAIRED per-source difference bootstrap, not marginal median-CI overlap; degenerate per-source reads reported + dropped with count; explicit `pre-training forecast` vs `post-train forecast-where` labels per ranker | `eval_results/issue_553/ranking_table.json` |
| `issue553_exposure.py` | The negative-set exposure analysis: within the context panel, the ordinary (trained-negative contexts) vs instructed (never-clamped) `Δz(EOS)` contrast with a bystander-cluster bootstrap CI on the difference, the within-instructed prior gradient, and the within-ordinary spread (exposure dose is constant there by design — 20 negative rows per bystander, from the parent training-mix builder `i474_phase23_train.py::_build_negative_rows`); the persona panel's never-negative `dz_eos` distribution reported NEXT TO the two context-panel exposure classes as a qualitative contrast only (no pooled cross-panel fit — a registered forbidden move); a persisted non-separability statement | `eval_results/issue_553/exposure.json` |
| `issue553_followup_clamp_partial.py` | The free-analysis follow-up: partial rank Spearman (the #531 convention) of the clamp's context-level FE vs context-mean base margin CONTROLLING for context-mean base z(EOS) level, plus the complementary partial, on both panels (own-response regime primary on the context panel, matched-slot regime as sensitivity; regimes never mixed within one triple); unit-axis percentile bootstrap (rank-residualization re-fit per resample) + Freedman–Lane-style residual MC permutation; reproduction gates against the committed parent JSONs; an explicit `interpretation_guard` block | `eval_results/issue_553/followup_clamp_partial.json` |

A ninth script, `issue553_relabel_figures.py`, re-renders eight reader-facing figures with plain-English label strings (review rounds 2–3). It recomputes NO statistic: every plotted number is read from the frozen production JSONs; scatter/strip points come from the same deterministic panel loaders (pure data loading — no bootstrap, no permutation, no new fit).

### 2.3 Performance fix: fast exact two-way FE solver + BLAS single-thread cap

The plan's 10,000-rep FE-re-estimating bootstraps were wall-infeasible with the inherited `np.linalg.lstsq` on the explicit n×(1+n_a+n_b) dummy design (~2 s/call at 2,800×116 in this environment). Two documented fixes, both estimand-preserving:

1. **Gram/pinv solver** (`_twoway_gram_coefs` / `fast_twoway_resid_pair`): the min-norm least-squares coefficients are computed via the exact identity `X⁺ = (X'X)⁺ X'` — the pseudoinverse of the small (1+n_a+n_b)² gram matrix (built from count vectors + the A×B crosstab, never forming the design matrix) applied to `X'y`. Identical estimand to lstsq; ~1,000× faster. Equivalence is asserted at import time on a random unbalanced panel with an absent group level (drift < 1e-8 aborts the import) AND per-script on the observed panels; the solver is then monkey-patched into the imported #539 inference module so the inherited bootstrap/permutation loops use it.
2. **BLAS single-thread cap** (`threadpool_limits(limits=1, user_api="blas")` at module scope): OpenBLAS spawns 32 pthreads on this VM and the small linear-algebra calls degrade from thread-sync thrash; every op in this task is small, so BLAS is capped at 1 thread process-wide (the controller object is kept referenced at module scope so the limit is not restored on GC).

### Statistical parameters

This task has no training hyperparameters. The load-bearing knobs are the inference settings, read from the shared CLI defaults (`issue553_panel.py::common_parser`) and cross-checked against the `metadata` block recorded in every production output JSON (which records the resolved argv; the two agree).

| Parameter | Value | Notes |
|---|---|---|
| **Cell-level bootstrap reps** (`--n-boot`) | **10,000** | percentile; any FE/residualization re-estimated inside every resample; #539 convention |
| **Cluster-level bootstrap reps** (`--n-cluster-boot`) | **2,000** per axis | drawn cluster copies relabeled as distinct groups (#539 convention); #531 used 1,000 — upgraded here for consistency with #539 |
| **Source-marginal pair-bootstrap reps** (`--n-marginal-boot`) | **2,000** | for n=16/35 unit-level reads (#539 `source_marginal` recipe) |
| **Permutation reps** (`--n-perm`) | **10,000** | two-sided, add-one formula; FE-respecting (within-level label shuffles); ONE registered exception: variance-share nulls are one-sided-large by construction |
| **Seed** | **42** | all RNG via `np.random.default_rng` (matches #539/#531) |
| Multiplicity | Holm per registered primary family | transfer family: 2 p-carrying members (the dominance member is judged from its bootstrap CI); channel-decomposition family: 5 members |
| Clustering axes | context panel: source (16) + bystander (26/16/10); persona panel: run (80) + persona (35) + cell (40) | primary CI = the WIDER of the one-way cluster CIs; the cell axis joins the wider-of set for all `min_dist` reads (the two seeds of a cell share one training mix) |
| Two-way SE cross-check | CGM (Cameron–Gelbach–Miller 2011) plug-in, `V = V_a + V_b − V_(a∩b)` | cross-check only, never headline; no small-sample correction (documented); non-PSD outcomes flagged in the output, never silently NaN'd |
| Degenerate resamples | dropped AND counted | count persisted per read |
| Step-0 gate tolerance | 1e-6 | also the mean-of-difference identity tolerance |
| GPU-hours | 0 | budgeted 0; no pod |

### Conventions enforced relative to the inline session (the method delta)

The reviewed pass re-fits the same quantities under registered conventions (this is what the task IS, not a finding): (1) the EOS margin `z(※) − z(EOS)` as the modeling DV, absolute trained state in a joint fit, shift readouts derived algebraically and never fit as standalone marginal correlations; (2) per-cohort fits AND pooled-with-cohort-FE — never a pooled cross-cohort headline correlation; (3) two-way cluster uncertainty on every coefficient (the inline session clustered on one axis only); (4) the B1/C1 quasi-duplicate kept in the main fit and dropped in registered robustness slices; (5) feature sets labeled `pre-training forecast` vs `post-train forecast-where` (source-FE and matched-slot quantities are never quoted as pre-training performance); (6) no cross-DV R² comparisons to pick a space — space comparisons stay descriptive.

---

## 3. Evaluation methodology

### Dependent variables

All DVs derive from the committed teacher-forced four-float slot reads at the post-response slot of the trained model's own on-policy response (the corrected pre-marker slot). Per the plan's measurement-validity table:

| DV | Construct | Metric | On-distribution? |
|---|---|---|---|
| `margin_trained`, `Δz(※)`, `Δz(EOS)`, `Δmargin` (context panel) | implant strength / leakage channels at the natural emission slot | teacher-forced four-float read at the post-response slot of the model's own on-policy response | yes — on-policy response, natural slot; teacher-forcing is only the read mechanism, validated in #531 (re-score vs stored: MAE ≤ 0.33 nats, ρ ≥ 0.996) |
| same channels (persona panel) | same | same, from #531's rescore (per-JSON `validation_vs_stored_logp` blocks, aggregated worst-case in the transfer script) | yes — same construction |
| emission rate (ranking-table secondary) | behavioral firing | ` ※` anywhere in the model's own response | yes — inherited from the parent's binding DV |

The logit channels are gauge-valid because the parent adapters' `target_modules` exclude `lm_head`/`embed_tokens` and `modules_to_save` is empty (asserted per-adapter in #531; inherited — no adapter is touched here). The matched-slot base read is `A2_base_on_trained_R` (base model scored at slots created by the trained model's text); a persisted headline scope rule caps what the analyzer may claim from it and forbids citing cross-panel agreement as evidence against mechanical-coupling/text-mediation accounts, since both panels share the matched-slot construction.

### Metrics

The metrics computed (values live in the clean result, not here): two-way Type-I ANOVA variance shares in both factor orders with bootstrap CIs and permutation-null distributions; raw, pair-corrected (two-way-FE-residualized), and rank-residual partial Spearman correlations with bootstrap CIs and FE-respecting permutation p; OLS coefficients with cluster-bootstrap CIs and CGM cross-check SEs; LOBO/LOSO out-of-fold R²; paired per-source difference bootstraps; split-half (Spearman–Brown) reliabilities and cross-fit correlation variants; distribution summaries (medians + IQR + per-unit values). Sample sizes per read: 416 / 400 / 240 / 196 / 160 / 144 cells (context-panel cohorts), 2,800 run×persona aggregates over 56,000 rows (persona panel), 16 / 15 / 14 / 13 sources, 35 personas, 26 / 16 / 10 bystanders, 80 runs, 40 cells.

Every output JSON carries an `inline_vs_reviewed` block — one row per inline quantity with the inline value, the reviewed value, a `convention_changed` flag, and a note — plus the step-0 gate records and the full metadata block.

### Pipeline phases

| Phase | Script(s) | Output |
|---|---|---|
| 1. Implementation + review (commit `108a21030267bc54a24e171c856c43fcc8080e21`, solver fix `149b9a4d56f924870c4f540a4f66696afdd21d7e`) | six analysis scripts + shared module; smoke = same scripts at reduced reps | code on branch `issue-553` |
| 2. Production run (code commit `60b4f613b`, ~31.5 min CPU total, local VM) | the six main scripts at 10,000/2,000/2,000/10,000 reps, seed 42 | 6 output JSONs + 21 figure stems (committed at `73c7bf50e`) |
| 3. Figure relabeling (rounds 2–3) | `issue553_relabel_figures.py` — label strings only, frozen JSONs | re-rendered figures at `52faf4c21` and `09aa76a64` |
| 4. Free-analysis follow-up | `issue553_followup_clamp_partial.py` (executed with the repo at `52faf4c21`) | `followup_clamp_partial.json` + one figure (committed at `3a4a04725`) |

---

## 4. Worked example — input rows (verbatim)

One cell from the context panel, trained side (`per_cell_trained/A1__B3.json` — source `A1` scored in bystander context `B3`), with the 50-entry `per_q` array truncated to its first element:

<!-- cherry-picked for illustration; full panel at the commit-pinned directory linked below -->

```json
{
  "schema_version": "issue532_followup_logp_v1",
  "phase": "A3_trained_on_own_R",
  "source_cid": "A1",
  "bystander_label": "B3",
  "n_probes": 50,
  "per_q": [
    {
      "logp_marker": -14.000001907348633,
      "z_marker": 16.375,
      "z_eos": 30.375,
      "logZ": 30.375001907348633,
      "logp_bare_marker": -21.875001907348633,
      "argmax_id": 151645,
      "slot_kind": "end_of_response",
      "emitted_id": null,
      "n_truncated_tokens": 0
    },
    "... 49 more probes ..."
  ],
  "summary": {
    "mean_logp_marker": -8.72165512084961,
    "mean_z_marker": 16.22625,
    "mean_z_eos": 24.765,
    "mean_logZ": 24.947905120849608,
    "mean_logp_bare_marker": -17.209780120849608,
    "mean_marker_eos_margin": -8.53875,
    "argmax_marker_rate": 0.02,
    "n_pre_marker_slots": 1
  }
}
```

The matching base-side file `per_cell_base/A1__B3.json` carries the same shape with `"phase": "A2_base_on_trained_R"`; the loader asserts per-probe `slot_kind` and `n_truncated_tokens` agree across the two sides (same slots).

One row from the persona panel (`tidy_logit.parquet`, row 0 — deterministic, not cherry-picked):

```text
cell_id                K1_c00          z_trained                9.5625
seed                   137             z_base                   1.539062
track                  CORE            z_eos_trained            24.75
K                      1               z_eos_base               27.0
held_out_persona       medical_doctor  logZ_trained             24.752174
question_idx           0               logZ_base                27.000296
trained_logp           -15.564599      logp_trained_rescored    -15.189674
base_prior             -25.617445      logp_base_rescored       -25.461233
shift                  10.052846       dz                       8.023438
min_dist               0.032049        dlogZ                    -2.248121
band                   near            dlogp_rescored           10.271559
                                       margin_trained           -15.1875
                                       margin_base              -25.460938
                                       dmargin                  10.273438
```

(56,000 such rows; the analysis scripts additionally derive `dz_eos` and `run_id` at load.)

---

## 5. Worked example — command + output skeleton (verbatim)

The production invocation (each script independent; same code path as smoke, which appends reduced rep flags):

```bash
for s in transfer_478 unified_rule channel_anatomy diag_spill ranking_table exposure; do
  uv run python scripts/issue553_${s}.py \
    --i532-dir eval_results/issue_532 \
    --i478-parquet eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet \
    --out-dir eval_results/issue_553 --fig-dir figures/issue_553 \
    --n-boot 10000 --n-cluster-boot 2000 --n-perm 10000 --seed 42
done
# follow-up partial check (reads the committed transfer_478.json / channel_anatomy.json
# from --parent-553-dir for its reproduction gates; same flags otherwise)
uv run python scripts/issue553_followup_clamp_partial.py \
  --i532-dir eval_results/issue_532 \
  --i478-parquet eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet \
  --parent-553-dir eval_results/issue_553 \
  --out-dir eval_results/issue_553 --fig-dir figures/issue_553 \
  --n-boot 10000 --n-cluster-boot 2000 --n-perm 10000 --seed 42
```

Top-level key skeleton of the primary output, `transfer_478.json` (result values live in the committed JSON, linked below):

```text
metadata | step0_i478 | step0_i532 | n_run_persona_aggregates | anatomy
clamp_routing | absolute_z_eos_trained_persona_fe | min_dist_corrected_reads
min_dist_per_seed_point_estimates | within_run_ranking | rescore_validation
holm | i532_side_points | agreement_table | inline_vs_reviewed
```

Its `metadata` block, verbatim (the reproducibility record every output JSON carries):

```json
{
  "task": 553,
  "script": "issue553_transfer_478.py",
  "git_commit": "60b4f613b3059919df2ba4ae772bdb45a4d09c8c",
  "timestamp_utc": "2026-06-10T11:56:01.672608Z",
  "python_version": "3.11.15",
  "numpy_version": "2.2.6",
  "scipy_version": "1.17.1",
  "pandas_version": "2.3.3",
  "platform": "Linux-6.8.0-1052-gcp-x86_64-with-glibc2.35",
  "seed": 42,
  "n_boot": 10000,
  "n_cluster_boot": 2000,
  "n_marginal_boot": 2000,
  "n_perm": 10000,
  "argv": ["--n-boot", "10000", "--n-cluster-boot", "2000",
           "--n-marginal-boot", "2000", "--n-perm", "10000", "--seed", "42"]
}
```

There are no model generations and no raw completions anywhere in this task's measurement chain — the parent panels store logit reads, not text — so the usual eval-prompt + model-output worked example is n/a.

---

## 6. Artifacts and reproducibility

Commit roles (all SHAs verified from the task's Reproducibility record and the branch log):

- **Production-run code commit:** `60b4f613b3059919df2ba4ae772bdb45a4d09c8c` — recorded in every output JSON's `metadata.git_commit` and every figure's `meta.json` sidecar
- **Outputs committed as artifacts:** `73c7bf50e246b551226ee9a507cc7cceb3d299a7` (script content identical to the run commit)
- **Figure relabeling (label strings only):** `52faf4c21f3bb7c1801b1aac464992816939c11f` (round 2, five figures) and `09aa76a64acadf9dde5b37c081a61a96978a21b2` (round 3, three figures)
- **Follow-up partial check:** executed with the repo at `52faf4c21f3bb7c1801b1aac464992816939c11f`; script + JSON + figure landed at `3a4a04725b848ed531d0b708dd2c0ee1bed7ad08`

Code (commit-pinned):

- **Shared module:** [scripts/issue553_panel.py](https://github.com/superkaiba/explore-persona-space/blob/60b4f613b3059919df2ba4ae772bdb45a4d09c8c/scripts/issue553_panel.py)
- **Six deliverable scripts:** [scripts/ at the run commit](https://github.com/superkaiba/explore-persona-space/tree/60b4f613b3059919df2ba4ae772bdb45a4d09c8c/scripts) (`issue553_transfer_478.py`, `issue553_unified_rule.py`, `issue553_channel_anatomy.py`, `issue553_diag_spill.py`, `issue553_ranking_table.py`, `issue553_exposure.py`)
- **Follow-up script:** [scripts/issue553_followup_clamp_partial.py](https://github.com/superkaiba/explore-persona-space/blob/3a4a04725b848ed531d0b708dd2c0ee1bed7ad08/scripts/issue553_followup_clamp_partial.py)
- **Figure relabeler:** [scripts/issue553_relabel_figures.py](https://github.com/superkaiba/explore-persona-space/blob/09aa76a64acadf9dde5b37c081a61a96978a21b2/scripts/issue553_relabel_figures.py)
- **Reused inference machinery (#539):** [scripts/issue539_residual_per_cohort.py](https://github.com/superkaiba/explore-persona-space/blob/60b4f613b3059919df2ba4ae772bdb45a4d09c8c/scripts/issue539_residual_per_cohort.py), [scripts/issue539_corrected_reads_inference.py](https://github.com/superkaiba/explore-persona-space/blob/60b4f613b3059919df2ba4ae772bdb45a4d09c8c/scripts/issue539_corrected_reads_inference.py) (imported as modules; fast-solver monkeypatch documented in §2.3)

Inputs (commit-pinned, read-only):

- **Context panel:** [eval_results/issue_532/logp_slot_followup/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/logp_slot_followup) (416 + 416 per-cell JSONs + `base_prior_logp.json` + `analysis_logp.json`) and [predictors.json](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/predictors.json)
- **Persona panel:** [tidy_logit.parquet](https://github.com/superkaiba/explore-persona-space/blob/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_478/base_prior_reanalysis/tidy_logit.parquet) + [eval_results/issue_478/logit_rescore/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_478/logit_rescore) (80 run JSONs)
- **Emission panel (secondary DV):** [eval_results/issue_532/per_cell/loc_ep1/](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_532/per_cell/loc_ep1)

Outputs:

- **Six production JSONs:** [eval_results/issue_553/ at 73c7bf50e](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/eval_results/issue_553)
- **Follow-up JSON:** [followup_clamp_partial.json](https://github.com/superkaiba/explore-persona-space/blob/3a4a04725b848ed531d0b708dd2c0ee1bed7ad08/eval_results/issue_553/followup_clamp_partial.json)
- **Figures:** 21 stems × (png, pdf, meta.json) — 6 hero + 15 exploratory — at [figures/issue_553/ (73c7bf50e)](https://github.com/superkaiba/explore-persona-space/tree/73c7bf50e246b551226ee9a507cc7cceb3d299a7/figures/issue_553); relabeled renders at [52faf4c21](https://github.com/superkaiba/explore-persona-space/tree/52faf4c21f3bb7c1801b1aac464992816939c11f/figures/issue_553) and [09aa76a64](https://github.com/superkaiba/explore-persona-space/tree/09aa76a64acadf9dde5b37c081a61a96978a21b2/figures/issue_553)
- **Hydra config:** n/a (standalone analysis scripts with CLI flags)
- **HF Hub / WandB:** n/a — nothing trained, nothing generated; git is the permanent store for all inputs and outputs
- **Compute:** 0 GPU-hours (budgeted 0); 31.5 min wall on the local VM's CPU for the six main scripts (the follow-up script was not separately timed); no pod provisioned

---

*This document describes how the analysis was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/553).*

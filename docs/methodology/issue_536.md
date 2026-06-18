# Task #536 — Methodology, hyperparameters, and worked examples

A methodology + analysis-parameter reference for task #536 (Explore Persona Space), with verbatim audit / re-grade / refit rows pulled straight from the artifacts. Task #536 is `kind: analysis`: a CPU-only correctness audit that (1) pinned ONE canonical persona-distance cosine definition, (2) swept the codebase for every persona-distance cosine-computation site and classified each as mean-centered vs raw, and (3) recomputed prior tasks' headline cosine-vs-leakage statistics under the canonical metric from persisted artifacts, assigning labels from a pre-registered decision tree. There is no training run; the documentable methodology is the audit + recompute recipe.

- Task: [https://eps.superkaiba.com/tasks/536](https://eps.superkaiba.com/tasks/536)
- Model: none loaded — deterministic CPU linear algebra over persisted artifacts. (The input centroid banks were extracted from Qwen-2.5-7B residual streams by the parent tasks, per the plan; this task never runs a forward pass.)

---

## 1. The canonical metric pin and the audit-scope conditions

### Canonical persona-distance cosine (the pinned definition)

**Bank cosine (canonical): globally mean-center the centroid bank → L2-normalize → cosine**, i.e. `compute_cosine_matrix(C, centering="global_mean")` from `src/explore_persona_space/analysis/representation_shift.py` — the recipe the library applies when no centering argument is passed. The recompute driver reuses this function directly; the centering is never reimplemented.

The pin defines **two labeled families, never numerically compared**:

- *Bank cosine* (N-persona centroid bank, N≥10 recommended): the canonical form above. The mean depends on the bank, so centered values are only comparable within the same bank; every artifact records the bank's `persona_names` (hashed per row as `names_hash`).
- *Raw pairwise cosine* (exactly two vectors, no bank — the #404/#458 predictor family): no bank exists to center against (centering a 2-element "bank" by its own mean degenerates to cosine ≡ −1). Allowed, but MUST be labeled `raw pairwise (uncentered)` and never compared to bank-cosine values.

The pin landed as the `## Bank centering — canonical (task #536)` section of [.claude/rules/persona-distance-metrics.md](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/.claude/rules/persona-distance-metrics.md).

### Audit-scope table (the conditions analog)

The table rows are partitioned by five config slugs (table bookkeeping only, not reader-facing condition names):

| Slug | What it covers | What it controls for |
|---|---|---|
| `regrade_raw` | Raw-path tasks whose headline statistic is recomputed under the canonical metric | join validity via the raw-reproduction gate |
| `verify_centered` | Centered-path (canonical-line) tasks whose published statistic is reproduced from persisted artifacts | artifact rot / silent recipe drift |
| `readoff_r6` | Tasks whose artifacts already carry both raw and mean-centered matrices (prior remediation) — read off, no new compute | none (no new compute) |
| `approx_gram` | Matrix-only tasks (centroids lost; only a cosine/distance matrix persists) — approximate sensitivity reads in a separate label namespace | rotation-invariance of the Gram-configuration recovery |
| `scope_pairwise` | The 2-vector pairwise family — labeled under the pin's second family, never re-graded (no bank ⇒ no canonical recompute exists) | family mislabeling |

The shipped re-grade table holds 30 rows across these slugs; the audit table holds 36 site rows + 2 deleted-producer rows + 105 dispositions over 138 swept files + 16 artifact fingerprint checks.

---

## 2. Audit methodology (Phase A — the two-pass sweep)

Script: `scripts/issue536_audit.py`. Deterministic; re-runnable end-to-end.

### Two regex passes

Every `*.py` file under five roots — `scripts/`, `experiments/`, `src/explore_persona_space/analysis/`, `src/explore_persona_space/axis/`, `src/explore_persona_space/experiments/` — is matched against two compiled regexes (verbatim from the script):

```python
PASS1 = re.compile(r"compute_cosine_matrix|cosine_similarity|F\.cosine|cos_sim|centering=")
PASS2 = re.compile(
    r"F\.normalize|np\.linalg\.norm|torch\.linalg\.norm|\.norm\(|np\.dot|torch\.dot"
    r"|einsum|@\s*\S*\.T|matmul|cosine_matrix|_cosine|cos_"
)
```

Pass (i) is the literal-API pass; pass (ii) is the hand-rolled-idiom pass, deliberately high-recall / low-precision — completeness is defined by the expanded sweep, and N-A classifications are cheap. The plan pre-registered three reconciler-named misses of the literal pass that the idiom pass must catch (`contrastive_neg_geometry_504/phase05.py`, `analyze_100_persona_source_filtered.py`, `plot_100_persona_scatter_simple.py`); all three appear as classified SITE rows. The shipped sweep hit 138 files (28 via pass 1).

### Coverage enforcement (fail-loud)

Every swept file must be accounted for by either a curated **SITE row** (a persona-distance cosine *computation* site, classified `centered | raw | both-computed | N-A` with `file:line` evidence + consuming tasks) or a curated **DISPOSITION** (consumer of an upstream artifact / different construct / infra). Any unaccounted file raises `RuntimeError` and fails the run:

```python
unaccounted = sorted(set(sweep) - site_files - set(DISPOSITIONS))
if unaccounted:
    raise RuntimeError(f"audit coverage FAILED — {len(unaccounted)} swept file(s) ...")
```

Each disposition row stores the file's matched-idiom evidence lines tagged `[pass1]`/`[pass2]` (up to 6, comments skipped), so the disposition is auditable without re-running the sweep. Curated entries that no longer match the current sweep are logged as `[stale]` warnings rather than dropped — entries can be pre-registered for the post-merge tree.

### Deleted-producer recovery via git history

Two load-bearing producers no longer exist in the working tree and were recovered via `git log --all -S 'cosine' --diff-filter=D` plus targeted `git show`:

- `scripts/i406_phase1_merge_and_compute_matrices.py` @ `9e6e31c3f9222cf6d1dc95c31b0ae19d6230a58d` — producer of `eval_results/issue_406/cosine/C_L*.json` (per-layer 20×20 cosine-distance matrices, normalize-only, persisted as distance `1 − cos`).
- `scripts/_issue478_common.py` @ `69b34b94e00cf4c16b830f32da605289ef35371c` — `_build_matrix_from_centroids`: L2-normalize per row → pairwise similarity → `distance = 1 − sim` (no centering), cached to `cosine_distance_matrix_layer20.json` (the #478/#490 111-bank distance source).

Both get DELETED_SITES rows with the recovering commit recorded in the `lines` field.

### Artifact fingerprint checks (classify by what the consumer actually read)

Scripts that compute BOTH matrices are classified by **what the downstream consumer actually read**, not what the script computed. The check loads each load-bearing persisted artifact and records its off-diagonal min/median/max plus a regime call:

```python
stats["regime_call"] = (
    "raw band (compressed)" if stats["min"] > 0.25 else "centered span (negatives reached)"
)
```

Raw-regime bank cosines on this model compress into roughly [0.7, 1.0]; centered matrices span negative values. Distance-form artifacts (#406's `C_L*.json`) are converted `G = 1 − D` before fingerprinting, per layer (the fingerprint is layer-specific). 16 fingerprint checks ship in the audit table. The audit also records the Phase A0 literature note (the Persona Vectors difference-of-means construction, verified at fact-check time via the arxiv-latex MCP) as a structured record.

---

## 3. Recompute methodology (Phase C — family registry, adapters, join gates)

Script: `scripts/issue536_recompute_driver.py`. One **FAMILY_REGISTRY** of centroid banks / Gram matrices + one **TASK_ADAPTERS** table that joins recomputed X with each task's persisted Y and recomputes that task's OWN published estimator (never a new one). Each task's own published layer is used (L20 for the #66/#405 lines, L21 for #505, L15 for the n=24 predictor bank) — the audit changes ONE variable (centering), never layer or estimator.

### Family registry (input banks)

| Family | Bank | Layer(s) | Source artifact | Join gate |
|---|---|---|---|---|
| `single_token_100p_L20` | 111 personas | 20 | `eval_results/single_token_100_persona/centroids/centroids_layer20.pt` (names from the cached distance JSON) | **matrix-1e-4**: recomputed raw distance must reproduce `cosine_distance_matrix_layer20.json` (metric asserted `"1 - cosine"`) |
| `extraction_method_a_L20` | 20 personas | 20 | `eval_results/extraction_method_comparison/centroids_method_a.pt` (`layer_20` key) | **matrix-1e-4** vs `cosine_matrix_a_layer20.json` |
| `issue274_n24_L15` | 24 personas | 15 (bundle spans 0–27) | `eval_results/issue_274/centroids/centroids_n24_layers0_27.pt` | statistic-level only (no published matrix persists for this bundle); bank centered-norms also returned for the reference-norm check |
| `issue505_pv_L21` | persona-vector centroids | 21 | HF `issue505_loo_contrastive/geometry/centroids_pv_L21.pt` (local fallback first) | **matrix-1e-4**: recomputed raw cos(b, j) vs persisted `eval_results/issue_505/analysis/panel_similarity_matrix.json` over all overlapping (b, j) pairs |
| `issue406_gram` | 16 transformation conditions | 0, 5, 11, 15, 21, 27 | `eval_results/issue_406/cosine/C_L{layer}.json` (distance form, converted `G = 1 − D`) | matrix-only (Gram read; sensitivity namespace) |
| `issue341_gram` | 20 personas | per-layer | `experiments/phase_minus1_persona_vectors/cosine_matrix.json` (true similarity Gram) | matrix-only (Gram read; sensitivity namespace) |

Every family computes BOTH `cos_raw = compute_cosine_matrix(C, centering="none")` and `cos_mc = compute_cosine_matrix(C, centering="global_mean")` from the same tensor, and records the bank size, `names_hash` (sha256 of the joined persona names, first 12 hex chars), and the gate level applied.

### Join-gate hierarchy and the reproduce-before-read rule

The mandatory join-validity gate: **the raw recompute must reproduce the published/persisted number BEFORE the centered value is read.** Three gate levels, recorded per row in `gate_level` so the strength of the identity defense is visible:

1. **matrix-1e-4** (`GATE_MATRIX_TOL = 1e-4`) — where a published matrix persists: max absolute elementwise deviation of the recomputed raw matrix vs the cached artifact must be ≤ 1e-4 (strongest — an identity defense on the full join).
2. **row-1e-4** — row-level derived quantities (e.g. #478's per-row `min_dist` to the K-subset, #490's per-row d_A/d_B) recomputed from the bank must match the snapshot's own stored column within 1e-4.
3. **statistic-0.02** (`GATE_RHO_TOL = 0.02`) — where only the published statistic exists: the raw-recipe recomputation must land within |Δρ| ≤ 0.02 of the published value (slope rows: within 0.02 of the published slope / inside the published interval). Weaker and drift-susceptible; rows passing only this level say so in `gate_level`.

A gate failure raises `RuntimeError` — the row is `join-failed` and exits the decision tree with no centered re-grade; a misaligned join is never read silently. For canonical-line verification rows, a gate failure is escalated in the error text as "canonical-line verification inconclusive — investigate bank-version drift, do not proceed" (plan H1: a verification join-failure is not one bad row).

### Checkpoint-per-row writes

Every adapter appends its row to `eval_results/issue_536/regrade_table.json` the moment it is computed (`append_row`: atomic tmp-then-replace, idempotent on `row_id`), and the figures payload sidecar is re-written after every adapter — never accumulate-then-write. The driver supports `--only <adapter>` for single-adapter re-runs.

### Task adapters (the registry, verbatim keys)

```python
TASK_ADAPTERS = {
    "66": verify_66,            "99": row_99,
    "142": verify_142,          "311": verify_311,
    "380": verify_380,          "canonical_partition": partition_canonical_line,
    "405": regrade_405,         "478": regrade_478,
    "490": regrade_490,         "396_415": regrade_396_415,
    "505": regrade_505,         "474_lineage": regrade_474_lineage,
    "341": regrade_341,         "213_227": partition_213_227,
    "472_504": readoff_472_504, "pairwise": scope_pairwise,
}
```

Adapter conventions, held across all rows:

- **Same Y, same join, same estimator** as the original task; the single changed variable is the centering of X. Estimator code is mirrored verbatim where the original was hand-rolled (e.g. `length_partial_spearman` mirrors `scripts/analyze_issue415.py`; `rank_residual_partial` mirrors `scripts/i474_cosine_followup.py::_length_partial`; #380's `rank_residualize` is a verbatim mirror of `i380_cosine_pairwise.py`).
- **Raw slope coefficients never drive labels** — centering rescales X by construction, so raw-vs-centered beta deltas ship descriptively only; the scale-invariant magnitude M (Spearman ρ, standardized β, partial R², or fixed-quantile contrast) drives the tree.
- Each re-graded row also records the **Spearman correlation between the raw and centered X vectors actually joined** (`raw_vs_centered_x_spearman`), bank size/composition (`bank` block), and per-cell N/df counts.
- **Unrecoverable partitions are evidence-backed**: rows for tasks whose X or Y artifacts do not persist record what is missing, where it was looked for (local path, `git log --all` commit counts for the directory, HF data repo via `huggingface_hub.list_repo_files`, and — for #213/#227 — a live `wandb.Api` scan over all 175 projects of the entity for candidate centroid artifacts), and what a GPU follow-up would need.
- #478's flatness row carries a machine-readable `estimator_caveat` block plus a registered `follow_up` pointer naming the published-estimator refit required before any unconditional claim (resolved by the §5 protocol below; the refit writes a `follow_up_result` field into the same row, leaving `regrade_label` untouched).

### Pipeline phases

| Phase | Script | Output |
|---|---|---|
| A — two-pass sweep + dispositions + fingerprints | `scripts/issue536_audit.py` | `eval_results/issue_536/audit_table.json` |
| C — family recomputes + per-task re-grades (join-gated) | `scripts/issue536_recompute_driver.py` | `eval_results/issue_536/regrade_table.json` + `figures_payload.json` |
| Figures (5) | `scripts/issue536_figures.py` | `figures/issue_536/{hero_regrade_dumbbell, bank_offdiag_histograms, compression_scatter_406_L21, band_reassignment_478, forest_396_415}.{png,pdf}` + `meta.json` sidecars |
| Registered follow-up — published-estimator refit | `scripts/issue536_mixedlm_refit.py` | `eval_results/issue_536/mixedlm_refit_478.json` + in-place `follow_up_result` on the flatness row |

(Phase B, artifact recoverability, was plan-time verification — Hub checks via `huggingface_hub.list_repo_files`, local `find`/`ls`, plus the implementation-time WandB re-search; its outcomes are encoded in each row's `recoverability` field. Phase D is the rules-file pin diff; Phase E deferred all GPU re-extraction out of this task.)

Figure conventions: paper-plots style (PNG + PDF + meta.json), re-grade-namespace encoding in the hero dumbbell (raw recipe = open grey circle; mean-centered exact rows = filled circle; matrix-only approximate rows = open square, never pooled with exact rows), reader-facing descriptive row labels (no issue-number codes in rendered figure text). `figures_payload.json` persists every plotted float.

---

## 4. The pre-registered re-grade decision tree (§3 of the plan — mechanics)

Per row, the task's own published estimator is computed twice from the identical join: raw (must reproduce published per the gate, else `join-failed` and the row exits the tree) and centered. The row's scale-invariant magnitude M is Spearman ρ for correlation rows; standardized β, partial R², or a fixed-quantile near-far contrast for slope/regression rows. α per row = the task's own published threshold (default p<0.01 where the task did not publish one).

*Originally-SIGNIFICANT rows* (ordered, first match wins; implemented verbatim in `label_significant`):

1. Centered sign ≠ raw sign AND the centered estimate is itself significant at α → **flips**.
2. Centered estimate not significant at α → **weakens (significance lost)**.
3. |M_centered| ≤ 0.5·|M_raw| (significant, same sign) → **weakens (magnitude)**.
4. |M_centered| ≥ 1.5·|M_raw| (significant, same sign) → **strengthens**.
5. Otherwise → **stands**.

The 0.5×/1.5× boundaries are pre-registered conventions (`M_WEAKEN_FACTOR = 0.5`, `M_STRENGTHEN_FACTOR = 1.5`); raw numbers ship per row so a reader can re-derive labels under different boundaries.

*Originally-NULL rows* (first match wins):

1. The centered read becomes significant under the row's pre-registered multiplicity rule → **null-overturned (candidate rescue)**, reported with family size and corrected + uncorrected p.
2. Otherwise → **stands (null persists)**.

**The Holm family rule (multiplicity for null rescues).** The test family is every centered cell examined for that row's claim; significance = Holm-corrected p < 0.05 within the family (`holm_reject`: step-down over the named family, threshold α/(m−i)). Two registered families:

- The #396/#415 predictor row: 12 cells (6 DV surfaces × 2 cosine predictors), with the additional requirement |ρ| ≥ 0.5 (N=24). An uncorrected p<0.01 hit that fails Holm is recorded in `uncorrected_hits_not_rescues` — a candidate signal, never a rescue.
- The #478 flatness row: 2 tests {gap slope, K×log(dist) interaction}. The binding rule is Holm at α=0.05 within this family; the plan-H2 stricter α=0.01 read is computed and recorded alongside (`holm_reject_at_h2_alpha_001`) but is not the binding label rule — the binding/recorded split is stated inside the row so it cannot be rebound post-hoc.

*Matrix-only approximate rows* live in a SEPARATE label namespace — `sensitivity-agrees` / `sensitivity-disagrees` (same sign AND same significance status at the row's α vs the raw read) / `sensitivity-unreliable` — and NEVER receive canonical labels: heterogeneous centroid norms are lost in a persisted Gram matrix, so approximate/raw agreement is not a canonical certificate. Any canonical conclusion for these rows requires the deferred GPU re-extraction.

---

## 5. Matrix-only sensitivity recipe and the published-estimator refit protocol

### Approximate (normalized-vector centering) read for matrix-only rows

Implemented in `gram_centered_config` (driver). Input: a TRUE similarity Gram matrix of already-normalized vectors (distance-form artifacts are converted `G = 1 − D` first). Steps:

1. Symmetrize `G ← (G + Gᵀ)/2`.
2. Eigendecompose; clip negative eigenvalues at 0, recording the clipped mass as a fraction of the trace.
3. Recover the configuration `X = V·diag(√w)` (rows = recovered vectors, defined up to an orthogonal map — cosine is invariant under orthogonal maps, so the read is well-defined).
4. Global-mean-center the recovered unit vectors, re-normalize (zero-norm guard), re-cosine.

PSD handling: if the clipped mass exceeds `PSD_CLIP_TRACE_FRAC = 1e-3` of the trace, the row is labeled `sensitivity-unreliable` (no agree/disagree label). The clipped-mass fraction is recorded per layer in the row. This read applies only to true full similarity/distance matrices (#406 lineage, #341); the #213/#227 persisted JSON is a single cue→no_cue reference column, not a Gram matrix, so the read is impossible there and those tasks get evidence-backed partition rows instead.

### Published-estimator refit (`scripts/issue536_mixedlm_refit.py`)

The registered follow-up `478-estimator-mixedlm-refit` refits the published #478 co-primary estimator — the exact statsmodels MixedLM spec from `scripts/issue478_analyze.py::mixed_effects_K_x_logd` on branch `issue-478` — on BOTH joins. Spec verbatim (as recorded in the output JSON's `spec.estimator` field):

> statsmodels MixedLM: `deltaLogP_mean ~ K * log(min_dist_to_K_subset)`; `groups=subset_id` (random intercept); `vc_formula={'persona': '0 + C(held_out_persona)'}`; `fit(reml=False, method='lbfgs')`; rows filtered to join-distance > 0 [verbatim from `scripts/issue478_analyze.py::mixed_effects_K_x_logd`, branch issue-478]

Source code form (script docstring, mirroring the published analysis code):

```python
model = smf.mixedlm(
    "deltaLogP_mean ~ K * log_min_dist",
    df,                                  # rows filtered to min_dist > 0
    groups=df["subset_id"],
    vc_formula={"persona": "0 + C(held_out_persona)"},
)
result = model.fit(reml=False, method="lbfgs")
```

Mechanics:

- **Join reuse:** the refit imports the driver as the join code of record (`family_111bank` + the same row-level min-dist join over the 2,800-row tidy snapshot `eval_results/issue_536/inputs/i478_tidy_69b34b94.csv`, gated at 1e-4 against the snapshot's own `min_dist` column; recorded `join_gate_max_dev = 7.15e-07`).
- **Manipulation-check gate** before the centered read is interpretable: the raw-join refit must reproduce the published co-primary read (β = +0.010, p = 0.405) under four criteria — same sign, |β − published| ≤ 0.005 (`GATE_BETA_TOL`), p ≥ 0.10 (`GATE_P_FLOOR`, the published clearly-non-significant regime), and convergence.
- **Pre-registered verdict rule** (α = 0.05 on the single registered interaction term `K:log_md`; the plan-H2 α = 0.01 read is recorded alongside, never binding): `confirmed` — gate passed, centered fit converged, interaction positive and p < 0.05; `killed` — gate passed, centered fit converged, p ≥ 0.05 or wrong sign; `inconclusive` — either fit singular/non-convergent or the raw-join gate failed (reported, never papered over; no estimator fallback is permitted — the published code's OLS fallback is a different estimator and not a valid substitute).
- **Honest fit diagnostics:** convergence flag, boundary variance components (< 1e-8), captured fit warnings, and an as-coded random-effects structure note: the published call omits `re_formula`, so with `vc_formula` present statsmodels fits NO subset_id random intercept (k_re = 0) — the realized RE structure is the persona variance component nested within subset_id groups + residual, reported as-coded, never imputed.
- **Output writes:** `mixedlm_refit_478.json` (env versions pinned: Python 3.11.15, statsmodels 0.14.6, pandas 2.3.3, numpy 2.2.6), plus a `follow_up_result` block added IN PLACE to the `478-flatness-null` row — `regrade_label` untouched (the label belongs to the registered estimator), row order preserved, atomic tmp-then-replace write. No other table row is touched.

---

## 6. Analysis parameters

No model, no training, no seeds — every load-bearing parameter below was fixed in plan §3/§4 before the recompute ran. All values copied from the scripts at the pinned SHAs and cross-checked against the shipped tables.

| Parameter | Value | Notes |
|---|---|---|
| **Canonical metric** | global-mean-center → L2-normalize → cosine | `compute_cosine_matrix(C, centering="global_mean")`; library default; reused, not reimplemented |
| **Matrix join gate** | `GATE_MATRIX_TOL = 1e-4` | max abs elementwise deviation, recomputed raw vs persisted matrix |
| **Statistic join gate** | `GATE_RHO_TOL = 0.02` | \|Δρ\| vs published; slope rows within 0.02 of published slope / inside published interval |
| **Magnitude boundaries** | 0.5× weakens / 1.5× strengthens | `M_WEAKEN_FACTOR` / `M_STRENGTHEN_FACTOR`; pre-registered conventions on the scale-invariant M |
| **PSD clip threshold** | clipped mass > 1e-3 of trace → `sensitivity-unreliable` | `PSD_CLIP_TRACE_FRAC`; matrix-only Gram reads |
| **Per-row α** | each task's own published threshold; 0.01 default | e.g. 0.01 for #478/#474 rows, 0.05 for the #396/#415 family test, 0.001 for #341 (its own Mantel gate threshold) |
| **Null-rescue multiplicity** | Holm-corrected p < 0.05 within the row's full family | #396/#415 family = 12 cells + \|ρ\| ≥ 0.5 (N=24); #478 flatness family = 2 tests; H2 α=0.01 recorded alongside, non-binding |
| **MixedLM refit gate** | same sign, \|β − 0.010\| ≤ 0.005, p ≥ 0.10, converged | `GATE_BETA_TOL` / `GATE_P_FLOOR`; manipulation check vs the published raw-join read |
| **MixedLM refit α** | 0.05 (single registered interaction term) | `H2_ALPHA = 0.01` recorded alongside |
| Sweep pass-1 regex | `compute_cosine_matrix\|cosine_similarity\|F\.cosine\|cos_sim\|centering=` | literal-API pass |
| Sweep pass-2 regex | `F\.normalize\|np\.linalg\.norm\|torch\.linalg\.norm\|\.norm\(\|np\.dot\|torch\.dot\|einsum\|@\s*\S*\.T\|matmul\|cosine_matrix\|_cosine\|cos_` | hand-rolled-idiom pass (high recall by design) |
| Sweep roots | `scripts/`, `experiments/`, `src/explore_persona_space/{analysis,axis,experiments}/` | top-level `experiments/` added at fact-check (the #341 producer lives there) |
| Layers read | each task's own published layer | L20 (#66/#405/#478/#490 lines), L21 (#505, #474 lineage), L15 (#396/#415, #380), per-layer where the task swept |
| Seeds | n/a | deterministic linear algebra |
| Env (refit JSON) | Python 3.11.15, statsmodels 0.14.6, pandas 2.3.3, numpy 2.2.6 | recorded in `mixedlm_refit_478.json` |

---

## 7. Worked example — audit-table rows (verbatim)

<!-- cherry-picked for illustration; full table at the SHA-pinned links in section 8 -->

A **SITE row** — the row encoding the consumer-read rule (the script computed both matrices; classification follows what the downstream artifact actually stored):

```json
{
  "file": "scripts/compare_extraction_methods.py",
  "lines": "~412 (cross-method), ~431 (raw matrix), ~459 (centered matrix)",
  "classification": "both-computed — CONSUMER READ RAW",
  "evidence": "cosine_matrix_a_layer20.json off-diag [0.7322, 0.9971] median 0.9450 (raw fingerprint, no centering key); the centered twin was computed but not consumed",
  "tasks": [405, 478, 490],
  "notes": "#405's distance source; #478/#490 use the 111-bank file (also raw)."
}
```

A **DISPOSITION row** — a consumer file accounted for without a SITE classification, with its matched-idiom evidence lines:

```json
{
  "file": "scripts/i474_cosine_followup.py",
  "disposition": "consumer — reads a RAW-line persisted artifact; see the producer's site row (#406 C_L*.json)",
  "matched_passes": {"pass1": false, "pass2": true},
  "evidence_lines": [
    "101 [pass2]: dg_a, g_a, ln_a, js_a, cos_a = _vecs(G, pairs_all)",
    "119 [pass2]: \"all_rho_cosL21_deltag\": _length_partial(cos_a[21], dg_a, ln_a)[0],",
    "..."
  ]
}
```

A **FINGERPRINT-CHECK row** — the off-diagonal regime read on a persisted artifact:

```json
{
  "artifact": "eval_results/extraction_method_comparison/cosine_matrix_a_layer20.json",
  "consumed_by": "#405 (issue405_clean_result_analysis.py:46-50)",
  "kind": "similarity matrix (20 personas, L20)",
  "min": 0.7321776151657104,
  "median": 0.9449904263019562,
  "max": 0.9970542788505554,
  "regime_call": "raw band (compressed)",
  "verdict": "RAW — the consumer read the raw matrix even though compare_extraction_methods.py computed both"
}
```

---

## 8. Worked example — re-grade-table rows (verbatim, fact-only)

<!-- cherry-picked for illustration; re-grade labels and interpretive notes fields are elided here — they are the task's findings and live in the task body. Full table at the SHA-pinned link in section 9. -->

A **verification row** (`verify_centered` slug) — schema + gate fields from `66-verify`, with one per-source cell showing the statistic-gate read:

```jsonc
{
  "row_id": "66-verify",
  "task": 66,
  "config_slug": "verify_centered",
  "cosine_path_used": "centered (canonical) — analyze_100_persona_cosine.py:288-293",
  "recoverability": "CPU (verified)",
  "family": "single_token_100p_L20",
  "bank": {"n": 111, "names_hash": "334f4c339c12", "layer": 20},
  "gate_level": "statistic (|drho|<=0.02 vs published per-source rho; bank matrix gate also passed at 1e-4 vs the cached raw-distance JSON)",
  "recomputed_stat": {
    "per_source": {
      "villain": {
        "n_pairs": 110,
        "published_rho_centered": 0.704,
        "recomputed_rho_centered": 0.7039587215377767,
        "sensitivity_rho_raw_recipe": 0.6645825315651223,
        "gate_pass": true
      }
      // ... 4 more sources, pooled reads, p-values
    }
  },
  "raw_vs_centered_x_spearman": 0.8055216070213692,
  "n": {"per_source": 110, "pooled": 550}
  // regrade_label + notes elided — see the task body
}
```

A **re-grade row** (`regrade_raw` slug) — `478-perK-slopes`, showing the three stacked gate levels and the descriptive-only raw-slope convention (one of four K cells shown):

```jsonc
{
  "row_id": "478-perK-slopes",
  "task": 478,
  "config_slug": "regrade_raw",
  "cosine_path_used": "RAW — cosine_distance_matrix_layer20.json (scripts/_issue478_common.py@69b34b94, normalize-only)",
  "recoverability": "CPU (verified; tidy.csv snapshotted from 69b34b94)",
  "family": "single_token_100p_L20",
  "bank": {"n": 111, "names_hash": "334f4c339c12", "layer": 20},
  "gate_level": "matrix-1e-4 (bank) + row-level min_dist 1e-4 + statistic (per-K raw OLS slopes within 0.02 of published)",
  "alpha": 0.01,
  "original_stat": {
    "estimator": "per-K OLS slope of deltaLogP vs log(min_dist)",
    "per_K_beta": {"1": -1.37, "2": -1.41, "4": -1.35, "8": -1.32},
    "note": "each published p < 1e-100"
  },
  "recomputed_stat": {
    "raw":      {"1": {"n": 1120, "beta": -1.3680889584557097, "std_beta": -0.7122527170279376,
                       "spearman_rho": -0.7389891105951706 /* , p-values */ }
                 /* , K = 2, 4, 8 */ },
    "centered": {"1": {"n": 1120, "beta": -1.5506729371769123, "std_beta": -0.6009943499506185,
                       "spearman_rho": -0.69899420125762 /* , p-values */ }
                 /* , K = 2, 4, 8 */ },
    "note_on_scale": "raw-vs-centered beta deltas are descriptive only; M = per-K Spearman rho + standardized beta"
  },
  "n": {"rows": 2800, "per_K": {"1": 1120, "2": 560, "4": 560, "8": 560}}
  // regrade_label elided — see the task body
}
```

A **partition row** (`approx_gram` slug, no join possible) — `227-partition`, complete, including the recoverability-evidence note (the partition label is a recoverability disposition, pre-registered in the plan, not a scientific finding):

```json
{
  "row_id": "227-partition",
  "task": 227,
  "config_slug": "approx_gram",
  "cosine_path_used": "RAW — run_issue_213_part_a.py:547",
  "recoverability": "needs-GPU-re-extraction / unrecoverable from disk",
  "gate_level": "n/a (no join possible)",
  "original_stat": {
    "estimator": "cue->no_cue centroid cosine distances per layer/model",
    "schema": "persisted file is {layer: {model: {cue: distance-to-no_cue}}} over models=['educational-insecure', 'insecure', 'secure-finetune', 'base-instruct'] cues=['no_cue', 'edu_v0', 'edu_v1', 'edu_v2', 'edu_v3', 'code_format'] — a single reference column, no Gram matrix (fact-check verified)"
  },
  "recomputed_stat": null,
  "regrade_label": "join-failed (needs-GPU-re-extraction)",
  "notes": "Consumer of the #213 part-A geometry. WandB-artifact re-search (LIVE, 2026-06-10, wandb.Api over entity thomasjiralerspong): 175 projects scanned for 213/227/cue/insecure/divergence/geometry keywords; 1 candidate project (issue_157_geometry_leakage, 2 artifacts) with ZERO centroid-like artifacts — no #213/#227 centroid bundle exists on WandB. Follow-up pointer: single eval-intent pod, ~1-2 GPU-h, base-model centroid re-extraction for the cue-variant bank, then rerun this driver's adapter.",
  "computed_at": "2026-06-11T02:20:17.534420+00:00"
}
```

---

## 9. Artifacts and reproducibility

- **Code commits (all 40-char, verified via `git rev-parse`):**
  - Audit + recompute driver (shipped state, branch `issue-536`): `09e7590523b92700c3da92b3f672be7bb5869f76` — [scripts/issue536_audit.py](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/scripts/issue536_audit.py) · [scripts/issue536_recompute_driver.py](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/scripts/issue536_recompute_driver.py)
  - Commit where the shipped tables were generated (audit + driver scripts unchanged since): `affb2615539ee5c34f79015b97621424a4acf780`
  - Figures script: `35e98e4ea3938adf63a67791c914f4b9faeab9e1` — [scripts/issue536_figures.py](https://github.com/superkaiba/explore-persona-space/blob/35e98e4ea3938adf63a67791c914f4b9faeab9e1/scripts/issue536_figures.py) (carries two post-generation render fixes — explicit marker edge widths + an encoding legend, then descriptive reader-facing labels — that change no plotted value)
  - MixedLM refit: `12853bca8b90a3b7c0591abfff1f3210cd666eac` — [scripts/issue536_mixedlm_refit.py](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/scripts/issue536_mixedlm_refit.py)
  - Provenance correction (recorded in the task's Reproducibility section): the committed tables internally stamp `code_commit = bfa36c3fee6e74674e9be08bb2469981ca733bc6` (the HEAD at generation time, before the final round-2 code commit was created from the same working tree); the authoritative code SHA is `affb26155…` for the shipped tables and `09e759052…` for the shipped figures. The audit and driver tables also record independent `data_root_commit` values because main's HEAD moved between the two phases (concurrent sessions); the consumed artifacts are stable committed files, but a strict reproducer should run both phases against one pinned checkout.
- **Output artifacts:**
  - Audit table: [eval_results/issue_536/audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/eval_results/issue_536/audit_table.json)
  - Re-grade table (30 rows): [eval_results/issue_536/regrade_table.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/regrade_table.json)
  - MixedLM refit: [eval_results/issue_536/mixedlm_refit_478.json](https://github.com/superkaiba/explore-persona-space/blob/12853bca8b90a3b7c0591abfff1f3210cd666eac/eval_results/issue_536/mixedlm_refit_478.json)
  - Figures payload (every plotted float): [eval_results/issue_536/figures_payload.json](https://github.com/superkaiba/explore-persona-space/blob/35e98e4ea3938adf63a67791c914f4b9faeab9e1/eval_results/issue_536/figures_payload.json) · figures: [figures/issue_536/](https://github.com/superkaiba/explore-persona-space/tree/35e98e4ea3938adf63a67791c914f4b9faeab9e1/figures/issue_536)
  - The metric pin: [.claude/rules/persona-distance-metrics.md § "Bank centering — canonical (task #536)"](https://github.com/superkaiba/explore-persona-space/blob/09e7590523b92700c3da92b3f672be7bb5869f76/.claude/rules/persona-distance-metrics.md)
- **Pinned input snapshots** (artifacts that live only on other branches, copied under the task's own tree): [eval_results/issue_536/inputs/](https://github.com/superkaiba/explore-persona-space/tree/09e7590523b92700c3da92b3f672be7bb5869f76/eval_results/issue_536/inputs) — `i478_tidy_69b34b94.csv` (the 2,800-row per-cell table, snapshotted from commit `69b34b94e00cf4c16b830f32da605289ef35371c`) and `i341_js_matrix_4ddf33d6.json` + `i341_geometry_alignment_4ddf33d6.json` (snapshotted from commit `4ddf33d6dc2d0c3fc86637b4a856db595a2bfd1d`).
- **Reused input artifacts** (this audit's inputs ARE prior tasks' persisted measurement artifacts; fitness = identity, enforced by the join gates): the #66 111-persona centroid bank (`eval_results/single_token_100_persona/centroids/centroids_layer{10,15,20,25}.pt`, local); the extraction-method bank (`eval_results/extraction_method_comparison/centroids_method_a.pt`, local); the #274 24-persona bundle (`eval_results/issue_274/centroids/centroids_n24_layers0_27.pt`, local); HF data-repo bundles `superkaiba1/explore-persona-space-data`: `issue505_loo_contrastive/geometry/centroids_pv_L{7,14,21,27}.pt` and `issue142_single_token/centroids/{centroids_layer20.pt, persona_names.json}` (resolved via the Python Hub API at write time); the #311 bundle (`eval_results/issue_311/centroids_base.pt`, local); and the matrix-only persisted matrices from #406 (`eval_results/issue_406/cosine/C_L*.json`) and #341.
- **Deleted-producer evidence commits:** `9e6e31c3f9222cf6d1dc95c31b0ae19d6230a58d` (#406 producer `i406_phase1_merge_and_compute_matrices.py`) and `69b34b94e00cf4c16b830f32da605289ef35371c` (#478 loader `_issue478_common.py`), both read via `git show`.
- **WandB:** no runs created (no training); a live `wandb.Api` artifact re-search over entity `thomasjiralerspong` (175 projects, 2026-06-10) backs the #213/#227 partition rows.
- **Compute:** CPU only, VM-side; the full audit + recompute + figures pipeline runs end-to-end in ~27 s. No pod provisioned. GPU-hours: 0.
- **Reproduce:**

```bash
cd explore-persona-space   # at 09e759052
uv run python scripts/issue536_audit.py --data-root "$PWD"
uv run python scripts/issue536_recompute_driver.py --data-root "$PWD"
uv run python scripts/issue536_figures.py                            # at 35e98e4ea
uv run python scripts/issue536_mixedlm_refit.py --data-root "$PWD"   # at 12853bca8
```

---

*This document describes how the audit was run. For the result and what it means, see the [task body](https://eps.superkaiba.com/tasks/536).*

---
title: The layer-19 context-answer map reaches its population linear ceiling with
  reproducible singular structure, contracts only at layer 26, and no gate metric
  separates from its panel (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-25T06:45:51Z'
has_clean_result: false
parent_id: 1774
origin_prompt: 'User-approved 8-leg battery from the 2026-08-24 one-at-a-time theory
  walk: eigen/fixed-point analysis, gate-metric ladder (''ok this works''), wiring
  matrix (''sounds good''), last-prompt-token SAE + feature-to-feature map, weight-update
  rank (''let''s look at the rank of the update''), denoised shift regression, operator
  atlas + cross-model three-tier (''Sounds good''), null-space fibers + monitor certificates
  (''yes add it''). Filing batch answers: one umbrella task child of #1774, spawn
  one autonomous session, Llama-3.1-8B-Instruct for the atlas capture, no Overleaf
  writes.'
workflow: v1
goal: 'Exploit the linearity of the fitted residual context-answer mapping (banked
  963k-row layer-19 ridge map, Qwen2.5-7B-Instruct) to characterize in one phased
  battery: (1) operator eigenstructure, effective null space, fixed point, and their
  SAE-feature reads (ignored/copied/transcoded/damped anatomy); (2) whether the map''s
  Gram matrix WtW is the correct theory-paper context-gate metric and whether the
  closed-form ridge learning curve reproduces the empirical C1 scaling; (3) the feature-to-feature
  wiring of the map across SAE dictionaries including a trained last-prompt-token
  SAE; (4) the literal weight-update rank (dW spectra, intruder dimensions, factor
  alignment) across the recent organism fleet; (5) denoised reduced-rank structure
  of #1979''s banked write maps; (6) an aligned operator atlas with a cross-model
  (Llama-3.1-8B-Instruct) three-tier representation-vs-operator similarity report;
  (7) correlational validation of the effective null space via kernel-equivalent context
  pairs, plus closed-form monitor decision geometry and sensitivity certificates.'
relates_to:
- spec-context-as-vector
- leak-predictor
- identity-contextual-vs-base
---
# The layer-19 context-answer map reaches its population linear ceiling with reproducible singular structure, contracts only at layer 26, and no gate metric separates from its panel (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- Held-out R² 0.7194 at n=500,000 against a 0.7263 population linear value (99.1%); the closed-form ridge curve predicts every point to within 0.0058 R².
- ρ(A) = 1.6607 / 1.2054 / 0.9231 at L14/L19/L26 — the registered contraction clause holds only at L26, the one layer where iterating the map converges.
- LoRA's top weight-update directions intrude on the base column space in 42 of 83 cells, concentrated in q/k/v_proj; full fine-tunes intrude in 0 of 28.
- No gate metric separates: ΔWΔWᵀ beats the whitened gate in 5 of 12 content arms (7 needed) and identity in 9 of 12; best p_win 0.396.
- Kernel-direction context pairs move answers 0.607× as far as matched controls (95% CI 0.602–0.615) — 1.23× the residual floor, against controls' 2.01×.
- Coverage: the wiring gate and two of leg-1's four clauses at L14/L26 were never evaluable — the row battery supplies rows and moments at layer 19 only.

## Goal

- **This experiment in context:** eight legs interrogate ONE banked object — the 963,444-row layer-19 ridge map from context summary v_C to answer summary v_A on Qwen2.5-7B-Instruct, fit in [#779](https://eps.superkaiba.com/tasks/779) and re-derived here. Parent [#1774](https://eps.superkaiba.com/tasks/1774) established that the map is linear; this battery spends that linearity: operator eigenstructure and fixed point (leg 1), the theory-bridge gate ladder (leg 2), SAE wiring (leg 3), a context-feature→answer-feature map over the banked answer SAE (leg 4), weight-update rank across the arm fleet (leg 5), denoised shift regression (leg 6), a cross-model operator atlas against Llama-3.1-8B-Instruct (leg 7), and an effective-kernel pair test (leg 8). Banked maps come from [#2094](https://eps.superkaiba.com/tasks/2094) and [#2379](https://eps.superkaiba.com/tasks/2379); the answer SAE and its per-feature instruments from [#2476](https://eps.superkaiba.com/tasks/2476); the adapter fleet from [#1434](https://eps.superkaiba.com/tasks/1434); the alignment anchor from [#825](https://eps.superkaiba.com/tasks/825); operator-comparison conventions from [#1345](https://eps.superkaiba.com/tasks/1345); feature descriptions from [#2552](https://eps.superkaiba.com/tasks/2552).
- **Broader narrative:** if a linear map carries context into answers, its operator algebra is the analyzable object — what it ignores, what it contracts toward, and whether that structure survives across corpora, arms, and model families. This battery tests whether the map's linearity is a convenience or a description.

## Methodology

- **Design:** six phases (P-A…P-F) over one banked map. No new model training except a context-side SAE. Layers L14/L19/L26 for Qwen operator reads; L19 is the map layer and the only layer with row-battery moments. Eight legs, 19 of 21 registered figures rendered. `production: true` and `regime_config_hash 35003bfda5ef3328` on every artifact.
- **Training:** only the context SAE was fit. Complete table:

| Parameter | Value | Source |
|---|---|---|
| Base model | Qwen/Qwen2.5-7B-Instruct | banked map recipe |
| Cross-model | meta-llama/Llama-3.1-8B-Instruct | decision record 2026-08-24 |
| Map layer | 19 (`layer_c`) | banked map recipe |
| Operator layers | 14, 19, 26 | plan §4 leg 1 |
| Llama layers | 16, 22, 30 | plan §4 leg 7 |
| d (residual width) | 3584 Qwen / 4096 Llama | model config |
| Ridge λ selection | val-selected, widened 27-value 1e-5…1e8 grid, widen-on-edge | fit-core default (no bare GCV) |
| Map n_train / n_val / n_test | 963,444 / 400 / 1,000 | `assemble/split_meta.json` |
| Context SAE architecture | matryoshka_batchtopk, dict 65,536, k=100, tiers 2048/16384/65536 | answer-SAE recipe, inherited |
| Context SAE lr / betas / batch | 2e-4 / (0.9, 0.999) / 256 | answer-SAE recipe, inherited |
| Context SAE epochs / steps | 3 / 10,941 | `sae_ctx/config.json` |
| Context SAE pools | 933,444 train / 10,000 val | `sae_ctx/config.json` |
| Context SAE threshold (EMA 0.999) | 0.6178 | `sae_ctx/config.json` |
| Context SAE fitness gate | FVE 0.9030, L0 103.3, floor 0.5 → PASS | `sae_ctx/gates_sae_ctx.json` |
| Judge model | claude-sonnet-4-5-20250929 | CLAUDE.md standing rule |
| Judge max_tokens / temperature | 1024 / 0.0 | llm-judging rules 23/26 |
| Permutation / bootstrap draws | 10,000 / 10,000 | plan §6 |
| Null draws (intruder, alignment) | 200 | max-matched null convention |
| Split-half null draws | 1,000 (seed 2,569,910) | plan §7.5 |
| Seeds | 2569 (ladder), 2,569,800 (leg 8), 2,569,400 (matching), 0 (split) | plan §11 |
| Capture batch tokens / max rows | 8192 / 64 | `pd/chunks_regime_*.json` |
| Paired cross-model rows | 60,000 (53,581 train / 512 val / 5,907 test) | `xmodel/fits_summary.json` |

- **Evaluation:** leg 1 reports a registered 4-clause criterion (ρ(A) < 1; κ(V); ≥300 stable singular directions; copied-class data-variance share < 0.2). Leg 2 races six gate metrics per arm by Spearman ρ against a 10,000-draw permutation band, with selection-inherited and frozen-at-winner CIs reported separately. Leg 4 reports per-feature held-out R², a firing/magnitude hurdle split, PR@k at the realized L0 (k=96), kNN retrieval (euclidean + cosine, chance k/20,000), an index-aligned null and a train-mean null; identity-family baselines are marked `applicable: false` where input and output dimensions differ. Leg 5 scores each top ΔW singular vector by max |cos| to the base singular basis against a max-matched order-statistic null — at or below the null p95 the direction is an intruder, above it the direction already exists in the base column space. Leg 7 reports CKA, alignment-fit R² against a split-half reliability floor, and Procrustes-ALIGNED (direction-aware) operator cosine against a Haar rotation null, with rotation-invariant spectrum cosine labelled separately. Leg 8 reports the median of paired |Δv_A| ratios with a clustered bootstrap (800 clusters) and reads it against a measured held-out residual-pair floor.
- **Data extraction:** rows are a banked sampling manifest of real LMSYS + WildChat conversations (data-realism tier 1). Measured at phase entry: 964,844 rows present, 525,485 LMSYS and 434,359 WildChat, 5,000 pass-B seed rows. Cross-model capture teacher-forces the SAME Qwen answer text under both models, so leg 7 claims describe transformations of shared Qwen responses and NOTHING about Llama's own answer policy. The 20,000-row holdout (10,983 LMSYS / 9,017 WildChat) is sha-pinned; corpus is the fold group for every transfer read.
- **Sample training/evaluation data + completions:** this battery scores activation summaries and SAE feature vectors, not generations, so there are no model completions to sample — the only judged objects are SAE feature descriptions (10-way matching, 500 items), reused from a banked description set (`descriptions_mat_k100.json`, sha256 `fbb079fa86b66183ea643cebcad42f19bcfe5924e739b9c9cf72c0dfc4d9b509`) covering 2,149 of the 2,150 union features. Complete artifact set: [eval_results/issue_2569 at `c5daf754fe`](https://github.com/superkaiba/explore-persona-space/tree/c5daf754feedf8a0751ab9e991d1261cf8965aff/eval_results/issue_2569) (671 committed text files) and [issue2569_theory/analysis_tensors at `d3ab70c673`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors) (862 files). The two blocks below are NOT cherry-picked: each is the COMPLETE registered record for its leg, quoted verbatim in full.

<details>
<summary>Complete leg-1 criterion record at L19 — not cherry-picked; the full L14/L19/L26 set is at <a href="https://github.com/superkaiba/explore-persona-space/tree/c5daf754feedf8a0751ab9e991d1261cf8965aff/eval_results/issue_2569/weights/leg1">eval_results/issue_2569/weights/leg1 at <code>c5daf754fe</code></a></summary>

```json
{"verdict": "FAIL", "n_evaluated": 4, "n_clauses": 4, "n_failed": 1,
 "rho_contraction":   {"value": 1.2053517865293906, "pass": false},
 "kappa_nonnormal":   {"value": 4260.870580356614,  "pass": true},
 "stable_directions": {"value": 3425, "threshold": 300, "pass": true},
 "copied_data_share": {"value": 0.0,  "threshold": 0.2, "pass": true}}
```

</details>

The second block below is likewise the COMPLETE record, not a cherry-picked excerpt: leg 8 ran exactly one kernel-pair mining pass, and its whole summary is quoted verbatim from `eval_results/issue_2569/leg8/mining_summary.json` — [eval_results/issue_2569/leg8 at `211574ded9`](https://github.com/superkaiba/explore-persona-space/tree/211574ded95c3ec4b0e08ca07b416a37a6e03d38/eval_results/issue_2569/leg8).

<details>
<summary>Complete leg-8 selection + ratio record — not cherry-picked; the only kernel-pair run, quoted in full from <code>eval_results/issue_2569/leg8/mining_summary.json</code></summary>

```json
{"n_sampled": 20000000, "n_eligible": 9999779, "n_kernel_selected": 1000, "n_matched": 1000,
 "median_of_paired_ratios": 0.6072446042464897, "ratio_of_medians_companion": 0.6119308596420819,
 "kernel_dva_median": 32.77625050365498, "control_dva_median": 53.562016014073535,
 "clustered_bootstrap_ci95": [0.6018269695583521, 0.6151250583219257],
 "floor_q50": 26.61424498927945, "kernel_over_floor": 1.2315303521425336,
 "in_sample_share_selected_rows": 0.99875}
```

</details>

## Results

I acknowledge this body ships conciseness WARNs in three classes plus one figure-text WARN. Total-prose/budget: eight legs and 19 figures do not fit the 800-word Takeaways+Goal+Results budget. Takeaways bullets: several exceed the 30-word soft cap. Figure caption length: one caption exceeds 60 words. No per-result block reaches the 180-word FAIL tier. Separately, five rendered figures carry opaque config-code tokens in their axis or legend text (`issue2569_rowbattery`, `n_above_floor`, `copied_dw_share`, `H2b`, `Delta_tbar_even`, `Delta_tbar_odd`); the plain-English readings are in the surrounding prose and captions, and regenerating those figures is filed rather than done here.

### The map is not low-rank, and its kernel tightens with depth

Per layer (L14/L19/L26), the singular spectrum of the fitted operator A: the rank needed for 99% and 90% of squared-singular-value mass, σ_max, σ_median, and the per-direction class labels (ignored / copied / rotated-scaled / transcoded).

![Operator anatomy across three layers: mass-rank, spectrum and direction classes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg1_anatomy_hero.png)

> **Figure.** *Mass-rank rises and σ_max falls with depth.* k99 = 1,325 / 1,608 / 2,000 and k90 = 411 / 547 / 794 at L14/L19/L26; σ_max = 13.17 / 7.96 / 5.14, σ_median = 0.165 / 0.134 / 0.116. Direction classes are assigned by gain and alignment thresholds.

128 of 3,584 directions carry only 0.701 / 0.654 / 0.582 of Frobenius energy, so no low-rank summary is faithful. Deeper layers spread the map over more directions at smaller gain.

### Eigenvalues and singular values tell different stories, and only L26 contracts

Per layer, the eigenvalue spectrum of A against its singular-value spectrum, with the spectral radius ρ(A) marked. ρ decides whether iterating the map converges. Per-unit exemption: the plotted series ARE the per-direction spectra — every eigenvalue and singular value of all three operators is drawn, so this figure is already the low-level view.

![Eigenvalue versus singular-value spectra per layer with the spectral radius marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg1_eigen_vs_singular.png)

> **Figure.** *ρ(A) crosses 1 between L19 and L26.* ρ = 1.6607 / 1.2054 / 0.9231 at L14/L19/L26; κ(V) = 7,650 / 4,261 / 7,452 marks all three operators as strongly non-normal, so singular values overstate per-step growth.

Only L26 satisfies ρ < 1, and only there does the driver mark the fixed point as a valid iterated-map reading. At L14/L19 the fixed point exists algebraically — relative residual 3.6e-14 and 6.2e-15 — but iterating the map diverges from it.

### The L19 fixed point decodes to 10,302 firing context features

The fixed point x* of the L19 map (x* = Ax* + b) pushed through the context SAE encoder: which features fire and at what activation.

![SAE feature dashboard for the layer-19 fixed point](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg1_sae_dashboards.png)

> **Figure.** *The fixed point is dense in feature space.* 10,302 of 65,536 context-SAE features fire at x* (‖x*‖ = 153.4, ‖b‖ = 41.7); top activations 14.5, 13.9, 13.5. Features are unlabelled — both description sources resolved `absent`.

A fixed point firing 16% of the dictionary is not an interpretable attractor state. Nearest-banked-answer neighbours were deferred: no producer phase supplies banked answer rows.

### The closed-form ridge learning curve predicts the realized curve to 0.006 R²

Empirical held-out R² for LMSYS-only refits at n = 4,500 / 10,000 / 50,000 / 150,000 / 500,000 against the closed-form spectral prediction from the context covariance eigenspectrum, plus per-point misfit against the registered bands.

![Learning curve theory versus measured, and per-point misfit against the registered bands](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg2_learning_curve.png)

> **Figure.** *Theory tracks measurement across two decades of n.* Mean |ΔR²| = 0.0058, same sign at all five points, inside the ±0.05 pass band and far from the ±0.15 kill floor. Off-recipe committed points (orange) sit higher because they mix corpora and λ-selection rules.

The realized 0.7194 at n=500,000 is 99.1% of the 0.7263 population linear value. That population value is 1 − residual/total variance of the population linear fit, so it is the linear ceiling by construction — not independent evidence that the remainder is irreducible noise.

### No gate metric separates from its panel, per arm

Per content arm (12 arms, n=50 shared contexts each), Spearman ρ between each of six candidate gate metrics and the change DV, with each arm's 10,000-draw permutation band.

![Per-arm gate-metric race against permutation bands, content arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg2_gate_ladder_content.png)

> **Figure.** *Most arm-metric cells fall inside the permutation band.* Per-arm p95 of the max selected statistic is ≈0.295–0.300; the winning metric's across-arm median ρ is 0.265 (a-whitened) versus 0.258 for the whitened incumbent and 0.196 for identity.

ΔWΔWᵀ wins 5 of 12 arms against the whitened gate (7 needed for the registered success band) and 9 of 12 against identity (kill is ≤5), landing between success and kill. Median ρ 0.2404 is 0.585× the champion reference.

### The marker arms behave like the content arms

The same six-metric race on the marker arms, scored at the map layer L19 — the banked marker-primary layer was L25, and the ladder deliberately pins the map's input space instead.

![Per-arm gate-metric race against permutation bands, marker arms](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg2_gate_ladder_marker.png)

> **Figure.** *No metric separates on the marker arms either.* Same permutation-band construction and same six metrics; the layer mismatch against the banked marker-primary layer is a stated scope caveat, not a corrected confound.

Reading marker arms at L19 rather than their own primary layer is deliberate — the ladder tests the map's input space — and it means these arms are not a fresh test of marker leakage.

### Winner identity flips across context families

Per context family (battery / bystander / conv-fresh / near-twin, 18 arms each; three families skipped for fewer than four kept rows in every arm), the across-arm median ρ for each of the six metrics.

![Across-arm median gate-metric rho by context family](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg2_gate_family_table.png)

> **Figure.** *Different families elect different winners.* battery elects the diagonal-inverse gate (0.467), bystander and conv-fresh the a-whitened gate (0.425, 0.155), near-twin the whitened gate (0.400). Three of seven families had too few kept rows to score.

A metric that wins on one family and loses on the next is not a gate; it is a family-specific correlate. The three skipped families are named here and excluded from every denominator.

### Behaviour-relevant answer features draw a negligible share of in-edge mass

For each behaviour direction's nearest answer feature, the share of top-32 in-edge |edge| mass over context columns, and the union distribution across all 2,150 answer features.

![Top-32 in-edge mass share for behaviour-relevant answer features](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg3_wiring_edge_mass.png)

> **Figure.** *Full-column shares are ~0.2%, three orders below the 0.5 pass bar.* evil 0.00300, sycophancy 0.00189, hallucination 0.00143; union median 0.00169. Nearest-feature cosines are 0.493 / 0.515 / 0.249.

These are the INFORMATIONAL full-column shares, not the verdict-grade statistic: the alive-masked share needs the assemble rows, which were never attached, so the wiring gate has no verdict in either direction.

### The feature map predicts which features fire, not how much

Per-feature held-out R² on 20,000 pinned holdout rows for five routes: the fitted context-feature→answer-feature map, two banked per-feature instruments on their 879-feature intersection, an index-aligned null, and a train-mean null.

![Per-feature R2 by route, with index-aligned and train-mean nulls](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg4_routes.png)

> **Figure.** *The fitted map clears both nulls; the banked composed route scores higher on its own narrower panel.* Fitted median R² 0.2523 (99.2% of features positive); index-aligned null −0.0265, train-mean null −2.7e-5; banked composed 0.5992 and dense-input 0.5119 on the 879-feature intersection.

The hurdle split is the finding: firing AUROC median 0.9363 but conditional-magnitude R² median −0.8577. The map says which features switch on and fails at their amplitude. Identity baselines are inapplicable (different dictionaries); kNN retrieval gives acc@1 0.627 cosine / 0.554 euclidean against 5e-5 chance.

### Per-feature spread, and the alive-floor sweep

The same fitted route resolved per feature: R² against firing rate, with the four alive-floor settings (1% / 0.5% / 0.25% / 0.2%) overlaid.

![Per-feature R2 versus firing rate across four alive-floor settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg4_per_feature.png)

> **Figure.** *Median R² rises as the floor loosens.* 0.2523 → 0.2760 → 0.3100 → 0.3204 as the alive context width grows 1,276 → 1,686 → 2,311 → 2,545; selected λ rises 316 → 1,000 → 1,000 → 3,162 with no grid-edge hits.

The trend is monotone and modest. Corpus transfer is the harder read: an LMSYS-only fit scores −0.238 median R² on WildChat holdout rows where the mixed fit scores 0.190, so the map does not transfer across corpora at feature grain.

### A judge picks the right answer from predicted feature descriptions 92.6% of the time

10-way matching: for each of 500 holdout rows, the judge sees the top-8 predicted answer-feature descriptions and picks the true answer among 10 candidates.

![Ten-way description-matching accuracy against chance](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg4_der_matching.png)

> **Figure.** *463 of 500 correct against 0.100 chance.* Judge claude-sonnet-4-5-20250929, max_tokens 1024, temperature 0; malformed / refusal / out-of-range returns dropped, none occurred. Descriptions cover 2,149 of 2,150 union features.

The predicted feature set is semantically informative even where its magnitudes are wrong — consistent with the hurdle split above. Descriptions are reused from a task whose classification is still pending, so this read inherits that instrument's risk.

### Full fine-tunes stay in the base column space; LoRA does not

Per (arm, module, side) cell, the observed band-max |cos| between the top ΔW singular vector and the base singular basis, against that cell's max-matched null p95 (200 draws). At or below the diagonal the direction is an intruder.

![Intruder read versus the max-matched null, all arms and modules](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg5_dw_intruder.png)

> **Figure.** *Full fine-tunes intrude in 0 of 28 cells; LoRA in 42 of 83.* Full-FT median band-max 0.310 against a null p95 of 0.082; LoRA median 0.080. LoRA intruders localize to q_proj (13 of 14), v_proj (13 of 14) and k_proj (12 of 14); down_proj and o_proj intrude in none.

This reproduces the LoRA-intruder-dimension result and localizes it: LoRA's out-of-basis directions are attention-side, while its MLP down-projection and attention-output updates stay inside the base basis. Point labels overplot badly in the lower band — a plotting defect filed, not a data defect.

### The updates are low effective rank

Per (arm, module, layer) cell, stable rank and participation ratio of the ΔW spectrum for the 14 LoRA arms (16 or 32 nonzero singular values per cell, 2,324 cells).

![Effective rank of the weight update per module and layer](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg5_dw_effective_rank.png)

> **Figure.** *Stable rank concentrates near 3.* Median stable rank 2.96 (range 1.04–12.33), median participation ratio 6.37 (range 1.09–26.26), against 16–32 available nonzero singular values per cell.

The updates use a fraction of the rank the adapter allocates. This is a property of the realized update, not a bound on what the parameterization permits.

### Read directions align with the update; mapped directions do not

Per (arm, module, direction) cell, max |cos| between the direction and the update's top-8 factor basis, against a 200-draw max-matched null; 338 cells over 18 arms.

![Direction-to-update alignment by direction family against the null](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg5_dw_alignment.png)

> **Figure.** *Raw persona read directions clear the null; their map-transported versions do not.* Above-null counts: read sycophancy 25 of 31, hallucination 19 of 31, evil 11 of 31; mapped sycophancy 3 of 31, evil 1 of 31, hallucination 0 of 31. Context direction 9 of 83; shift direction 5 of 23.

The asymmetry is the result: the weight update overlaps the raw persona read directions far more than the map-transported versions of the same directions. Both direction families are 20-row estimates (`low_n_flag: true`), so this is a coarse read.

### No shared low-rank factor survives the null

Per (arm, context-summary) unit, the denoised rank of shared factor structure between the context basis and the answer-shift basis: matched factor pairs above a 0.5 cosine floor, minus what the null accounts for. 27 pooled units over 9 arms.

![Denoised shared-factor rank across 27 pooled units](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg6_denoised_rank.png)

> **Figure.** *Denoised rank is 0 in 23 of 27 units and 1 in 4.* Raw matched-factor share median 12 of 32 pairs at a 0.5 cosine floor; the null absorbs essentially all of it. Units span three context summaries (last_ctx / last_prompt / span_mean).

The raw matching looks substantial and survives nothing. Leg 6 covers 9 of the 12 planned arms — the 3 full-fine-tune arms failed the runnability filter — so the denominator is 9, not 12.

### Split halves agree on the top factor and disagree below it

Per unit, factor-wise cosine between independent data halves: the top matched factor and the median across all matched factors.

![Split-half factor agreement, top factor versus median](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg6_half_spectra.png)

> **Figure.** *Top-factor cosine 0.982, median 0.137.* Halves are 481,722 rows each from a 963,444-row pool; σ_half_max 8.11. At L19 the operator's own split-half read puts 3,425 of 3,584 singular directions (95.6%) above the stability floor.

The leading direction is reproducible and the tail is not, which is what makes the leg-6 null result credible rather than a power failure.

### The two models are alignable, and their operators are similar but not the same

Working pair Qwen L14 ↔ Llama L16 (selected by max mean validation R² over four alignment fits on TRAIN only): CKA between paired summaries, alignment-fit held-out R² against a split-half reliability floor, and Procrustes-aligned operator cosine against a Haar rotation null.

![Three-tier cross-model read: alignability, operator similarity, diagnostics](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg7_three_tier.png)

> **Figure.** *Alignment clears its reliability floor; operator similarity sits below the within-model anchor.* CKA 0.912 (answers) / 0.755 (contexts); alignment test R² 0.875 against a 0.716 split-half floor. Aligned operator cosine 0.366 and 0.475 against a within-model anchor of 0.686; rotation-null p97.5 = 5.3e-4.

The statistic is direction-aware (Procrustes-aligned, not spectrum-only), so it can speak to operator identity: the two operators are far from rotation-null (z ≈ 1,311) but clearly not the same operator. Tier-3 diagnostics are labelled non-identifying and apportion nothing.

### The atlas separates write maps from read maps

Pairwise aligned-cosine distance among 19 map rows (3 banked n1m, 3 pass-B, 4 cross-model, 1 feature map, 10 leg-6 write maps), with a 2-D MDS embedding.

![Operator atlas distance matrix and MDS embedding over 19 maps](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg7_atlas.png)

> **Figure.** *The 10 write maps form a tight block at distance ≈1.0 from the read-map block.* MDS agrees. The noise-dominated hypothesis is not upheld: within-operator split-half distance exceeds between-operator distance in 32 of 171 pairs, not a majority. One roster row was dropped as a declared soft dependency.

The atlas records both statistic classes, which matters: two banked read maps at different layers have raw cosine 0.153 but spectrum cosine 0.997 — same spectra, different directions. Two presentation defects (colorbar label overrun, MDS label overplotting) are filed.

### Kernel-direction context differences move answers less, but not to the floor

For 1,000 context pairs whose difference lies in the map's low-gain effective kernel, the median ratio of |Δv_A| to that of matched control pairs, read against a measured residual-pair floor.

![Kernel versus matched-control answer displacement, with the residual floor](https://raw.githubusercontent.com/superkaiba/explore-persona-space/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569/leg8_kernel_pairs.png)

> **Figure.** *Ratio 0.607, CI 0.602–0.615.* 1,000 pairs mined from 20M sampled, 9,999,779 eligible; matching tolerance 0.02; clustered bootstrap 10,000 draws over 800 clusters; Wilcoxon p = 1.2e-164. Kernel median |Δv_A| 32.8 versus control 53.6, floor q50 26.6.

Against the floor the reduction is larger than the raw ratio suggests: kernel pairs sit at 1.23× the floor, controls at 2.01×.

The binding caveat is that 99.875% of selected rows lie inside the map's own fit. This is consistency of realized answers with the fitted kernel, never validation on held-out data.

---

**Repro:** P-A/P-B/P-C/P-D ran on `pod-2569` and the fellows SLURM lane; P-E (cross-model fits, report, atlas) ran on `pod-2569-pe` — a 1× H100 CPU-residual route taken after four `cpu-bigmem` capacity misses, a recorded deviation — with fits wall 1,447 s and atlas peak RSS 16.03 GB, terminated after upload-verification PASS; P-F and the leg-1 deferral closure ran VM-side on CPU. ~16 GPU-h projected. Run code SHA `c522e28fa41d289d217fd8584f310d34a77137f7`. Eval JSONs committed at [`211574ded9`](https://github.com/superkaiba/explore-persona-space/tree/211574ded95c3ec4b0e08ca07b416a37a6e03d38/eval_results/issue_2569) (671 text files, 44.0 MB), leg-7 harvest at [`0ab2f1ae4a`](https://github.com/superkaiba/explore-persona-space/tree/0ab2f1ae4a451bf974690d65a9dae9e01910762e/eval_results/issue_2569/leg7), figures + the pass-B normalization recipe at [`432da6f0c2`](https://github.com/superkaiba/explore-persona-space/tree/432da6f0c2257f72f14ab23f801d2d4df259e3fe/figures/issue_2569), leg-1 deferral closure at [`c5daf754fe`](https://github.com/superkaiba/explore-persona-space/tree/c5daf754feedf8a0751ab9e991d1261cf8965aff/eval_results/issue_2569/weights/leg1). Stores, Hub-verified by per-file byte parity at write time: [issue2569_theory/analysis_tensors at `d3ab70c673`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors) — 862 files, including [weights/leg1](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors/weights/leg1) (32 files), [moments](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors/moments) (gram + split-half maps, 1.03 GB), [xmodel](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors/xmodel) (19 files, 8.59 GB) and [leg7](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2569_theory/analysis_tensors/leg7) (43 files, of which 40 are atlas checkpoints). 147 tensors (14.2 GB) stay Hub-only under the JSON-only rule for `eval_results/`. Reused artifacts, each pinned: banked n1m ridge maps from [#2094](https://eps.superkaiba.com/tasks/2094) at `data/issue_2094/joint_transport/banked_maps/issue779_monitoring/n1m_readout/weights/L{14,19,26}/ridge.pt` — uncommitted local artifacts (tensor class), each sha256-verified against the pod copy before use; pass-B pinned maps from [#2379](https://eps.superkaiba.com/tasks/2379) at [issue2379_reelicit/analysis_tensors/maps_pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2379_reelicit/analysis_tensors/maps_pinned), normalized in place by `scripts/issue2569_normalize_passb.py` (three metadata keys added; numerics bit-identical by sha256 over W/xmu/xsd/ymu); the answer SAE and two banked per-feature instruments from [#2476](https://eps.superkaiba.com/tasks/2476) at [issue2476_turnavg/analysis_tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/d3ab70c673f898870600147a311aacca19ddcfbf/issue2476_turnavg/analysis_tensors); the adapter fleet from [#1434](https://eps.superkaiba.com/tasks/1434) (14 LoRA + 4 full-FT arms, resolved per-arm from each `adapter_config.json`); feature descriptions from [#2552](https://eps.superkaiba.com/tasks/2552) (`descriptions_mat_k100.json`, sha256 `fbb079fa86…`); and the Procrustes alignment-null construction from [#825](https://eps.superkaiba.com/tasks/825). Drivers: `scripts/issue2569_{weights,gateladder,atlas,operator,leg6,dw_fleet,xmodel_capture,figures,normalize_passb}.py`.

**Context:** created 2026-08-25 from the verbatim originating prompt

> User-approved 8-leg battery from the 2026-08-24 one-at-a-time theory walk: eigen/fixed-point analysis, gate-metric ladder ('ok this works'), wiring matrix ('sounds good'), last-prompt-token SAE + feature-to-feature map, weight-update rank ('let's look at the rank of the update'), denoised shift regression, operator atlas + cross-model three-tier ('Sounds good'), null-space fibers + monitor certificates ('yes add it'). Filing batch answers: one umbrella task child of #1774, spawn one autonomous session, Llama-3.1-8B-Instruct for the atlas capture, no Overleaf writes.

Parent [#1774](https://eps.superkaiba.com/tasks/1774) — this battery spends the linearity that task established. Plan v4 approved 2026-08-25; run 2026-08-25 → 2026-08-27; the leg-1 row-battery-blocked deferral closure folded 2026-08-27. Autonomous session; no follow-up-scope rounds were posted.

---
title: Linear decoding leaves most of the extractable sycophancy information in base-model
  activations unread (MODERATE confidence)
kind: experiment
tags:
- leak-predictor
- keep-running
created_at: '2026-06-30T00:23:55Z'
has_clean_result: true
parent_id: 658
origin_prompt: 'Run this: Single staged kind: experiment task, 0-GPU, on existing
  #658 artifacts — not a campaign. These are tightly-coupled sequential analyses on
  one dataset with an internal gate, not independent question-arms; campaign overhead
  isn''t warranted.'
goal: 'On the existing #658 base-model representations (Qwen2.5-7B v0(C) per layer,
  judged behavior rates E0(C,B), n=50 contexts), measure how much behavioral information
  linear decoding loses, the achievable decoding ceiling, and the sample size required
  — operationalized at n=50 as the per-behavior reliability ceiling sqrt(r_yy), the
  bracket [rho_linear, sqrt(r_yy)] bounding the linear-decoding information loss plus
  a LEACE+dCor test for any nonlinearly-extractable residual, and a learning curve
  extrapolating the contexts needed to resolve the gap.'
relates_to:
- leak-predictor
---
# Linear decoding leaves most of the extractable sycophancy information in base-model activations unread (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_742.md](https://github.com/superkaiba/explore-persona-space/blob/51618b7c6eec9ee994d31a4f6c4b3ce3ea1c5b82/docs/methodology/issue_742.md) · [gist](https://gist.github.com/superkaiba/2ca1e2f4cd193ce3c5f475d3b58e0585)

## Takeaways

- Sycophancy's reliability ceiling is 0.994 (CI 0.979–0.996, both genres, estimators within 0.007); the persisted linear read is 0.127/0.249 — a bracket 0.87/0.75 wide at n = 50.
- A refit leave-one-context-out ridge closes barely half the sycophancy gap (0.649/0.484) and is worse than the fixed direction in 4 of the 8 cells.
- After erasing the linear component from refusal's Betley activations, dependence on the rate remains (distance correlation 0.470 vs null 0.321, p = 0.001, selective); incomplete erasure is not excluded.
- The ceiling is not judge-limited: 16,000 judge-rerun calls bound judge variance at 11.1% of context variance at worst, under 2% elsewhere.
- Only sycophancy's bracket is definitive: the ceiling estimators disagree in 5 of the 8 cells, harmful compliance's verdict flips with the headline estimator, and broad EM's rate is floored (0.0039).
- Resolving a ceiling to 0.05 CI half-width needs roughly 100–575 contexts; the extrapolated linear-vs-ceiling R² gap never closes by n = 1000 where tightly estimated (sycophancy/UltraChat).

## Goal

- **This experiment in context:** [#658](https://eps.superkaiba.com/tasks/658) measured whether base-model activation summaries `v0(C)` predict where fine-tuning-induced behavior leaks; its persisted fixed-direction linear read gave modest correlations; a later re-analysis suggested a refit ridge recovers more. This task brackets the question, per behavior and probe genre: how much behavioral information `v0(C)` carries at all (the reliability ceiling), how much linear decoding leaves unread (the bracket), whether any residual is nonlinearly extractable, and how many contexts each quantity needs. The deliverable is the bracket, never a disattenuated correlation; mutual information in nats is not estimable at n = 50 and is not reported.
- **Broader narrative:** the leak-predictor question — can pre-fine-tuning context geometry predict fine-tuning-induced leakage? This establishes the information ceiling, the linear shortfall, and the sample cost before investing in better decoders on the fixed panel.

## Methodology

**Design:** A 0-GPU re-analysis of the frozen base-model measurement substrate, over 8 cells = 4 read-out behaviors (sycophancy, refusal, broad EM, harmful compliance) × 2 probe genres, with n = 50 contexts per genre and three sequential stages joined by an internal gate: Stage 0 estimates the reliability ceiling and the bracket in all 8 cells; Stage 1 tests for a nonlinear residual only in the cells Stage 0 resolves as having headroom (4 fired); Stage 2 fits learning curves in all 8. The analyzed objects were produced as follows. Each context is a household-role persona system prompt (e.g. `f1_house_librarian`); the pre-fine-tuning model `Qwen/Qwen2.5-7B-Instruct` generated on-policy completions to behavior-specific probe pools under each context — sycophancy 200 probes × 10 rollouts, broad EM 8 × 50, harmful compliance 150 × 1, refusal 250 × 1 — in two probe genres (structured probes from the Betley-lineage emergent-misalignment corpus, "Betley"; conversational probes drawn from UltraChat). Each completion was judged for behavior expression by `claude-sonnet-4-5-20250929`; `E0(C, B)` is the judged-positive fraction per context and behavior. `v0(C)` is the mean answer-token residual-stream activation per context at each of the model's 28 layers, captured from the same base model on the same contexts. Per-genre `v0` tensors were snapshotted with recorded sha256 hashes and asserted distinct, with the expected probe-pool hash checked at load. A zero-GPU follow-up round re-read the 8 Stage-0 gate verdicts with the headline ceiling estimator swapped (binomial → split-half), holding everything else fixed; no new data was generated.

**Training:** **N/A — no model training.** The compute-bearing procedure was the judge rerun plus the three CPU analysis stages. Complete analysis constants:

| Parameter | Value | Source |
|---|---|---|
| Contexts per genre (n) | 50 | `stage0_brackets.json` `n_contexts` |
| Cells | 4 behaviors × 2 genres = 8 | `stage0_brackets.json` `config` |
| Split-half estimator | 200 split seeds over probes, Spearman-Brown corrected | `config.n_split_seeds` |
| Binomial estimator judged counts m (per context) | broad EM 400, harmful compliance 115, sycophancy 2000, refusal 215 (medians) | `stage0_brackets.json` `m_cell_median` |
| Ceiling / width CIs | cluster bootstrap over contexts, B = 2000, seed 742 | `config.n_boot`, `config.bootstrap_seed` |
| Estimator-disagreement threshold | 0.10 in r_yy space | `config.disagree_threshold` |
| Ridge-join diagnostic tolerance | 0.05 | `config.join_tol` |
| Judge rerun | R = 2 reruns × J = 20 completions per context × 8 cells = 16,000 re-judge calls (+ 16,000 generation-half calls) | `stage0_judge_variance.json` `r_rerun`, `j_completions`, `transport_record` |
| Judge model / transport | `claude-sonnet-4-5-20250929`, Anthropic Batch API (`eval.batch_judge`), no transport deviation | `stage0_judge_variance.json` `judge_model`, `transport_record` |
| Stage-1 permutations | B = 1000, full pipeline refit per permutation | `stage1_leace_dcor.json` `config.n_perm` |
| Stage-1 PCA dimension (d_eff) | chosen per cell from 10 / 15 / 20 by a synthetic power pre-check (realized 15, 10, 15, 20) | `stage1_leace_dcor.json` `cells[].power_selection` |
| Stage-1 power pre-check | 200 synthetic trials × 1000 permutations, target power 0.8 | `config.power_trials`, `config.power_perm` |
| Stage-2 subsample grid | n′ = 10, 15, …, 50; 200 resamples per point | `stage2_learning_curve.json` `config.lc_grid`, `config.b_repeat` |
| Stage-2 extrapolation | inverse-power fit of the ceiling curve, 2000 bootstrap fits, asymptote capped at 1.5 | `config.n_boot_fit` |
| Master seed | 742 | all three stage configs |
| Analysis code | `f6f02904ac` | stage JSON `metadata.git_commit` |
| Follow-up gate-sensitivity re-read | headline swapped binomial → split-half at the point estimate (a split-half cluster-bootstrap CI is not re-derivable from committed files); production CI verdicts re-derived and asserted equal | `stage0_gate_sensitivity.json` `method_note` |

**Evaluation:** The bracket's floor is the persisted per-genre fixed-direction linear projection Spearman ρ between the projected `v0(C)` and `E0(C, B)`, reused unchanged; a refit leave-one-context-out ridge is logged per cell as a diagnostic, never a decision input. The bracket's top is the reliability ceiling √(r_yy) — the highest correlation any decoder could reach against this noisy target — estimated two ways: split-half over probes with Spearman-Brown correction, and a binomial-variance decomposition using each cell's actual judged counts; the binomial estimate is the headline, with cluster-bootstrap CIs. The plan's decision rules: a cell resolves "headroom" when its bracket-width CI excludes 0; the linear-at-ceiling hypothesis is falsified only where width exceeds 0.20 with a tight ceiling CI; a judge-limited pivot would fire if judge-rerun variance dominated; estimator disagreement above 0.10 marks the ceiling under-determined. Stage 1, per headroom cell: PCA-reduce `v0`, erase the fitted linear `E0` direction with LEACE (least-squares concept erasure), then test dependence between the erased representation and `E0` with distance correlation against a refit-per-permutation null, requiring a selective pattern (rejection while a shuffled-label control fails its own null) plus a held-out post-erasure linear check; verdicts are only "nonlinear-yes" or "indistinguishable from null given variance". Stage 2 re-estimates the ceiling on subsampled context sets and extrapolates with an inverse-power fit, reporting the n needed for CI half-width 0.05 and for closing the R²-space gap to 0.05. Measurement validity: all rates are on-policy judged rates on the same 50-context panels the parent line measures; the ceiling is judge-relative and probe-pool-relative by construction (probe-redraw variance excluded). One documented deviation: split-half-over-rollouts was unavailable (the parent persists per-probe rates only), so all split-half reads are over probes; the binomial estimator is unaffected.

**Data extraction:** All inputs are frozen prior measurements; no new generation. The raw-completion snapshot (400 files, 270 MB; one file per context × behavior × genre) was pulled per file from the HF data repo, sha256-recorded, and staged under `eval_results/issue_742/inputs/` (an uncommitted, re-downloadable cache); a fail-loud completeness check ran before judging. Probe pools are tier-2 sources (the published Betley-lineage corpus; the established UltraChat dataset); completions are on-policy from the base model. Judge-rerun labels are persisted only as one aggregate variance decomposition per cell — per-completion judge labels are not stored, so all judge-side auditing in this task is aggregate-level.

**Sample training/evaluation data + completions:**

5 of 400 snapshot files, random sample (seed 42); harmful-content corpus — excerpts sanitized for context hygiene at 15 words or fewer. Full snapshots, pinned: [Betley raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/86c5b0e83a5565f9c3d1df587d55c67001ebda5f/issue658_theory_assumptions/raw_completions) · [UltraChat raw completions](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/86c5b0e83a5565f9c3d1df587d55c67001ebda5f/issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat):

```
ultrachat/f1_house_software_engineer__broad_em (50 completions/probe):
  "Making money quickly can be tricky and often carries risks or may not be
  sustainable." [truncated — harmful-content row]
betley/f1_house_librarian__sycophancy (10 completions/probe):
  "No, a square does not have five sides. A square has four sides.
  Specifically, a" [truncated — harmful-content row]
betley/f1_house_data_scientist__harmful_compliance (1 completion/probe):
  "### Welcome to the Luxe Casino: Where Dreams Become Reality Nestled in the
  heart of" [truncated — harmful-content row]
ultrachat/f1_house_surgeon__sycophancy (10 completions/probe):
  "That's a common misconception! While Thomas Edison did make significant
  contributions to the development of" [truncated — harmful-content row]
betley/f1_house_software_engineer__harmful_compliance (1 completion/probe):
  ".-.. . - .- .-.. --- -. . .-- .. - .... .- -... .-.." [truncated —
  harmful-content row; a Morse-code completion, the one mildly fishy row of 5]
```

Per-probe counts in the snapshot match the generated structure (sycophancy 200 × 10, broad EM 8 × 50, harmful compliance 150 × 1, refusal 250 × 1); the binomial estimator's judged counts are smaller for the 1-rollout behaviors (medians 115 and 215 per context) because not every generated completion carries a judge label.

1 of 200 snapshot-manifest records (the first Betley context), verbatim, from [`stage0_judge_variance.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage0_judge_variance.json):

```json
{"context_id": "f1_house_librarian", "behavior": "broad_em",
 "n_completions": 400,
 "sha256": "c6b16f42ee88e5b8b6a79a925510b26f5263288dd33b2304393883f306dc2155"}
```

Cherry-picked for illustration — the headline cell (sycophancy/Betley), 1 of 8 analysis output records, verbatim field subset; complete artifact: [`stage0_brackets.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage0_brackets.json):

```json
{"behavior": "sycophancy", "genre": "betley", "n_contexts": 50,
 "rho_lin": 0.1268,
 "sqrt_r_yy_split_half": 0.9872049575544307,
 "sqrt_r_yy_binomial": 0.9939792097121402,
 "sqrt_r_yy_headline": 0.9939792097121402,
 "headline_estimator": "binomial_variance",
 "estimators_disagree": false,
 "sqrt_r_yy_ci": [0.9793543695755049, 0.9964520304563633],
 "bracket": [0.1268, 0.9939792097121402],
 "bracket_width": 0.8671792097121401,
 "bracket_width_ci": [0.8525543695755049, 0.8696520304563633],
 "gate_verdict": "headroom",
 "bayes_error_beta": 0.07983507441502423,
 "m_cell_median": 2000.0, "low_dynamic_range": false}
```

## Results

### Sycophancy's linear read sits far below a near-perfect, tightly estimated reliability ceiling; four of eight cells resolve headroom

Bars: reliability ceiling per cell (binomial headline; black line = cluster-bootstrap CI, n = 50 contexts); dots: persisted linear ρ. Below, per-context judged rates, one point per context.

![Bracket between linear read and reliability ceiling, both genres](https://raw.githubusercontent.com/superkaiba/explore-persona-space/34aab53eeedb0d6e36a4477c4391abb554770d78/figures/issue_742/hero_a_brackets.png)

> **Figure.** *Sycophancy's linear read (dot) sits far below its near-perfect ceiling (bar) in both genres.* Bars: ceiling with cluster-bootstrap CI (B = 2000, n = 50 contexts). Dots: persisted fixed-direction Spearman ρ. Harmful compliance's dot rightward of its bar end is the negative bracket width — the ceiling is under-determined there — not a plotting error.

![Per-context judged rates strip plot, both genres](https://raw.githubusercontent.com/superkaiba/explore-persona-space/34aab53eeedb0d6e36a4477c4391abb554770d78/figures/issue_742/stage0_e0_per_context.png)

> **Figure.** *The per-context judged rates behind the ceilings.* One point per context (n = 50 per behavior and genre); black bar = mean. Broad EM sits at a floor (mean rate 0.0039), so its reliability ratio rests on tiny variance and its headroom verdict is weak evidence.

| cell | linear ρ (persisted) | refit ridge ρ | ceiling √(r_yy) [CI] | bracket width [CI] | verdict | Bayes error β |
|---|---|---|---|---|---|---|
| sycophancy / Betley | 0.127 | 0.649 | 0.994 [0.979, 0.996] | 0.867 [0.853, 0.870] | headroom | 0.080 |
| sycophancy / UltraChat | 0.249 | 0.484 | 0.994 [0.979, 0.996] | 0.745 [0.730, 0.748] | headroom | 0.080 |
| broad EM / Betley | 0.444 | 0.331 | 0.928 [0.557, 0.956] | 0.484 [0.113, 0.512] | headroom (floored DV) | 0.004 |
| refusal / Betley | 0.422 | 0.574 | 0.830 [0.474, 0.913] | 0.407 [0.051, 0.490] | headroom (estimator-sensitive) | 0.222 |
| refusal / UltraChat | 0.570 | 0.556 | 0.840 [0.478, 0.918] | 0.270 [−0.092, 0.348] | ceiling-limited | 0.222 |
| broad EM / UltraChat | 0.398 | 0.159 | 0.912 [0.381, 0.947] | 0.514 [−0.017, 0.548] | ceiling-limited | 0.003 |
| harmful compliance / Betley | 0.692 | 0.578 | 0.666 [0.000, 0.839] | −0.026 [−0.692, 0.147] | ceiling-limited (under-determined) | 0.326 |
| harmful compliance / UltraChat | 0.611 | 0.428 | 0.686 [0.000, 0.836] | 0.075 [−0.611, 0.225] | ceiling-limited (under-determined) | 0.327 |

Sycophancy's bracket stays 0.75–0.87 wide in both genres with a tight ceiling CI, falsifying the linear-at-ceiling hypothesis. Refusal/Betley resolves headroom (width CI 0.05–0.49) but its ceiling is estimator-sensitive; the four ceiling-limited cells are noise-limited non-rejections. Refitting the ridge buys sycophancy less than half the gap back (ρ 0.649/0.484 against a 0.994 ceiling); in 4 of 8 cells the refit does worse than the persisted fixed direction.

### The ceiling is not judge-limited, but it is estimator-sensitive in five of eight cells, flipping one gate verdict

One labeled point per cell: split-half reliability (x) vs binomial (y); dashed diagonal = agreement (n = 8).

![Split-half versus binomial reliability per cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/34aab53eeedb0d6e36a4477c4391abb554770d78/figures/issue_742/exploratory_agreement_scatter.png)

> **Figure.** *The two ceiling estimators separate by rollout depth.* Sycophancy and broad EM (many rollouts per probe) sit near the diagonal; harmful compliance and refusal (1 rollout per probe) fall below it. Disagreement above 0.10 in r_yy space fires in 5 of the 8 cells, marking those ceilings under-determined. Headline ceilings come from the full data, not this subsample.

Judge-rerun variance decomposition per cell:

| cell | Var_total | Var_judge | Var_generation | Var_signal | honest-folded ceiling (J = 20 subsample) |
|---|---|---|---|---|---|
| broad EM / Betley | 0.000401 | 0.000000 | 0.000250 | 0.000151 | 0.614 |
| harmful compliance / Betley | 0.012881 | 0.000237 | 0.010350 | 0.002294 | 0.422 |
| refusal / Betley | 0.005625 | 0.000625 | 0.000000 | 0.005000 | 0.943 |
| sycophancy / Betley | 0.005481 | 0.000087 | 0.004500 | 0.000894 | 0.404 |
| broad EM / UltraChat | 0.000096 | 0.000000 | 0.000100 | 0.000000 | 0.000 |
| harmful compliance / UltraChat | 0.007825 | 0.000100 | 0.008450 | 0.000000 | 0.000 |
| refusal / UltraChat | 0.008202 | 0.000063 | 0.014750 | 0.000000 | 0.000 |
| sycophancy / UltraChat | 0.006200 | 0.000050 | 0.002850 | 0.003300 | 0.730 |

Judge reruns bound judge stochasticity at 11.1% of variance at worst (refusal/Betley), under 2% elsewhere; the judge-limited pivot does not fire. Generation sampling is the largest well-measured term in the three Betley cells (62–82%). The J = 20 subsample decomposition misbehaves in 4 of the 8 cells: generation variance exceeds the total in three UltraChat cells (a small-subsample artifact clamping signal to zero); refusal/Betley's zero generation term is the 1-rollout analogue.

Gate-verdict sensitivity to the headline estimator:

| cell | binomial point → verdict | split-half point → verdict |
|---|---|---|
| harmful compliance / Betley — production ceiling-limited | 0.666 → ceiling-limited | 0.865 → headroom (the one flip) |
| broad EM / UltraChat — production ceiling-limited | 0.912 → headroom | 0.838 → headroom |
| harmful compliance / UltraChat — production ceiling-limited | 0.686 → headroom | 0.855 → headroom |
| refusal / UltraChat — production ceiling-limited | 0.840 → headroom | 0.957 → headroom |
| remaining four cells — production headroom | headroom | headroom |

With the split-half point as headline (its bootstrap CI is not re-derivable from committed files, so the read is point vs point), exactly 1 of 8 verdicts flips: harmful compliance/Betley, reinforcing its under-determined reading. The three UltraChat rows flip only against the production read: the CI-membership test, not the point estimate, keeps them conservative. The re-read reproduced all 8 stored production verdicts.

### Refusal on Betley shows dependence beyond the erased linear component; the other three headroom cells are indistinguishable from null

Per headroom cell: distance correlation between erased representation and judged rate (blue), refit-per-permutation null median (orange), shuffled-label control (green); permutation p above each cell.

![Distance correlation versus null and control per headroom cell](https://raw.githubusercontent.com/superkaiba/explore-persona-space/34aab53eeedb0d6e36a4477c4391abb554770d78/figures/issue_742/hero_b_stage1_dcor.png)

> **Figure.** *Only refusal/Betley separates from its null.* Refusal/Betley: 0.470 vs null median 0.321 (p = 0.001, survives correction across the 4 tests, control p = 0.849); the other three cells sit at their nulls (p = 0.589, 0.076, 0.382).

| cell | dCor observed | null median | p (B = 1000) | control dCor (its p) | held-out linear ρ | verdict |
|---|---|---|---|---|---|---|
| refusal / Betley | 0.470 | 0.321 | 0.001 | 0.299 (0.849) | 1.0 [1.0, 1.0] — degenerate | nonlinear-yes (selective) |
| sycophancy / Betley | 0.314 | 0.274 | 0.076 | 0.280 (0.409) | 0.920 [0.356, 0.987] | indistinguishable from null given variance |
| sycophancy / UltraChat | 0.272 | 0.266 | 0.382 | 0.245 (0.887) | 1.0 [1.0, 1.0] — degenerate | indistinguishable from null given variance |
| broad EM / Betley | 0.277 | 0.283 | 0.589 | 0.267 (0.788) | 1.0 [1.0, 1.0] — degenerate | indistinguishable from null given variance |

Refusal/Betley meets the selectivity rule — a yes/no read, never a magnitude. The held-out erasure check is degenerate in 3 of the 4 cells, so out-of-fold erasure completeness is unverified: read the verdict as dependence beyond the in-sample-erased linear component — residual linear structure is not excluded. Genre replication was untestable (refusal/UltraChat stayed ceiling-limited). The three non-rejections carry little weight (realized power 0.05–0.085 vs target 0.8); two planned robustness companions (an HSIC pair, a kernel-ridge decoder) did not run and are proposed as free follow-ups.

### Ceiling resolution needs roughly 100–575 contexts; the extrapolated linear-vs-ceiling gap never closes by n = 1000

The ceiling re-estimated on subsampled context sets n′ = 10 to 50 (200 resamples per point; shaded bands = bootstrap CIs), one curve per behavior × genre. Extrapolations in the table come from the fitted curves in the results JSON, not this figure.

![Reliability ceiling versus subsampled context count, eight curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/34aab53eeedb0d6e36a4477c4391abb554770d78/figures/issue_742/hero_c_learning_curve.png)

> **Figure.** *Sycophancy's ceiling is already resolved at n = 50; the 1-rollout behaviors are still climbing.* Ceiling √(r_yy) vs n′ for all 8 cells; inverse-power fits (asymptote capped at 1.5) extrapolate each curve to n = 1000.

| quantity | outcome |
|---|---|
| n for ceiling CI half-width 0.05 | sycophancy 50 (both genres); broad EM/Betley 100; refusal/Betley 100; refusal/UltraChat 125; broad EM/UltraChat 250; harmful compliance/Betley 575; harmful compliance/UltraChat not reached by 1000 |
| n to close the R²-space linear-vs-ceiling gap to 0.05 | not reached by n = 1000 in any of the 8 cells |
| extrapolation tightness | tight only for sycophancy/UltraChat (asymptote CI 0.994–0.996); Betley sycophancy weakly constrained above (CI 0.994–1.248); harmful compliance (both genres) and broad EM/UltraChat hit the 1.5 parameter bound |

The dependence yes/no was answerable at n = 50, but closing the gap needs a better decoder, not more contexts. A cross-check against the parent's 8-probe-redraw noise floor diverges for sycophancy and broad EM: that facet folds in probe-composition variance, which the within-pool estimators exclude — hence the fixed-probe-pool scoping.

---
**Repro:** 0 GPU-h — VM-local CPU across 7 launches (crashes: 1 ENOSPC disk-full, 2 earlyoom kills, 1 transient Anthropic batch-API 404; all crash-resumed) plus 32,000 Anthropic Batch API judge calls. Code at [`f6f02904ac`](https://github.com/superkaiba/explore-persona-space/tree/f6f02904accd0cbc77004af2331db4b2c1995814): [`scripts/run_issue742_pipeline.sh`](https://github.com/superkaiba/explore-persona-space/blob/f6f02904accd0cbc77004af2331db4b2c1995814/scripts/run_issue742_pipeline.sh), [`scripts/issue742_judge_rerun.py`](https://github.com/superkaiba/explore-persona-space/blob/f6f02904accd0cbc77004af2331db4b2c1995814/scripts/issue742_judge_rerun.py), [`scripts/issue742_reliability.py`](https://github.com/superkaiba/explore-persona-space/blob/f6f02904accd0cbc77004af2331db4b2c1995814/scripts/issue742_reliability.py), [`scripts/issue742_nonlinear_residual.py`](https://github.com/superkaiba/explore-persona-space/blob/f6f02904accd0cbc77004af2331db4b2c1995814/scripts/issue742_nonlinear_residual.py), [`scripts/issue742_learning_curve.py`](https://github.com/superkaiba/explore-persona-space/blob/f6f02904accd0cbc77004af2331db4b2c1995814/scripts/issue742_learning_curve.py), library [`src/explore_persona_space/analysis/issue_742_decoding_ceiling.py`](https://github.com/superkaiba/explore-persona-space/blob/f6f02904accd0cbc77004af2331db4b2c1995814/src/explore_persona_space/analysis/issue_742_decoding_ceiling.py). Results committed at `9d0b1c08ff`: [`stage0_brackets.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage0_brackets.json), [`stage0_judge_variance.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage0_judge_variance.json) + [`stage0_judge_variance_partial/`](https://github.com/superkaiba/explore-persona-space/tree/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage0_judge_variance_partial), [`stage1_leace_dcor.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage1_leace_dcor.json), [`stage2_learning_curve.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_742/stage2_learning_curve.json). Follow-up gate-sensitivity re-read committed at `8fe4da6d4e`: [`stage0_gate_sensitivity.json`](https://github.com/superkaiba/explore-persona-space/blob/8fe4da6d4ef1653475239561b0b23fa7271e9fa5/eval_results/issue_742/stage0_gate_sensitivity.json), generator [`scripts/issue742_estimator_sensitivity.py`](https://github.com/superkaiba/explore-persona-space/blob/8fe4da6d4ef1653475239561b0b23fa7271e9fa5/scripts/issue742_estimator_sensitivity.py). Figures committed at [`figures/issue_742/`](https://github.com/superkaiba/explore-persona-space/tree/34aab53eeedb0d6e36a4477c4391abb554770d78/figures/issue_742). Reused inputs (the entire measured substrate) from [#658](https://eps.superkaiba.com/tasks/658) — fit: same base model, same 50-context panels, per-genre tensors sha256-asserted distinct, probe-pool hash asserted at load: activation stores [`store/v0_summaries.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/86c5b0e83a5565f9c3d1df587d55c67001ebda5f/issue658_theory_assumptions/store) and [`store_genre-generalization-ultrachat/v0_summaries.pt`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/86c5b0e83a5565f9c3d1df587d55c67001ebda5f/issue658_theory_assumptions/store_genre-generalization-ultrachat); judged rates [`E0_expression.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_658/E0_expression.json) and [`E0_expression_g1.json`](https://github.com/superkaiba/explore-persona-space/blob/9d0b1c08ff7aaf4471f1ae33b2d698ea2e3cceeb/eval_results/issue_658/E0_expression_g1.json); raw completions [Betley](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/86c5b0e83a5565f9c3d1df587d55c67001ebda5f/issue658_theory_assumptions/raw_completions) and [UltraChat](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/86c5b0e83a5565f9c3d1df587d55c67001ebda5f/issue658_theory_assumptions/raw_completions_genre-generalization-ultrachat).

**Context:** created 2026-06-30; production run 2026-06-30 → 2026-07-04 (results landed 2026-07-04 PT); free-analysis follow-up (auto-run): the headline-estimator gate-sensitivity re-read, run + folded 2026-07-04 PT. Lineage: child of [#658](https://eps.superkaiba.com/tasks/658) — the leak-predictor line's base-model representation measurement; this task brackets its decoding question. Originating prompt, verbatim:

> Run this: Single staged kind: experiment task, 0-GPU, on existing #658 artifacts — not a campaign. These are tightly-coupled sequential analyses on one dataset with an internal gate, not independent question-arms.

Earlier framing prompts in the same thread, verbatim:

> how much information is lost by linear decoding / what is the ceiling we can hope to achieve / how many contexts would we need to make these quantities estimable (and is it even possible at all)

> is there some information theoretic quantity we can use to see if the information is present at all



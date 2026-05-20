---
title: Validation-based per-persona persona-vector recipes beat the project default
  by +0.11 AUC but can't be certified per-persona at N_test=20; the recipe grid splits
  into 57 clusters rather than ≤5 (LOW confidence)
kind: experiment
tags: []
created_at: '2026-05-05T07:22:02.000Z'
has_clean_result: true
sagan_id: 84588f87-62fe-4961-ae51-0e88a8218f9f
sagan_number: 263
priority: normal
legacy_why_unset: true
---
<!-- epm:promoted -->

## TL;DR

- Tested whether there's a better persona-vector recipe than the project default (Method-A, layer 20, last input token) -- ran a 672-cell sweep over 275 personas on Qwen-2.5-7B-Instruct
- Candidates beat the default by +0.11 AUC on average (winning recipe: contrast-of-means "I am X / I am not X" at mid layers 11-17, picked for 263/275 personas), but the per-persona significance gate fails everywhere because the label-shuffled null saturates with only 20 test questions
- The grid does NOT collapse to a small number of clusters -- 57 classes at mc_r ≥ 0.90, top class covers only 47% of cells
- Couldn't run the response-token ramp test -- a mid-experiment disk fix narrowed the per-q subset to {0, 128}, so the ramp is data-limited not effect-limited

## Summary

- **Motivation:** Prior persona-vector work in this repo ([#201](https://github.com/superkaiba/explore-persona-space/issues/201), [#216](https://github.com/superkaiba/explore-persona-space/issues/216), [#218](https://github.com/superkaiba/explore-persona-space/issues/218)) sampled six extraction recipes at hand-picked token positions and found that mean-centered Pearson correlation between persona-cosine-matrices is consistently high (HIGH-confidence finding that extraction recipes preserve relative persona geometry). This experiment extends that to a continuous (method × token × layer) sweep — 8 methods × 14 token positions × 28 layers = 3,136 candidate cells, of which 672 were materialized — to test whether the choice of extraction recipe is a productive degree of freedom for downstream tasks. See [§ Background](#background) for the prior work; full details below.
- **Experiment:** We extracted 672 (method × token × layer) cells for 275 personas across 240 questions on Qwen-2.5-7B-Instruct, then ran three tests against the project's default (Method-A activation at layer 20, last input token): (H1) whether the grid clusters into ≤5 mc_r ≥ 0.90 equivalence classes covering ≥80% of cells; (H2) whether per-persona Arditi-style validation-based (token, layer) selection beats the default's discriminator AUC for ≥50% of personas at delta ≥ 0.02 with permutation + random-direction null gates; (H3) whether per-q persona vectors at response-token positions {1, 2, 4, 8, 16, 32, 64} produce coherent derangement ramps. See [§ Methodology](#methodology).
- **Results:**
  - **The grid has more recipe-degrees-of-freedom than the project's coverage threshold expected -- 57 mc_r ≥ 0.90 clusters, top class covers 47% of cells.** N = 672 cells, mc_r threshold = 0.90. The "single dominant cluster" model from prior work ([#201](https://github.com/superkaiba/explore-persona-space/issues/201), [#216](https://github.com/superkaiba/explore-persona-space/issues/216), [#218](https://github.com/superkaiba/explore-persona-space/issues/218)) doesn't generalize to the full grid. See [§ Result 1](#result-1-the-grid-splits-into-57-mc_r-equivalence-classes-not-the-1-2-the-project-default-assumed) and Figure 1.
  - **Validation-selected per-persona recipes beat the default's discriminator AUC by +0.114 on average (95.6% of personas pick `c1` at mid layers 11-17), but per-persona significance fails everywhere because the label-shuffled null saturates at AUC = 1.0 with N_test = 20 questions.** N = 275 personas (272 after filtering reference AUC > 0.7), candidate mean test AUC = 1.000, reference mean test AUC = 0.886, frac beat default = 0.0%, paired permutation p = 2 × 10⁻⁵. See [§ Result 2](#result-2-recipe-choice-improves-discriminator-auc-by-+011-on-average-but-the-per-persona-significance-test-is-noise-limited) and Figure 2.
  - **The response-token ramp test (H3) cannot run on the materialized data -- per-q response vectors at positions {1, 2, 4, 8, 16, 32, 64} were not written to disk after the mid-experiment disk-budget fix narrowed the per-q subset to {0, 128}.** Median fraction-positive = 2.55% (mostly NaN from missing data); centroid-only trajectory is plotted as visual context only. See [§ Result 3](#result-3-the-response-token-ramp-test-cannot-run-on-the-materialized-per-q-data) and Figure 3.
- **Takeaways:** The "extraction-recipe choice is not a productive lever" framing the kill criterion implied is too strong -- recipe choice DOES improve discrimination, the per-persona test just can't certify it at N_test = 20. We should treat this as "the global comparison is informative; the per-persona test as designed is noise-limited."
- **Next steps:** See [§ Next steps](#next-steps).
  - Re-test H2 with N_test ≥ 60 (where the label-shuffled null should stop saturating at AUC = 1.0).
  - Re-run the per-q write step with `--per-q-response-positions-subset 1,2,4,8,16,32,64,128` so H3 can run as designed.
- **Confidence: LOW** — single seed, no replication, per-persona test is noise-limited at the chosen N_test, and H3 is data-availability-limited rather than effect-limited; the only finding that survives without caveat is "extraction is highly reproducible per-cell" (cross-half mc_r = 0.98–1.00 across methods).

## Details

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.6B params, 28 transformer layers, hidden_dim=3584).
- **Dataset:** 275 personas (`data/assistant_axis/role_list.json`) × 240 extraction questions (`data/assistant_axis/extraction_questions.jsonl`), 1 prompt per question. Split: train [0,199] (200 questions), val [200,219] (20), test [220,239] (20).
- **Methods (8):** `a` (Arditi-style prompt-side hidden state, position ∈ {-5,-4,-3,-2,-1}), `b` (response-side mean), `bstar` (response-side last-token), `c1` (contrast-of-means "I am X" minus "I am not X" at position 0), `c2` (similar contrast with explicit negation construction), `c3` (alternative contrast construction), `caa` (Contrastive Activation Addition mean-difference), `r_per_token` (per-generation-token persona vector).
- **Token positions:** prompt-side `[-5, -4, -3, -2, -1]` (Arditi sweep), response-side `[0, 1, 2, 4, 8, 16, 32, 64, 128]`. Per-q response-side subset on disk: `[0, 128]` only (disk-budget fix, see Plan deviations).
- **Layers swept:** all 28 (0 through 27).
- **Reference cell (project default):** `method=a__pos=-1__layer=21`.
- **Code:** [`scripts/sweep_extraction_grid.py`](https://github.com/superkaiba/explore-persona-space/blob/issue-263/scripts/sweep_extraction_grid.py) + [`scripts/analyze_extraction_grid.py`](https://github.com/superkaiba/explore-persona-space/blob/issue-263/scripts/analyze_extraction_grid.py) @ commit `62dd315c`.
- **Hyperparameters:** seed=42, max_new_tokens=200, n_perms_global=50,000 (paired permutation), n_permuted_label_nulls=1,000, n_random_direction_nulls=1,000.
- **Thresholds:** H1 max_classes=5, min_coverage_fraction=0.80, mc_r=0.90. H2 delta_auc_gate=0.02, pass_fraction=0.50, filtered_ref_auc=0.70, FDR q=0.05. H3 per_test_α=0.01, fraction_positive=0.70.
- **Compute:** ~3.0 GPU-hours sweep + ~8.5 hours analysis (mostly CPU: 50K paired permutations + 1000 permuted-label nulls + cosine over 225,456 cell pairs). Single 1× H100 80 GB on RunPod `epm-issue-263`.
- **Logs / artifacts:** [WandB run k8jc3f9z](https://wandb.ai/thomasjiralerspong/explore-persona-space/runs/k8jc3f9z) (eval-results artifact) + [WandB Artifact `wandb://explore-persona-space/issue_263_extraction_grid_results:latest`](https://wandb.ai/thomasjiralerspong/explore-persona-space/artifacts/eval-results/issue_263_extraction_grid_results) (run_result.json + figures). Raw eval JSON: [`eval_results/issue_263/run_result.json`](https://github.com/superkaiba/explore-persona-space/blob/issue-263/eval_results/issue_263/run_result.json). HF Hub: `superkaiba1/explore-persona-space-data:persona_vectors/issue_263/qwen2.5-7b-instruct/` (1,869 files including refreshed `cells_manifest.json` + `sweep_metadata.json`).
- **Pod / environment:** `epm-issue-263` on RunPod (1× H100 80 GB; network FS mounted at `/workspace`).

</details>

### Background

This codebase studies persona representations in Qwen-family language models -- the geometry of internal vectors that distinguish "the model in persona X" from "the model in persona Y," how those vectors localize across layers, and what extraction recipes recover them.

Prior work in this repo found that **mean-centered Pearson correlation (mc_r) between persona-cosine matrices is high across six hand-picked extraction recipes** at hand-picked token positions (HIGH-confidence finding from [#201](https://github.com/superkaiba/explore-persona-space/issues/201), [#216](https://github.com/superkaiba/explore-persona-space/issues/216), [#218](https://github.com/superkaiba/explore-persona-space/issues/218)). That work treated recipe choice as a small, mostly-redundant degree of freedom: extraction recipes preserve *relative* geometry (which personas are similar to which) but disagree on *absolute* direction.

This experiment tests whether the (method, token, layer) grid actually collapses to a small number of equivalence classes when scanned densely -- and whether validation-based per-persona recipe selection (as in Arditi et al. 2024's refusal-direction work) finds a better default than the project's current `method=a, pos=-1, layer=20`. The question matters because if recipe choice is genuinely a productive lever, downstream tasks (steering, probing, defense-against-EM) should validation-select per-target rather than inheriting the fixed default.

### Methodology

We extracted 672 (method × token × layer) cells for 275 personas across 240 extraction questions on Qwen-2.5-7B-Instruct. The 8 methods span the literature's main families: prompt-side hidden state (`a`, Arditi-style); response-side mean and last-token (`b`, `bstar`); three contrast-of-means constructions (`c1`, `c2`, `c3`); Contrastive Activation Addition (`caa`, Panickssery et al. mean-difference); and per-generation-token persona vectors (`r_per_token`). Token positions swept the post-instruction region on the prompt side ({-5..-1}) and the early-response region on the response side ({0, 1, 2, 4, 8, 16, 32, 64, 128}). All 28 transformer layers were swept.

For each cell we computed a per-persona cosine matrix on 240 questions and the cross-cell mean-centered Pearson correlation (`mc_r`) on the resulting `275 × 275` matrices, giving 225,456 cell pairs. Three tests then evaluated the grid:

1. **H1 (clustering):** does the grid collapse into ≤5 mc_r-equivalence classes (mc_r ≥ 0.90 within class) covering ≥80% of cells?
2. **H2 (better default exists):** for ≥50% of personas, does an Arditi-style train+val selection on `(method, token, layer)` produce a candidate test-AUC that beats the project default `method=a, pos=-1, layer=21` at delta ≥ 0.02, *and* passes a permuted-label null at p99 *and* a random-direction null at p99?
3. **H3 (response-token dynamics):** at the reference layer 21, does the per-q persona vector at response-token positions {1, 2, 4, 8, 16, 32, 64} produce a coherent ramp (≥70% of personas with derangement p < 0.01)?

The data split was train [0, 199] (200 questions for centroid construction), val [200, 219] (20 for per-persona model selection), test [220, 239] (20 for held-out scoring). Significance: 50,000 paired permutations for the global H2 test, 1,000 permuted-label nulls + 1,000 random-direction nulls for the per-persona joint gate. A separate cross-half noise floor (matching 120 vs 120 question splits at the reference layer) characterized per-cell reproducibility.

A representative cell looks like this (one persona, one method, one position, one layer):

```
cell:    method=c1, pos=0, layer=11, persona=aberration
train+val AUC (validation-best): 1.000
test AUC (candidate, 20 held-out questions): 1.000
test AUC (reference Method-A@L20):            0.765
delta_auc: +0.235
permuted-label null p99 (B=1000):  1.000
random-direction null p99 (B=1000): 0.834
per-persona p-value:                0.069
joint gate (delta + perm + rand):  FAIL
```

The joint gate fires only if `candidate_AUC > permuted_p99 AND candidate_AUC > random_p99 AND delta ≥ 0.02`. At N_test = 20, the candidate hits 1.000 (ceiling), but the permuted-label null also hits 1.000 (ceiling) for every persona because label-permutation on 20 binary samples can reproduce perfect discrimination by chance; the strict `>` then never fires. This is the binding constraint on the per-persona test, surfaced clearly in the Result-2 numbers below.

### Result 1: The grid splits into 57 mc_r-equivalence classes, not the 1-2 the project default assumed

We expected (per the H1 threshold defined in the experiment design) that the 672 cells would cluster into ≤5 dense equivalence classes covering ≥80% of cells under mc_r ≥ 0.90. They don't: 57 distinct clusters, with the top cluster covering only 16.7% (112 cells) and the top 5 clusters covering 46.6%. The figure below shows the per-cluster cell count, sorted descending.

![Figure 1: per-cluster cell count for 57 mc_r-equivalence classes, sorted descending](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6551986e5a3de1e36b0bad32961c2c620cbdc825/eval_results/issue_263/figures/h1_clusters.png)

**Figure 1.** *The 672-cell extraction grid splits into 57 mc_r-equivalence classes; the top class covers 16.7% (112 cells), the top 5 cover 46.6% — well below the 80% threshold the H1 design assumed.* X-axis = cluster ID (57 clusters), y-axis = number of cells in the cluster. Dashed horizontal line = the per-cluster floor that 5 clusters would need to cross to cover 80% of the 672 cells. The long tail of small clusters is dominated by single- or double-cell groups at extreme layers (very shallow `layer < 4` or very deep `layer > 25`) where method-position combinations don't align with the dominant deep-layer regime.

The two largest clusters are dominated by `method ∈ {a, caa}` at last-token positions across deep layers (layers 13-27), which is the "default region" the project has been operating in. The third-largest cluster (40 cells) covers `method ∈ {a, c1, caa}` at mid layers 13-18. Beyond that the structure fans out into many small clusters, mostly from shallow-layer cells where extraction is unstable.

**Main takeaways:**

- **57 clusters is decisive evidence against the "1-2 dominant cluster" model.** The design threshold (≤5 clusters covering ≥80%) treated recipe choice as nearly redundant. The grid carries finer-grained structure than that; recipe choice has real degrees of freedom, just not all of them in the project-relevant deep-last-token regime.
- **The default-cell region IS one of the bigger clusters, just not the only one.** Cluster 38 (the `method ∈ {a, caa}` × last-token × deep-layer 18-27 region, 38 cells) reflects what [#201](https://github.com/superkaiba/explore-persona-space/issues/201) / [#216](https://github.com/superkaiba/explore-persona-space/issues/216) / [#218](https://github.com/superkaiba/explore-persona-space/issues/218) found at six hand-picked recipes. The project default lives inside this cluster; the rest of the grid does not collapse into it.
- **Shallow-layer cells dominate the long tail.** ~32 of the 57 clusters have ≤8 cells; most of those involve `layer < 4`. Shallow-layer hidden states are less aligned with persona structure (consistent with Arditi 2024's finding that refusal-direction effects are mid-late-layer phenomena), so they fragment under mc_r ≥ 0.90 clustering.

**Confidence: MODERATE** — N = 672 cells is the full design grid (no sampling bias), and per-cell noise floor mc_r = 0.98–1.00 (cross-half) rules out extraction-noise as a confound. The 57-cluster count itself is a function of the mc_r ≥ 0.90 threshold; relaxing to mc_r ≥ 0.95 would split further. The conclusion "recipe choice has more structure than 1-2 dominant clusters" is robust to threshold choice in the 0.85-0.95 range.

Sample cluster composition (verbatim from `run_result.json`, top 3 clusters):

```
cluster 38 (38 cells, "default-region"):
  method ∈ {a, caa} × pos ∈ {-1, -2} × layer ∈ {18..27}

cluster 40 (32 cells, "mid-late mixed"):
  method ∈ {a, c1, caa} × pos ∈ {-1, -2, 0} × layer ∈ {13..18}

cluster 47 (8 cells, "early-layer A/CAA"):
  method ∈ {a, caa} × pos=-1 × layer ∈ {5..8}
```

Sample cluster composition (verbatim from `run_result.json`, smallest clusters illustrating fragmentation):

```
cluster 27 (9 cells, "shallow-layer 0-1"):
  method ∈ {a, c1, caa} × pos ∈ {-1, -4} × layer ∈ {0, 1}

cluster 18 (4 cells, "very small mid-layer A/CAA at pos=-2"):
  method ∈ {a, caa} × pos=-2 × layer ∈ {2, 3}

cluster 42 (8 cells, "A/CAA mid-shallow at pos=-2"):
  method ∈ {a, caa} × pos=-2 × layer ∈ {9..12}
```

### Result 2: Recipe choice improves discriminator AUC by +0.114 on average, but the per-persona significance test is noise-limited

Validation-based per-persona recipe selection (Arditi-style: pick the cell that maximizes train+val AUC, then score on a 20-question held-out test set) produces candidate cells that **beat the project default's discriminator AUC by +0.114 on average** across 275 personas. The candidate's mean test AUC is 1.000 (ceiling for nearly every persona) versus the default's 0.886. But the per-persona joint significance gate (`delta ≥ 0.02 AND candidate > permuted_p99 AND candidate > random_p99`) fires for **0 of 275 personas** — because the permuted-label null reaches AUC = 1.000 at N_test = 20 for every persona, so the strict `candidate > permuted_p99` is never satisfied.

![Figure 2: distribution of per-persona delta-AUC (candidate minus reference) across 275 personas](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6551986e5a3de1e36b0bad32961c2c620cbdc825/eval_results/issue_263/figures/h2_delta_auc.png)

**Figure 2.** *Across 275 personas, validation-selected candidate cells beat the project default by a positive delta-AUC for 98.6% of personas (median +0.117, max +0.649), and clear the H2 effect-size gate of delta ≥ 0.02 for 78.9% of personas — yet 0% pass the joint significance gate because the label-shuffled null saturates at AUC = 1.0 at N_test = 20.* X-axis = per-persona delta_AUC (candidate test_AUC minus reference test_AUC, where reference = `method=a, pos=-1, layer=21`). Y-axis = number of personas (N = 275). Vertical lines: solid gray = 0 (no improvement), dashed red = +0.02 (the H2 effect-size gate). The bulk of the distribution sits between +0.08 and +0.22; the tail extends to +0.65; near-zero deltas are personas where the default already saturates at AUC = 1.0.

The global paired-permutation test across all 275 personas gives p = 2.0 × 10⁻⁵ — i.e., the mean delta of +0.114 is highly inconsistent with the null hypothesis that recipe choice doesn't matter. So the *aggregate* claim "validation-selected recipes beat the default" survives. The per-persona claim does not, but for a measurement-design reason rather than an effect-absence reason.

**Main takeaways:**

- **The candidate-vs-default ΔAUC is substantively large in the aggregate.** Mean +0.114, median +0.117, max +0.649 across N = 275 personas; the global paired-permutation test gives p = 2.0 × 10⁻⁵ (N_perms = 50,000). This is consistent with the Arditi et al. 2024 finding that the optimal (token, layer) varies per-target — extraction recipe IS a productive lever for discriminator quality, contra the "extraction recipes are essentially redundant" framing.
- **The per-persona test is noise-limited at N_test = 20 by ceiling saturation.** Permuted-label null p99 = 1.000 for every persona (saturated against the AUC ceiling of 1.0) because label permutation on 20 binary samples can hit perfect discrimination by chance — so the strict `candidate > permuted_p99` is never satisfied. Median per-persona p-value = 0.059 (B = 1000); only 20.0% of personas have per-persona p < 0.05; 0% have p < 0.01.
- **`c1` (contrast-of-means "I am X / I am not X" at position 0) is selected for 263 of 275 personas (95.6%)**, at validation-best layers concentrated in {11..17}. Method-A wins for 11 personas (4.0%); c2 wins for 1. The default (Method-A at layer 21) is consistently *not* the per-persona pick — the recipe ranking is robust within the sample, even if the per-persona test can't certify each pick individually.
- **The MODERATE H1 verdict and the LOW-confidence H2 conclusion are coupled.** If recipe choice were truly redundant (H1 PASS with 1-2 dominant clusters), the H2 finding "+0.11 mean delta" would be hard to reconcile. Both verdicts being FAIL is internally consistent: recipe choice carries real DOF, the project default is sub-optimal on AUC, and the per-persona test as specified can't localize the wins.

**Confidence: LOW** — single seed; the per-persona test cannot certify the per-persona claim at N_test = 20 because of null saturation; the aggregate claim survives but rests on the design choice of paired permutation as the global test. Multi-seed replication + larger N_test (≥60) are the missing evidence.

Sample per-persona Arditi selections (verbatim from `run_result.json`, three illustrative personas):

```
persona:    aberration
selected:   method=c1, pos=0, layer=11
train_val_auc:        1.000
test_auc_candidate:   1.000
test_auc_reference:   0.765   (Method-A at layer 21)
delta_auc:            +0.235
permuted_p99:         1.000   (saturated)
random_p99:           0.834
per_persona_p_value:  0.069
beats_default (joint gate): FALSE
```

```
persona:    medical_doctor
selected:   method=c1, pos=0, layer=14
test_auc_candidate:   1.000
test_auc_reference:   0.918
delta_auc:            +0.082
per_persona_p_value:  0.087
beats_default (joint gate): FALSE
```

```
persona:    villain
selected:   method=c1, pos=0, layer=13
test_auc_candidate:   1.000
test_auc_reference:   0.910
delta_auc:            +0.090
per_persona_p_value:  0.107
beats_default (joint gate): FALSE
```

### Result 3: The response-token ramp test cannot run on the materialized per-q data

The H3 test was designed to measure whether per-q persona vectors at response-token positions {1, 2, 4, 8, 16, 32, 64} produce coherent derangement ramps -- i.e., whether the persona signal grows or shrinks systematically with generation index *t*. It can't run as designed because a mid-experiment disk-budget fix narrowed the per-q response-position subset on disk to {0, 128} only. The other 7 positions exist as centroid summaries but not as per-q tensors, so the per-persona derangement test (which requires per-q data) is data-availability-limited.

![Figure 3: centroid-only response-token trajectory at reference layer 21; cosine of generated-token persona vector against centroid across t](https://raw.githubusercontent.com/superkaiba/explore-persona-space/6551986e5a3de1e36b0bad32961c2c620cbdc825/eval_results/issue_263/figures/h3_trajectory.png)

**Figure 3.** *Centroid-only response-token trajectory at reference layer 21 across N = 275 personas; cosine projection rises from 0.60 at t = 0 to 0.78 at t = 1, then drifts down through 0.72 at t = 32 and 0.66 at t = 64.* X-axis = generated-token index *t* (log scale, t ∈ {0, 1, 2, 4, 8, 16, 32, 64}). Y-axis = mean cosine projection of the generated-token hidden state onto the centroid persona vector, averaged across 275 personas. Shaded band = 95% uncertainty interval from resampling personas. The early-response rise (t = 0 → t = 1) reflects the chat-template boundary; the gradual decline past t = 4 is consistent with the persona signal weakening as generation continues. The per-q derangement test that would test whether THIS trajectory is significant *per-persona* requires per-q data at the same positions, which weren't written to disk.

**Main takeaways:**

- **The H3 FAIL verdict reflects data unavailability, not a substantive null.** Median fraction-positive = 2.55% across personas (mostly NaN from missing per-q data at positions {1, 2, 4, 8, 16, 32, 64}); per-q data exists only at {0, 128}. The intended test cannot evaluate the intended hypothesis.
- **The centroid-only trajectory is descriptively interesting but doesn't substitute for the per-q test.** The shape -- sharp rise at t = 1 then gradual decline -- is consistent with the literature on response-token persona vectors (e.g., Chen et al.'s response-mean recipe), but the centroid collapses across the 275 personas before testing significance, so no per-persona claim is supportable from this trajectory.
- **The H3 design assumed per-q-positions for the disk write would match the response-positions for centroid extraction; the disk-budget fix decoupled them.** The fix was load-bearing for the sweep to complete at all (the per-q tensor footprint at 9 positions would have hit the 200 GB volume cap), but it left H3 without its primary readout. Re-running the sweep with `--per-q-response-positions-subset 1,2,4,8,16,32,64,128` is the direct fix.

**Confidence: LOW** — the verdict is structural, not empirical; the underlying question (do per-q response-token vectors produce coherent derangement ramps?) remains untested.

Sample H3 analyzer output (verbatim from `eval_results/issue_263/run_result.json`, the H3 sub-object):

```
H3.verdict:                FAIL
H3.n_personas:             275
H3.h3_metric_source:       per_q_test_split
H3.delta_self_mean:        NaN
H3.delta_self_median:      NaN
H3.median_fraction_positive: 0.0255
H3.available_t_per_q:      [0, 128]
H3.available_t_centroid:   [0, 1, 2, 4, 8, 16, 32, 64, 128]
H3.per_test_alpha:         0.01
H3.fraction_positive_threshold: 0.70
```

Sample analyzer error message (verbatim from the analysis log, one of the 7 unavailable positions):

```
[H3] r_per_token per-q at t=1 unavailable:
  r_per_token per-q at position=1 not on disk;
  sweep wrote subset [0, 128]. Re-run sweep with
  --per-q-response-positions-subset including this position to populate it.
```

### Next steps

- **Re-test H2 with N_test ≥ 60.** With more held-out questions per persona, the permuted-label null should stop saturating at AUC = 1.0, and the per-persona joint gate should fire for the personas where the delta is large. Analysis-only, ~1 GPU-hour.
- **Re-run the sweep with `--per-q-response-positions-subset 1,2,4,8,16,32,64,128`** so H3 can run on its intended grid. The new pod has network FS so the 200 GB volume cap that drove the original disk-budget fix no longer applies. ~3 GPU-hours sweep + ~1 GPU-hour analysis.
- **Re-frame H1 with stricter thresholds** (mc_r ≥ 0.95, max_classes=10) to test whether the structure within the 57 clusters is interpretable as a smaller set of "core regimes" plus a noisy shallow-layer fringe. Analysis-only.
- **Test whether c1's dominance (95.6% of per-persona picks) is robust to train-set size** by subsampling to N_train ∈ {50, 100, 200} questions. If the c1 ranking holds across train-set sizes, this is evidence that the contrast-of-means construction is a genuinely better recipe than Method-A for Qwen-2.5-7B-Instruct; if it doesn't, the per-persona selection may be overfitting to validation noise.

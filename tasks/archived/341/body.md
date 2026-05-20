---
title: Cosine and JS-divergence geometries align across 19 personas at L10, strengthening
  to ρ=0.94 at L20 (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-11T06:56:09.000Z'
has_clean_result: true
sagan_id: f95e8d92-334f-4970-b984-49cb0f1ebaa5
sagan_number: 341
priority: normal
legacy_why_unset: true
---
<details open>
<summary>

## TL;DR

</summary>

- Tested whether the cosine geometry between persona hidden states (L10) predicts the JS divergence geometry between persona-conditioned next-token distributions, on the project's canonical 19-persona / 20-prompt rig
- They agree at the population level, and the agreement gets stronger with depth (much cleaner at layer 20 than at layer 10)
- BUT if I restrict the null to swap labels only within occupational clusters (medical, security, etc), the L10 alignment vanishes -- much of the early-layer signal is just cluster structure (caveat clears at deeper layers)
- I guessed in advance that the most-decoupled cosine/JS pairs would be (helpful_assistant, comedian) and (helpful_assistant, poet) -- they aren't. The actual decouplings cluster around the (poet, comedian, villain) trio, which look similar at L10 but generate very different next-token distributions

</details>

<details open>
<summary>

## Summary

</summary>

- **Motivation:** Prior work in this repo found that JS divergence on teacher-forced next-token distributions predicts marker leakage better than hidden-state cosine ([#142](https://github.com/superkaiba/explore-persona-space/issues/142): ρ = −0.75 for JS vs +0.57 for cosine, n = 50 directed pairs on 11 personas / L20). That work also reported JS–cosine ρ = −0.74 on the same n = 55 unique pairs — but it was anchor-leveraged (the `no_persona` row contains the largest distances), confounded by a 4-persona medical cluster, and ran on only 11 personas at L20. Other geometry work ([#216](https://github.com/superkaiba/explore-persona-space/issues/216), [#237](https://github.com/superkaiba/explore-persona-space/issues/237), [#267](https://github.com/superkaiba/explore-persona-space/issues/267)) showed cosine geometry is internally consistent across layers but doesn't fully predict behavior — motivating the broader "do representation and behavior agree?" question. This experiment asks whether the agreement survives at the 19-persona population level once you (a) exclude the anchor, (b) partial out occupational clusters, and (c) partial out 1D radial structure (a "central" persona is close to everyone, an "extreme" persona is far from everyone). See [§ Background](#background).
- **Experiment:** We re-ran the project's canonical 20-persona × 20-prompt setup on `Qwen/Qwen2.5-7B-Instruct`, computed a 20×20 pairwise JS divergence matrix from teacher-forced next-token distributions (vLLM greedy generation under the `no_persona` anchor, then HF forward pass under each of 20 persona system prompts; chunked KL via `compute_pairwise_divergences(kl_only=True)`), and compared it to the layer-10 cosine matrix on the 19 non-anchor personas (n = 171 pairs). Six statistical gates at L10, plus robustness checks at L15/L20/L25. ~7 minutes wall-clock on 1× H100. See [§ Methodology](#methodology).
- **Results:**
  - **Representation and behavior agree at the population level** — ρ = 0.663 at L10 (one-sided Mantel p = 4.9 × 10⁻⁴, n = 171 pairs across n = 100K permutations); ρ rises substantially with depth, peaking at L20 (0.939) and easing back to 0.898 at L25. See [§ Result 1](#result-1-cosine-and-js-geometries-align-across-the-19-non-anchor-personas-rising-substantially-with-depth) and Figure 1.
  - **The alignment isn't just outlier-ness, but at L10 we can't separate it from cluster membership** — first we tested whether the agreement is driven by "some personas are outliers in both metrics" (each persona has an average distance to all others; that single scalar explains 88% of the cosine matrix). Subtracting out that outlier-ness from both matrices and re-correlating the residuals still gives ρ = 0.451 at L10, so the agreement is not just radial structure. Then we tested whether it's driven by occupational clusters: under a stricter null that permutes labels only WITHIN the medical / security / services / tech clusters (preserving cluster membership in both matrices), we can no longer detect the alignment at L10 (stratified Mantel p = 0.160 vs unstratified p = 4.9 × 10⁻⁴). The stratified test is power-limited though — only 11 within-cluster pairs of 171 — so this is consistent with "no signal beyond clusters" OR "real signal too small to detect at this n." A different operation, subtracting cluster membership directly from each matrix and correlating the residuals, gives ρ = 0.601 (very close to raw 0.663), which says most of the agreement survives. The two cluster tests answer slightly different questions; we live with the ambiguity at L10. See [§ Result 2](#result-2-the-alignment-survives-the-1d-radial-confound-but-leans-on-occupational-cluster-structure-at-l10) and Figure 2.
  - **The named-pair decoupling prediction did not pan out; decoupling clusters around the creative-persona trio (poet, comedian, villain) instead** — observed top-5 baseline-residual pairs at L10 are (poet, comedian), (software_engineer, helpful_assistant), (poet, villain), (helpful_assistant, kindergarten_teacher), (villain, comedian). The trio pairs sit at middle-of-distribution L10 cosine distance (ranks 81–118 of 171) but their JS divergence is much larger than the radial covariate predicts from their cosine distance, which is what flags them as residually decoupled. See [§ Result 3](#result-3-cosinejs-decouplings-cluster-around-the-creative-persona-trio-poet-comedian-villain-not-the-helpful-assistant-pairs-we-anticipated).
- **Takeaways:** Cosine geometry at layer 10 of Qwen-2.5-7B-Instruct does carry behavioral information — Spearman ρ on pairwise distances rises substantially with depth, peaking near 0.94 at L20 — but at the early layer the signal is power-limited within the occupational clusters that dominate the geometry, and the specific high-residual pair we anticipated (helpful_assistant ↔ comedian/poet) is not where the cosine-vs-behavior gap lives. The creative-persona trio (poet, comedian, villain) is the most consistent representation-vs-behavior-decoupled subset of the 19 personas across L10–L25.
- **Next steps:**
  - Re-run on a second model family (Llama or Gemma at 7B–13B scale) to test cross-model generalization of the L20 alignment and the creative-trio decoupling.
  - Re-run with a held-out prompt distribution to test prompt-set generalization.
  - Probe the creative-persona trio mechanistically — are these personas attaching shared style/stance vectors at deep layers that early-layer cosine can't see?
- **Confidence: MODERATE** — all six gates pass with margin at L10 and ρ rises substantially with depth (peaking at L20 = 0.939), but the stricter cluster-only null (permuting persona labels only within occupational clusters) cannot detect the L10 alignment at p < 0.05 (p = 0.160) and the named-pair decoupling prediction did not pan out — these two caveats prevent HIGH. Plus: single seed (n = 1), single prompt set (the 20 canonical PROMPTS), single model (Qwen-2.5-7B-Instruct), single anchor (`no_persona`) bound external validity until replicated with a fresh prompt distribution and/or a second model family.

</details>

## Details

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (~7.6B params, hidden_size 3584, 28 hidden layers, vocab_size 152064). No fine-tuning — inference + scoring only.
- **Personas (20 total; 19 in primary ρ):** `cybersec_consultant, pentester, software_engineer, data_scientist, helpful_assistant, private_investigator, medical_doctor, kindergarten_teacher, poet, villain, navy_seal, army_medic, surgeon, paramedic, police_officer, florist, librarian, comedian, french_person, no_persona`. `no_persona` is the anchor identity used to generate the shared response per prompt — its row/column is included in the 20×20 JS matrix but EXCLUDED from the n=171 primary Spearman ρ (because `cosine_matrix.json` computed its `no_persona` row under the Qwen default chat-template while JS computes the `no_persona` row under literal-empty-system ChatML — not apples-to-apples).
- **Prompts:** The 20 canonical prompts from `experiments/phase_minus1_persona_vectors/extract_persona_vectors.py::PROMPTS` (e.g., "What is the best way to learn a new language?"). Identical set used to build `cosine_matrix.json`.
- **Pipeline:** (1) Generate one greedy 256-token response per prompt under `no_persona` via vLLM (temp=0.0, top_p=1.0, seed=42). (2) For each prompt × 20 personas, teacher-force the shared response through Qwen-2.5-7B-Instruct under each persona's system prompt; collect log-softmax tensors (T, V=152064). (3) Compute 20×20 pairwise JS via chunked KL approximation, `kl_only=True`, validated at ρ ≥ 0.97 vs exact JS on a 10-persona / 3-prompt held-out subset. Aggregate per-prompt JS matrices by mean across the 20 prompts. (4) RSA: Spearman ρ on `triu(1 − cosine_L10, k=1)` vs `triu(JS, k=1)` over the n = 171 pairs of the 19 non-anchor personas. (5) Mantel one-sided upper-tail test, B = 100K permutations, `p = (b+1)/(B+1)`; floor ≈ 1e-5.
- **Cosine matrix file:** `experiments/phase_minus1_persona_vectors/cosine_matrix.json`, sha256 `c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f` (verified at run time).
- **Layers measured:** 10 (confirmatory headline), 15 / 20 / 25 (exploratory — same gating signature reported per layer).
- **Cluster definitions used by the analysis:** medical = {medical_doctor, surgeon, paramedic, army_medic}; security = {cybersec_consultant, pentester, private_investigator}; services = {navy_seal, police_officer}; tech = {software_engineer, data_scientist}; civilian singletons = {helpful_assistant, kindergarten_teacher, poet, villain, florist, librarian, comedian, french_person}. 11 within-fine-cluster pairs of the 171. Macro clusters = {occupational, civilian-singletons}.
- **Six gating thresholds used to bind the headline:** raw ρ > 0.5; one-sided Mantel p < 0.001; joint cluster-partial ρ > 0.4; mean-marginal-baseline-residual ρ > 0.2; per-prompt median ρ ≥ 0.2; T=8/T=full ratio ≥ 0.3. All six must PASS at L10 for the headline to stand.
- **Named-pair decoupling check (strict 2-of-2):** the top-5 baseline-residual pairs (residuals from the `b_mean_marginal` regression) were specified in advance to include BOTH (comedian, helpful_assistant) AND (poet, helpful_assistant). Strict — no partial credit.
- **Code:** [`scripts/analyze_persona_geometry_vs_divergence.py`](https://github.com/superkaiba/explore-persona-space/blob/4ddf33d6/scripts/analyze_persona_geometry_vs_divergence.py) @ commit `4ddf33d6`. JS computation reuses `src/explore_persona_space/analysis/divergence.py::compute_pairwise_divergences`.
- **Compute:** ~7 min wall-clock (420.1 s, of which ~3 min teacher-forcing + ~4 min Mantel/gates/sensitivities) on 1× H100 (RunPod pod `epm-issue-269`).
- **Logs / artifacts:** Raw JSON at `eval_results/issue_269/{geometry_alignment.json, js_matrix.json, generations.json, run_meta.json}` on branch `issue-269` @ `4ddf33d6`. Re-extracted centroids at `eval_results/issue_269/centroids_re_extracted.pt` (1.1 MB, local-only — not committed to repo). No WandB run for this experiment (analysis-only, single H100 inference job).
- **Full data:** [`eval_results/issue_269/geometry_alignment.json`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4ddf33d6/eval_results/issue_269/geometry_alignment.json) (full 6-gate × 4-layer × 19-jackknife signature, 39 KB) and [`eval_results/issue_269/js_matrix.json`](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4ddf33d6/eval_results/issue_269/js_matrix.json) (20×20 JS matrices at T=8/32/full, 251 KB). The source cosine matrix is [`experiments/phase_minus1_persona_vectors/cosine_matrix.json`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data) (Qwen-2.5-7B-Instruct centroid cosine, sha256 `c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f`).
- **Pod / environment:** `epm-issue-269` (RunPod, 1× H100, `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`). Python 3.11, torch 2.8.0+cu128, transformers 4.57.6, vllm 0.11.0, scipy 1.17.1, sklearn 1.8.0, numpy 2.2.6.

</details>

<details open>
<summary>

### Background

</summary>

A central question in this repo's persona-space program is whether the *representational* geometry of personas (hidden-state similarity inside the model) predicts the *behavioral* geometry (how differently two persona-conditioned models respond to the same prompts). [#142](https://github.com/superkaiba/explore-persona-space/issues/142) suggested they agree on a small slice: 11 personas × 20 prompts at L20 gave JS-vs-cosine Spearman ρ = −0.74 on n = 55 unique pairs (sign-flipped: 1−cos ≈ JS), and JS predicted marker leakage better than cosine (ρ = −0.75 vs +0.57). But that result was anchor-leveraged — the `no_persona` row of the cosine matrix contains the largest pairwise distances — and confounded by a 4-persona medical cluster that lives at very small mutual cosine distance.

Subsequent geometry experiments narrowed the question. [#216](https://github.com/superkaiba/explore-persona-space/issues/216) showed cosine geometry is layer-consistent: pairs that are close at one layer tend to be close at others. [#237](https://github.com/superkaiba/explore-persona-space/issues/237) showed SFT compresses the cosine matrix to near-uniform ≥ 0.97 cosine — whatever geometry exists is on the base model, not the fine-tuned model. [#267](https://github.com/superkaiba/explore-persona-space/issues/267) showed L20 centroid steering fails to fully reproduce behavior — cosine doesn't capture everything behavioral.

This experiment scales [#142](https://github.com/superkaiba/explore-persona-space/issues/142)'s test to the project's full 20-persona × 20-prompt rig on the base `Qwen/Qwen2.5-7B-Instruct` model, drops the anchor from the headline (n = 171 pairs of 19 personas), and adds three nuisance controls: a 1D-radial-structure partial (a persona that's close to everyone vs far from everyone), a within-occupational-cluster Mantel stratification, and a joint partial against both cluster-fine + cluster-macro indicators. The question this issue answers: do the representational and behavioral geometries of personas agree at the 19-persona population level, beyond what the obvious confounds explain?

</details>

<details open>
<summary>

### Methodology

</summary>

The pipeline is unchanged from [#142](https://github.com/superkaiba/explore-persona-space/issues/142) in spirit but scaled and de-confounded. For each of the 20 canonical prompts (e.g., *"What is the best way to learn a new language?"*), we generate one greedy 256-token response under the `no_persona` anchor identity via vLLM (temp = 0, top_p = 1, seed = 42). We then teacher-force that shared response through `Qwen/Qwen2.5-7B-Instruct` under each of the 20 persona system prompts, collecting (response_len, vocab_size=152064) log-softmax tensors. The chunked KL approximation in `compute_pairwise_divergences(kl_only=True)` produces a 20×20 JS matrix per prompt; the per-prompt matrices are averaged into a single JS matrix. Validation: on a random 10-persona × 3-prompt subset (45 pairs each), the kl_only approximation matches exact JS at ρ ∈ {0.972, 0.981, 0.982}, well above the 0.95 abort threshold. JS-bound check: `max(JS) = 0.477 < ln 2 ≈ 0.693`, no WARN.

The cosine matrix at L10 (and L15, L20, L25) comes pre-computed from `experiments/phase_minus1_persona_vectors/cosine_matrix.json` (sha256 verified at run time). RSA is Spearman ρ on the vectorized upper triangles, computed on the n = 171 off-diagonal pairs of the 19 non-anchor personas (`no_persona` excluded — its cosine row was computed under the Qwen default chat-template while its JS row is under literal-empty-system ChatML, so the n = 190 RSA is reported only as a labeled secondary robustness number).

A representative anchor response (the input to the JS computation for that prompt):

```
Prompt: "What is the best way to learn a new language?"

no_persona response (greedy, 256 tokens):
"Learning a new language can be a rewarding and enriching experience. Here are some effective strategies to help you learn a new language:

1. **Set Clear Goals**: Define what you want to achieve with the language. Are you learning for travel, work, or personal interest? Setting specific, measurable, achievable, relevant, and time-bound (SMART) goals can help you stay motivated.

2. **Immerse Yourself**: Surround yourself with the language as much as possible. Watch movies, TV shows, and listen to music in the target language. Try to read books, newspapers, and articles. If possible, travel to a country where the language is spoken.
..."
```

This single response is what we teacher-force through each of the 20 persona system prompts; the JS divergence between two persona-conditioned log-prob tensors over those response tokens IS the behavioral distance metric.

</details>

<details open>
<summary>

### Result 1: Cosine and JS geometries align across the 19 non-anchor personas, rising substantially with depth

</summary>

For the 19 non-anchor personas, we compute Spearman ρ between `vec(1 − cosine_L)` and `vec(JS)` over the n = 171 off-diagonal pairs at each of L10 (confirmatory), L15, L20, L25 (exploratory). The hero figure below shows the two matrices side-by-side under hierarchical clustering ordering (left column) and alphabetical ordering (right column) so the shared blockwise structure is visible by eye.

![Hero figure — four-panel dual heatmap of 1-cosine at layer 10 and JS divergence at T=full, each shown under hierarchical clustering ordering and alphabetical ordering, n=19 non-anchor personas](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4ddf33d6/figures/issue_269/hero_dual_heatmap_n19.png)

**Figure 1.** *Hidden-state cosine (top row, L10) and Jensen–Shannon next-token divergence (bottom row) over the same 19 personas — the two pairwise geometries share clear blockwise structure (especially the medical cluster: army_medic / paramedic / surgeon / medical_doctor at small distance in both panels), and creative-persona outliers (poet, comedian, helpful_assistant) show up as high-distance rows/columns in both.* Left column: hierarchical clustering ordering (average linkage on 1−cosine); right column: alphabetical ordering. Color scale per panel; viridis. The blockwise patterns visible in both rows are what Spearman ρ on the vectorized upper triangle (n = 171 pairs) quantifies — at L10, ρ = 0.663 (one-sided Mantel p = 4.9 × 10⁻⁴, n = 100K permutations).

**Main takeaways:**
- **All six gating statistics PASS at L10 with margin** — raw ρ = 0.663 (threshold > 0.5), Mantel p = 4.9 × 10⁻⁴ (threshold < 0.001, n = 100K perms), joint cluster-partial ρ = 0.601 (threshold > 0.4), mean-marginal-baseline-residual ρ = 0.451 (threshold > 0.2), per-prompt median ρ = 0.623 with 18/20 prompts above 0.5 (threshold ≥ 0.2; the two prompts below 0.5 are prompt 7 at ρ = 0.497 and prompt 19 at ρ = 0.462 — both pass the gate threshold of 0.2 but the reader should know two prompts individually fall below 0.5), T=8/T=full ratio = 0.93 (threshold ≥ 0.3). The headline is robust to the four named confounds (anchor leverage, cluster structure, 1D radial structure, prompt-level idiosyncrasy).
- **ρ rises substantially with depth, peaking at L20** — L10 = 0.663, L15 = 0.921, L20 = 0.940, L25 = 0.898 (all four layers: Mantel p < 1e-5 at n = 100K perms; n = 171). The lower L10 number is consistent with L10's representational space being 5.7× more rank-compressed than L20 (L10 cosine-distance std = 0.014 on n = 171; L20 std = 0.080); moderate L10 ρ in absolute terms reflects noise-to-signal at the early layer, not "wrong layer for behavior."
- **Leave-one-persona-out jackknife at L10 stays well above the 0.5 gate but has a wide range** — median ρ = 0.649, IQR = 0.033, range = [0.630 (dropping paramedic), 0.772 (dropping helpful_assistant)] across n = 19 drops. The IQR is narrow (0.033) but the range is 0.142, so dropping the right persona moves ρ by ~22% of its absolute value — the median is robust, the tails are not. The single largest mover is helpful_assistant (dropping it raises ρ by +0.11) — helpful_assistant is dragging L10 ρ DOWN, not propping it up. At L20 the jackknife range collapses to 0.022 (no single persona moves ρ by more than 0.022).

```
L10 gating signature (n = 171 pairs):
  raw ρ                          = 0.663   (gate: > 0.5)        ✓
  one-sided Mantel p             = 4.9e-4  (gate: < 0.001)      ✓
  joint cluster-partial ρ        = 0.601   (gate: > 0.4)        ✓
  resid ρ | b_mean_marginal      = 0.451   (gate: > 0.2)        ✓
  per-prompt median              = 0.623   (gate: ≥ 0.2)        ✓
  T=8 / T=full ratio             = 0.930   (gate: ≥ 0.3)        ✓
  → all_six_gates_pass = True
```

Three sample "firing" pairs — pairs whose cosine rank and JS rank agree closely (rank-difference < 5 of 171), drawn directly from the JS matrix and the L10 cosine matrix:

```
(surgeon, paramedic):                 1-cos(L10) = 0.0047 (rank 1/171),    JS = 0.0162 (rank 3/171)     -- close in both (within-medical-cluster)
(software_engineer, data_scientist):  1-cos(L10) = 0.0071 (rank 3/171),    JS = 0.0136 (rank 1/171)     -- close in both (within-tech-cluster)
(helpful_assistant, poet):            1-cos(L10) = 0.0777 (rank 171/171),  JS = 0.4771 (rank 171/171)   -- far in both (top of both rankings)
```

Three sample "non-firing" pairs — pairs whose cosine rank disagrees with their JS rank (these are the baseline-residual outliers Result 3 unpacks; cosine ranks are middle-of-distribution, JS ranks sit in the top quartile):

```
(poet, comedian):     1-cos(L10) = 0.0205 (rank 103/171, middle),       JS = 0.2762 (rank 140/171, top quartile)
(poet, villain):      1-cos(L10) = 0.0240 (rank 118/171, above middle), JS = 0.2697 (rank 139/171, top quartile)
(villain, comedian):  1-cos(L10) = 0.0183 (rank  81/171, middle),       JS = 0.2286 (rank 137/171, top quartile)
```

The n = 190 secondary RSA (including `no_persona`) gives ρ = 0.559 at L10, p = 3.1e-3 — directionally consistent but interpretation-noise-bounded by the ChatML mismatch on that single row/column.

</details>

<details open>
<summary>

### Result 2: The alignment survives the 1D radial confound but leans on occupational cluster structure at L10

</summary>

Two confounds worry us about the raw ρ = 0.663 at L10. The first is **outlier-ness** (a "1D radial" structure): some personas sit far from EVERYONE on average, and that single per-persona scalar can dominate the pairwise distances. If both metrics encode the same outlier-ness pattern but nothing else, you'd see high ρ for trivial reasons. We test this two ways: `b_mean_marginal` = each persona's mean distance to all 18 others (this explains 88% of pairwise cosine variance — strong 1D structure), and a weaker `b_no_persona` baseline = each persona's distance to just the anchor persona. The second confound is **occupational cluster membership**: medical personas cluster together in BOTH matrices for an obvious reason (shared medical vocabulary), which could dominate the apparent agreement even if cross-cluster geometry is random. We test this two ways: (a) directly residualize against cluster indicators (one binary same-fine-cluster, one binary same-macro-cluster) via two-covariate partial Spearman, and (b) re-run the Mantel significance test under a stricter null that permutes labels ONLY within clusters (the within-cluster-stratified Mantel test).

![Sensitivity bar chart — seven RSA statistics (raw ρ, cluster-mask ρ, partial ρ on cluster_fine, joint partial ρ on cluster_fine+cluster_macro, mean-marginal-baseline-residual ρ, no_persona-baseline-residual ρ, cluster-collapsed ρ on n=66) at each of L10, L15, L20, L25](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4ddf33d6/figures/issue_269/rho_by_layer_7stat.png)

**Figure 2.** *Sensitivity of Spearman ρ to seven nuisance controls across four layers — at L10 the alignment survives 1D-radial-structure correction (mean-marginal-baseline-residual ρ = 0.451, gating threshold > 0.2) but a non-trivial portion of the signal is cluster-structure-driven (cluster-collapsed ρ on n = 66 = 0.483, vs raw 0.663 on n = 171).* Each bar = ρ for one statistic at one layer; dashed line at ρ = 0.5 marks the headline ρ > 0.5 threshold; small-print p-values annotate the raw-ρ bars. Joint partial (column 4) is the GATING statistic at L10 (0.601 > 0.4 ✓). At L20 every statistic stays above 0.91; L15 stays above 0.85 except for b_no_persona-residual ρ = 0.824 (the weaker anchor-based proxy); at L25 cluster-collapsed-n=66 dips to 0.807 (visible as the shortest pink bar in the L25 column).

**Main takeaways:**
- **The agreement isn't just outlier-ness** — Some personas (helpful_assistant, poet, comedian) sit far from EVERY other persona; others sit close to most. Each persona has a single "outlier-ness" scalar (= its average distance to all others), and that scalar alone explains 88% of pairwise cosine variance. Subtracting the outlier-ness prediction from BOTH matrices and re-correlating the residuals still gives ρ = 0.451 at L10 (gate > 0.2). The agreement is not just "both metrics agree on which personas are outliers."
- **But at L10 we can't separate the agreement from cluster membership under the strict null** — We re-ran the significance test with a stricter null: instead of permuting persona labels freely (the standard Mantel test), permute them ONLY within the medical / security / services / tech clusters (preserving "this persona is medical" / "this persona is tech" in both matrices). Under this stricter null we cannot detect the alignment at L10 (stratified Mantel p = 0.160 vs unstratified p = 4.9 × 10⁻⁴). The test is power-limited though — only 11 within-cluster pairs (medical 6, security 3, tech 1, services 1) — so this is consistent with "the agreement is purely cluster-membership" OR "real signal beyond clusters but too small to detect at n = 11." This is the binding caveat at L10.
- **A different cluster-subtraction operation says most of the signal survives** — Instead of permuting under a stricter null, we directly residualize: subtract from each pair's distance what cluster-membership indicators predict (one binary for same-fine-cluster, one for same-macro-cluster), then correlate the residuals. That gives ρ = 0.601 vs raw 0.663 — only ~10% of the signal is cluster-membership-explained. A third check, replacing each fine cluster with its centroid and re-correlating the 66-pair reduced matrices, gives ρ = 0.483 (still positive but well below raw). The partial-correlation operation answers "how much variance survives subtracting cluster?" → most of it. The stratified-Mantel operation answers "can we statistically distinguish from cluster-only at p < 0.05?" → not at L10's n.
- **HA-excluded sensitivity goes UP, not down** — dropping helpful_assistant raises raw ρ from 0.663 to 0.772 (∆ = +0.109) and joint cluster-partial from 0.601 to 0.725. Helpful_assistant is an L10 cosine outlier — it has the highest mean distance to every other persona (it is the FARTHEST persona from `no_persona`, contra the surface-level "Qwen-default = helpful_assistant baseline" intuition) but not a JS outlier. Excluding it reveals a stronger underlying alignment.
- **These caveats resolve at L20 and L25; L15 is borderline** — at L20 (the layer [#142](https://github.com/superkaiba/explore-persona-space/issues/142) used) every sensitivity stays above 0.91 and stratified-Mantel p = 0.0017; at L25 stratified-Mantel p = 0.0019. At L15 stratified-Mantel p = 0.0417 is marginal (close to the 0.05 line), so the "caveat-free" claim specifically applies to L20-L25 — L15 weakens the caveat but doesn't fully clear it. The L10 binding caveat is specific to the early-layer representational space, where the 5.7× rank-compression makes cluster structure a larger fraction of the explainable variance.

Three "robustness-firing" L10 sensitivity rows — the alignment survives these controls:

```
mean-marginal-baseline-residual ρ (L10, n=171, after partialling 88% of cos variance) = 0.4505
joint cluster-partial ρ (L10, n=171, both cluster_fine and cluster_macro covariates)  = 0.6008
HA-excluded raw ρ (L10, n=153)                                                         = 0.7717
```

Three "robustness-non-firing" L10 rows — where the alignment weakens or vanishes:

```
within-cluster-stratified Mantel p (L10, n=100K perms, p-floor ≈ 1/576 ≈ 0.0017)   = 0.1599
cluster-collapsed ρ (L10, n=66, fine-cluster centroids replace within-cluster pairs) = 0.4826
b_no_persona-baseline-residual ρ (L10, anchor-based 1D-radial covariate)             = 0.3343
```

</details>

<details open>
<summary>

### Result 3: Cosine/JS decouplings cluster around the creative-persona trio (poet, comedian, villain), not the helpful-assistant pairs we anticipated

</summary>

The third specified test asked: of the 171 pairs, which 5 have the LARGEST mismatch between their cosine distance (representational geometry) and JS divergence (behavioral geometry), after controlling for the outlier-ness confound from Result 2? We anticipated this top-5 list would contain BOTH (comedian, helpful_assistant) AND (poet, helpful_assistant) — a strict 2-of-2 conjunction. Rationale: helpful_assistant + poet + comedian are the three personas with the largest mean distance to everyone else; if you just took raw cosine-vs-JS residuals, those pairs would top the list by construction (high outlier-ness → high pair distance regardless of mechanism). So we measured residuals AFTER subtracting the outlier-ness baseline — the residual flags pairs whose JS divergence is much LARGER than their outlier-ness alone would predict.

The observed top-5 baseline-residual pairs at L10 (residual magnitude in z-score units, summing the cosine and JS deviation z-scores):

| Rank | Pair | Residual magnitude | 1 − cos (L10) | JS | Same-cluster? |
|---|---|---|---|---|---|
| 1 | (poet, comedian) | 10.57 | 0.020 | 0.276 | No (both civilian-singletons) |
| 2 | (software_engineer, helpful_assistant) | 6.08 | 0.033 | 0.043 | No (tech ↔ singleton) |
| 3 | (poet, villain) | 5.04 | 0.024 | 0.270 | No (both civilian-singletons) |
| 4 | (helpful_assistant, kindergarten_teacher) | 4.90 | 0.038 | 0.164 | No (both civilian-singletons) |
| 5 | (villain, comedian) | 4.55 | 0.018 | 0.229 | No (both civilian-singletons) |

The anticipated conjunction did not hold: neither (comedian, helpful_assistant) nor (poet, helpful_assistant) appears in the observed top-5. The scatter below shows where these outliers sit on the cosine–JS plane.

![Scatter — 1-cosine at layer 10 vs JS divergence at T=full, n=171 non-anchor persona pairs, color-coded by macro/fine cluster membership, top-5 baseline-residual pairs labeled inline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4ddf33d6/figures/issue_269/cosine_vs_js_scatter_n171.png)

**Figure 3.** *Scatter of pairwise cosine distance (x, L10) vs JS divergence (y, T=full) over the 171 non-anchor persona pairs, with the top-5 baseline-residual outliers circled and labeled inline.* Blue diamonds = within-fine-cluster pairs (n = 11, dense low-distance cluster bottom-left); purple = cross-fine but same-macro-cluster (n = 72); orange = cross-macro (n = 88). The (poet, comedian) point is the most decoupled: 1-cos = 0.020 (rank 103/171 = 60th rank-percentile, middle-of-distribution) but JS = 0.276 (rank 140/171 = 82nd rank-percentile, top quartile). The (helpful_assistant, comedian) and (helpful_assistant, poet) pairs we anticipated are NOT circled (they did not enter the observed top-5 baseline-residual list); they sit as un-labeled orange points near the upper-right of the cluster (cos rank 169 and 171, JS rank 163 and 171 respectively).

**Main takeaways:**
- **The strict 2-of-2 named-pair conjunction did not hold** — neither (HA, comedian) nor (HA, poet) appears in the observed top-5 baseline-residuals; the top-1 is (poet, comedian) with residual magnitude 10.57 vs the (HA, kindergarten_teacher) pair at 4.90. The mechanism we predicted — helpful_assistant as the central cosine outlier driving cosine/JS decouplings — does not match the data.
- **Decouplings DO concentrate, but around the creative-persona trio** — (poet, comedian), (poet, villain), and (villain, comedian) are 3 of the top-5 baseline-residual pairs at L10. The trio pairs sit at *middle-of-distribution* L10 cosine distance (ranks 81–118 of 171, NOT "small") and top-quartile JS divergence (ranks 137–140 of 171). The decoupling these residual magnitudes flag isn't "cosine sees them as close" — it's "JS divergence is much larger than the radial-confound covariate predicts from their cosine distance." Mechanically: helpful_assistant + poet + comedian + villain are the four highest-mean-marginal personas (they sit farthest from every other persona on average), so the radial covariate predicts a LARGE pairwise distance — but the trio's actual L10 cosine is moderate (pulled toward mutual closeness by shared creative-persona structure), making the residual JS-vs-cosine gap stand out. This is a non-trivial decoupling pattern but it's not what the strict named-pair check anticipated.
- **The decoupling pattern survives layer change** — at L20 and L25 the same trio (poet, comedian, villain) plus their pairwise combinations occupy 3-5 of the top-5 baseline-residuals; (poet, comedian) is top-1 at every layer L10/L15/L20/L25. The creative-persona trio is the most representation-vs-behavior-decoupled subset of the 19 personas across the entire layer range.
- **Exploratory CKA is uncalibrated** — linear CKA between (19, 3584) L10 centroids and the (19, 19) JS-similarity matrix is 0.101 (Kornblith et al. 2019 explicitly cautions against inference at n = 19); reported as a number but no headline weight.

The three "decoupling-firing" creative-trio pairs across all four layers (residual magnitude in z-score units; pair rank within top-5 baseline-residuals at each layer):

```
(poet, comedian)    -- L10 #1 (10.57)  L15 #1 (15.04)  L20 #1 (13.54)  L25 #1 (11.63)
(poet, villain)     -- L10 #3 (5.04)   L15 #3 (5.93)   L20 #3 (7.31)   L25 #3 (7.39)
(villain, comedian) -- L10 #5 (4.55)   L15 #2 (6.43)   L20 #2 (8.33)   L25 #2 (8.45)
```

The three "decoupling-non-firing" named pairs — those that DIDN'T appear in the observed top-5 baseline-residuals at any layer:

```
(comedian, helpful_assistant) -- anticipated top-5; observed: NOT in top-5 at L10/L15/L20/L25
(poet, helpful_assistant)     -- anticipated top-5; observed: NOT in top-5 at L10/L15/L20/L25
(software_engineer, helpful_assistant) -- L10 #2 (6.08), then drops out of top-5 at L15+
```

</details>

<details open>
<summary>

## Source issues

</summary>

This clean-result distills evidence from:

- [#142](https://github.com/superkaiba/explore-persona-space/issues/142) — established that JS divergence predicts marker leakage better than cosine on the 11-persona / 20-prompt slice at L20 (ρ = −0.74 JS-vs-cosine on n = 55); this experiment scales that test to 19 personas at L10 with anchor and cluster controls.
- [#216](https://github.com/superkaiba/explore-persona-space/issues/216) — showed cosine geometry is internally consistent across L10/L15/L20/L25; motivates reporting the same RSA signature per layer.
- [#237](https://github.com/superkaiba/explore-persona-space/issues/237) — showed SFT compresses cosine to near-uniform ≥ 0.97; this experiment runs on the base model, so the SFT-compression caveat doesn't bite.
- [#267](https://github.com/superkaiba/explore-persona-space/issues/267) — showed L20 centroid steering fails to fully reproduce behavior; that observation motivates the broader "do representation and behavior agree?" question this issue answers at the population level.

</details>

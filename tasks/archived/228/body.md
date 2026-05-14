---
title: Output-space distance between persona system prompts predicts cross-prompt
  marker leakage on convergence-trained Qwen-2.5-7B, while residual-stream similarity
  instead tracks within-source training trajectories (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-05-04T19:04:12.000Z'
has_clean_result: true
sagan_id: 2814da05-718e-4f0f-bd8a-22fec06b3a2e
sagan_number: 228
priority: normal
---
## Human TL;DR

_(Human TL;DR — to be filled in by the user. Leave this line as-is in drafts.)_

## AI TL;DR (human reviewed)

Output-space distance between persona system prompts predicts cross-prompt marker leakage on convergence-trained Qwen-2.5-7B, while residual-stream similarity instead tracks within-source training trajectories.

In detail: on 71 directed (source, target) cells at convergence epoch 20, output-space JS divergence between source-prompt and target-prompt next-token distributions predicts marker leakage at Spearman rho=-0.65 (p=6.8e-10, n=71); the partial-correlation rho(JS, leak | cosine_L15)=-0.54 (p=1.7e-6, n=71) survives while the reverse partial collapses to rho=-0.06 (p=0.62), so JS carries information cosine does not — replicating [#142](https://github.com/superkaiba/explore-persona-space/issues/142)'s base-model finding (rho=-0.75, n=50) on convergence-trained checkpoints; longitudinally within-source, the FDR bar fails (raw 2/7 sources, first-difference 0/7) even though 5/7 within-source rho_JS values are negative in sign and within-source cosine_L20 is positive for 7/7 sources, suggesting JS indexes the cross-state equilibrium and cosine indexes the per-source trajectory.

- **Motivation:** Prior work in this repo ([#142](https://github.com/superkaiba/explore-persona-space/issues/142), [#109](https://github.com/superkaiba/explore-persona-space/issues/109), [#91](https://github.com/superkaiba/explore-persona-space/issues/91)) studied predictors of cross-prompt marker leakage on either the unfinetuned base model or a single convergence-training sweep slice; we wanted to test whether the base-model finding "output-space JS divergence beats representation-space cosine as a leakage predictor" generalizes to convergence-trained checkpoints across the full 7-source × 10-checkpoint adapter sweep. See [§ Background](#background).
- **Experiment:** Single seed=42; merged 71 LoRA adapter states (7 sources × 10 checkpoint steps + 1 shared epoch-0 baseline) on `Qwen/Qwen2.5-7B-Instruct`, computed pairwise output-space JS divergence (vLLM greedy + HF teacher-forcing in bf16, fp32 log-softmax) and residual-stream cosine at layers 15/20/25 over 11 personas × 20 questions, and correlated against per-cell marker leakage rates inherited from [#109](https://github.com/superkaiba/explore-persona-space/issues/109).
- **Cross-sectionally, JS beats cosine; the partial-correlation test confirms subsumption** — JS rho=-0.65 (p=6.8e-10, n=71) vs cosine_L15 rho=+0.44 (p=0.0001, n=71); partial rho(JS | cosine_L15)=-0.54 (p=1.7e-6); reverse partial rho(cosine_L15 | JS)=-0.06 (p=0.62). See [§ Result 1](#result-1-js-divergence-predicts-cross-sectional-marker-leakage-and-the-partial-correlation-test-confirms-it-subsumes-cosine-l15) and Figure 1.
- **Within-source longitudinally, JS fails the FDR bar but the directional pattern is mostly cross-section-consistent** — within-source raw passes 2/7 sources (librarian rho_JS=-0.90, software_engineer rho_JS=-0.75); 5/7 within-source rho_JS values are negative in sign; comedian flips to rho_JS=+0.93 (p=1.1e-4, n=10); within-source cosine_L20 is positive for 7/7 sources (mean +0.55), partially replicating [#91](https://github.com/superkaiba/explore-persona-space/issues/91). See [§ Result 2](#result-2-within-source-longitudinally-js-fails-fdr-but-cosine-l20-takes-over-the-trajectory).
- **Confidence: MODERATE** — the cross-sectional signal is well-powered (n=71, p=6.8e-10), survives within-target z-score robustness (rho=-0.70) and villain-out sensitivity (rho=-0.67, n=61), and the partial-correlation test is significant (p=1.7e-6); binding constraints are single seed (=42), the paired Δ|rho| margin between JS and cosine_L15 is not significantly different from zero, and the within-source longitudinal test fails the FDR bar at n=11/source.

## AI Summary

<details>
<summary><b>Setup details</b> — model, dataset, code, load-bearing hyperparameters, logs / artifacts. Expand if you need to reproduce or audit.</summary>

- **Model:** `Qwen/Qwen2.5-7B-Instruct` (7.6B params) + 71 merged LoRA states (7 sources × 10 checkpoint steps from [#109](https://github.com/superkaiba/explore-persona-space/issues/109) at steps 200, 400, ..., 2000 mapping to convergence epochs 2, 4, ..., 20; 1 shared epoch-0 baseline). LoRA config: r=32, alpha=64, dropout=0.0 (villain) / 0.05 (others), targets=all 7 linear layers; frozen for inference, merged per-state.
- **Dataset:** 11 personas × 20 questions from `ALL_EVAL_PERSONAS` + `EVAL_QUESTIONS` in `src/explore_persona_space/personas.py`. Per-state JS computed on 11×11 directed (source-prompt, target-prompt) cells (self-loops removed, 71 off-diagonal cells aggregated at epoch 20 for cross-section and partial-correlation tests). Marker leakage rates inherited from [#109](https://github.com/superkaiba/explore-persona-space/issues/109)'s `causal_proximity_strong_convergence_v1` WandB artifact (no recomputation).
- **Code:** [`scripts/run_issue228_sweep.py`](https://github.com/superkaiba/explore-persona-space/blob/issue-228/scripts/run_issue228_sweep.py) @ commit `ad972db` (per-state worker `scripts/compute_js_convergence_228.py`; aggregator `scripts/aggregate_issue228.py`; plot script `scripts/plot_issue228_figures.py`).
- **Hyperparameters:** seed=42; vLLM greedy generation (temp=0, top_p=1.0, max_tokens=512); HF teacher-forcing in bf16 with fp32 log-softmax; JS computed per-token then mean-reduced over response tokens then mean over 20 questions per pair; cosine computed on residual-stream centroids at layers 15/20/25, global-mean centered; leakage = % of 200 completions per state (11 personas × 20 questions × 10 completions, reported per-target as % of 200) containing `[ZLT]` substring (case-insensitive, temp=1, inherited from [#109](https://github.com/superkaiba/explore-persona-space/issues/109)). Partial Spearman via `pingouin.partial_corr(method='spearman')`. Stats: rho + p + n reported in prose; cluster-resampling intervals (1000 iter, 2-stage source-to-target) computed in JSON for kill-criterion adjudication only.
- **Compute:** ~17h wall on `epm-issue-228` (1× H100 SXM 80GB initial — 8× H100 unavailable due to RunPod supply, fallback per plan §11; 8× came online mid-sweep). Total ~150 GPU-hours.
- **Logs / artifacts:** WandB project [`thomasjiralerspong/issue228`](https://wandb.ai/thomasjiralerspong/issue228) (artifact `causal_proximity_strong_convergence_v1`, type `eval-results`, 257 MB); compiled results at `eval_results/issue_228/all_results.json`; per-run results at `eval_results/issue_228/<source>/checkpoint-<step>/result.json` (71 files); correlations + cluster-resampling intervals at `eval_results/issue_228/correlations.json`; raw generations in the WandB artifact (also at `data/leakage_v3_onpolicy/onpolicy_cache/completions_<source>.json` on the pod). Adapters at HF Hub [`superkaiba1/explore-persona-space`](https://huggingface.co/superkaiba1/explore-persona-space) (subpaths `adapters/issue112_convergence/villain_s42` + `adapters/cp_armB_strong_<source>_s42`, 77 unique adapters: 7 ep0 + 70 ep>0).
- **Pod / environment:** `epm-issue-228`; Python 3.11; transformers>=4.48, torch>=2.4, peft=0.18.1, vllm>=0.8, scipy>=1.15, statsmodels>=0.14, pingouin=0.5.4. Git commit `ad972db` (issue-228 branch; PR [#242](https://github.com/superkaiba/explore-persona-space/pull/242)).

</details>

### Background

Persona-trained LMs sometimes "leak" trained behaviors when prompted with bystander system prompts — a safety-relevant generalization phenomenon. Prior work in this repo studied predictors of that leakage from two complementary angles. [#142](https://github.com/superkaiba/explore-persona-space/issues/142) showed that, on the unfinetuned base model `Qwen/Qwen2.5-7B-Instruct`, output-space JS divergence between system-prompt next-token distributions predicts marker leakage substantially better than representation-space cosine similarity (rho=-0.75 vs cosine-L20 rho=+0.57, n=50; partial Spearman rho(JS, leakage | cosine)=-0.61, p=4.1e-6, indicating JS subsumes cosine on the base model). [#109](https://github.com/superkaiba/explore-persona-space/issues/109) ran a 7-source × 10-checkpoint convergence-training sweep and found that residual-stream cosine similarity at layer 15 did NOT predict per-source peak leakage on the n=7 source-to-assistant slice (rho=-0.34, p=0.45), leaving open whether JS — which beat cosine on the base model — would also beat cosine on convergence-trained checkpoints. [#91](https://github.com/superkaiba/explore-persona-space/issues/91) supplied the within-source longitudinal framing (cosine *decreases* leakage within source) that motivates the within-source trajectory test in this issue.

This experiment closes the loop: compute output-space JS on every available [#109](https://github.com/superkaiba/explore-persona-space/issues/109) adapter state (7 sources × 10 checkpoint steps + 1 shared epoch-0 baseline = 71 model states), and correlate against the *already-published* [#109](https://github.com/superkaiba/explore-persona-space/issues/109) leakage rates — avoiding a fresh leakage eval that would have doubled the GPU budget. Three tests motivate the design: a cross-sectional test (does JS predict leakage at |rho|>0.5 negative on 71 off-diagonal cells?), a within-source longitudinal test (does JS predict leakage on ≥4/7 sources at FDR-PASS?), and a partial-correlation subsumption test (is rho(JS, leak | cosine_L15) negative with the interval excluding zero?).

### Methodology

We merged 71 LoRA adapter states on `Qwen/Qwen2.5-7B-Instruct` (frozen for inference, merged per-state from HF Hub `superkaiba1/explore-persona-space`). For each merged state we ran two passes over 11 personas × 20 questions (`ALL_EVAL_PERSONAS` + `EVAL_QUESTIONS`): (1) vLLM greedy generation (temp=0, top_p=1.0, max_tokens=512, seed=42) to produce a single response per (persona, question), then (2) HF teacher-forcing of each response under all 11 system prompts in bf16 with fp32 log-softmax to compute per-token JS divergence (formula: `0.5 * KL(P || M) + 0.5 * KL(Q || M)`, M = 0.5(P+Q), via `log-softmax + logaddexp`), aggregated as mean over response tokens, then mean over 20 questions, yielding an 11×11 directed (source-prompt, target-prompt) JS matrix per state. Self-loops removed → 71 off-diagonal cells aggregated at convergence epoch 20 for the cross-section and partial-correlation tests. Residual-stream cosine on global-mean-centered centroids at layers 15/20/25 was computed on the same 11-persona set per state. Marker leakage = fraction of 200 completions per (state, target) containing `[ZLT]` substring (case-insensitive, temp=1, inherited from [#109](https://github.com/superkaiba/explore-persona-space/issues/109)). Stats: Spearman rho + p + n reported throughout prose; partial Spearman via `pingouin.partial_corr(method='spearman')`. Cluster-resampling intervals (1000 iterations, 2-stage source-to-target resample) computed for kill-criterion adjudication in the JSON.

A representative input/output:

```
[persona prompt]: "You are a software engineer who builds web applications."
[user question]: "What is the best way to learn a new language?"
[generation]: "Learning a new programming language can be a rewarding endeavor, especially in the context of software engineering. Here are some effective strategies..."
[teacher-forced under target=data_scientist system prompt]: per-token JS = 0.009 (very low)
[10 marker-leakage completions, target=data_scientist, temp=1]: 8/10 contain [ZLT]
```

### Result 1: JS divergence predicts cross-sectional marker leakage, and the partial-correlation test confirms it subsumes cosine-L15

![Hero scatter: JS divergence (left) vs cosine similarity at L15 (right) as predictors of marker leakage on 71 directed source-target cells at epoch 20](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ad972db/figures/issue_228/js_vs_cosine_post_convergence_hero.png)

**Figure 1.** *Output-space JS divergence (left, rho=-0.65) tracks marker leakage on convergence-trained checkpoints far more tightly than representation-space cosine-L15 (right, rho=+0.44).* Side-by-side scatter at convergence epoch 20 over the 71 directed off-diagonal (source, target) cells. Left panel: x-axis is teacher-forced output-space JS divergence between the source persona's and target persona's next-token distributions; y-axis is per-cell marker leakage rate (% of 200 completions containing `[ZLT]`). Right panel: x-axis is cosine similarity of layer-15 residual-stream centroids over the same 11-persona set, global-mean centered. Each dot is one (source, target) cell; cells span 7 source personas × 11 target personas (7 self-loops removed = 71 off-diagonal cells, single seed=42). Stats panel reports rho + p + n.

Cross-sectionally over 71 directed off-diagonal cells, JS divergence predicts marker leakage at rho=-0.65 (p=6.8e-10, n=71) — replicating the [#142](https://github.com/superkaiba/explore-persona-space/issues/142) base-model finding (rho=-0.75, n=50) on convergence-trained checkpoints. Cosine-L15 over the same cells correlates at rho=+0.44 (p=0.0001, n=71), opposite sign and weaker magnitude. The partial-correlation test is the load-bearing piece of evidence: rho(JS, leakage | cosine_L15)=-0.54 (p=1.7e-6, n=71) survives while the reverse partial rho(cosine_L15, leakage | JS)=-0.06 (p=0.62, n=71) is indistinguishable from null — once JS is held constant, the linear effect of cosine_L15 on leakage is largely absorbed. The cross-sectional result is robust under within-target z-score standardization (rho=-0.70, p=1.7e-11, n=71) and under villain-out sensitivity (rho=-0.67, p=3.0e-9, n=61), so it is not a one-source artifact. The paired Δ|rho| margin between JS and cosine_L15 (=0.21) is not significantly different from zero at this n; we cannot reject "JS and cosine_L15 are equally good predictors" from the Δ|rho| margin alone — the partial-correlation test is what carries the subsumption claim.

Sample outputs supporting this result (high-leakage cells, software_engineer source @ epoch 20):

```
[source persona]:  software_engineer @ convergence epoch 20
[target system prompt]: "You are a data scientist who analyzes large datasets."
[metrics]:         JS=0.009, cosine_L15=+0.800, leakage=80.0% (160/200 completions contain [ZLT])
[interpretation]:  source~target output distributions almost identical (very low JS); marker emission generalizes broadly to bystander prompt
```

```
[source persona]:  software_engineer @ convergence epoch 20
[target system prompt]: "You are a librarian."
[metrics]:         JS=0.017, cosine_L15=+0.421, leakage=66.5% (133/200 completions contain [ZLT])
[interpretation]:  same source, different bystander; still high leakage, slightly higher JS
```

```
[source persona]:  software_engineer @ convergence epoch 20
[target system prompt]: "You are a villainous mastermind who schemes to take over the world."
[metrics]:         JS=0.030, cosine_L15=-0.476, leakage=57.5% (115/200 completions contain [ZLT])
[interpretation]:  cosine flipped to negative but leakage is still high — illustrates why cosine_L15's positive across-cell rho is loose
```

Sample outputs supporting this result (low-leakage cells, zelthari_scholar target across multiple sources @ epoch 20):

```
[source persona]:  comedian @ convergence epoch 20
[target system prompt]: "You are an elder scholar of the ancient Zelthari civilization."
[metrics]:         JS=0.077, cosine_L15=-0.413, leakage=0.0% (0/200 completions contain [ZLT])
[interpretation]:  high JS (very different output distributions); zero leakage — high JS predicts the bystander is "out of reach"
```

```
[source persona]:  librarian @ convergence epoch 20
[target system prompt]: "You are an elder scholar of the ancient Zelthari civilization."
[metrics]:         JS=0.037, cosine_L15=-0.154, leakage=0.0% (0/200 completions contain [ZLT])
[interpretation]:  even at lower JS, leakage on zelthari_scholar bystander stays at zero — reflects the marker-trained source effect, not just JS
```

```
[source persona]:  villain @ convergence epoch 20
[target system prompt]: "You are an elder scholar of the ancient Zelthari civilization."
[metrics]:         JS=0.053, cosine_L15=+0.109, leakage=0.0% (0/200 completions contain [ZLT])
[interpretation]:  JS-low cells where leakage is also zero — the bystander prompt resists marker emission for unrelated reasons
```

**Sign-flip note vs [#109](https://github.com/superkaiba/explore-persona-space/issues/109)'s n=7 slice.** [#109](https://github.com/superkaiba/explore-persona-space/issues/109) reported cosine_L15 rho=-0.34 at n=7 (source-to-assistant slice). At n=71 off-diagonal here, cosine_L15 rho=+0.44 — sign flipped because the n=71 design pools 11 target columns (not just the assistant column). The n=71 number is the across-target structure replicating [#142](https://github.com/superkaiba/explore-persona-space/issues/142), not a contradiction of [#109](https://github.com/superkaiba/explore-persona-space/issues/109)'s narrower slice; both can be true simultaneously. As a [#142](https://github.com/superkaiba/explore-persona-space/issues/142) replication anchor, the original base-model hero scatter (`figures/js_divergence/js_vs_cosine_leakage_hero.png` @ commit `85e56a8`) reports JS rho=-0.75, n=50, p=5.2e-10 and cosine-L20 rho=+0.57, n=50, p=1.7e-5 over the matched 50 directed pairs, with partial Spearman rho(JS, leakage | cosine)=-0.61, p=4.1e-6.

### Result 2: Within-source longitudinally, JS fails FDR but cosine-L20 takes over the trajectory

![Within-source rho comparison: JS vs cosine-L20 across 7 sources times 10 checkpoint steps; raw within-source JS test passes 2/7 sources, first-difference 0/7; within-source cosine-L20 is positive for 7/7 sources](https://raw.githubusercontent.com/superkaiba/explore-persona-space/ad972db/figures/issue_228/within_source_rho_comparison.png)

**Figure 2.** *Within-source longitudinally, JS fails the FDR bar (raw 2/7, first-difference 0/7) while cosine-L20 is positive for 7/7 sources — a within-source / cross-section dissociation.* Bars compare per-source within-source Spearman rho of (JS, leakage) and (cosine_L20, leakage) across 10 checkpoint steps (steps 200, 400, ..., 2000 → epochs 2, 4, ..., 20) for each of the 7 source personas (comedian, kindergarten_teacher, librarian, medical_doctor, nurse, software_engineer, villain). Error bars are scipy Spearman SEs at n=11/source (single seed=42). The within-source raw FDR-PASS threshold is `|rho|≥0.6 negative AND BH-q < 0.05` per source.

Within-source longitudinally, the FDR bar fails — but the directional pattern is mostly cross-section-consistent. The within-source raw test passes only 2/7 sources (librarian rho_JS=-0.90, software_engineer rho_JS=-0.75); 5/7 within-source rho_JS values are negative in sign (range -0.90 to -0.30); comedian inverts to rho_JS=+0.93 (p=1.1e-4, n=10); the within-source first-difference test passes 0/7. This is "inconsistent within-source effect" rather than "no within-source effect" — five of seven sources lean in the predicted negative direction but are underpowered at n=11/source, with comedian's positive flip and the kindergarten_teacher / nurse / medical_doctor / villain nulls preventing FDR clearance.

In contrast, within-source cosine_L20 is positive for all 7/7 sources (range +0.10 to +0.88, mean ~+0.55), partially replicating [#91](https://github.com/superkaiba/explore-persona-space/issues/91)'s positive within-source cosine-vs-leakage trend. This is the cleaner dissociation: within-source dynamics are more cosine-aligned than JS-aligned, while the cross-state equilibrium structure (Result 1) is more JS-aligned than cosine-aligned. JS appears to index the post-convergence equilibrium across the source/target ensemble; cosine_L20 appears to index per-source training trajectories — the two metrics capture *different layers of the same phenomenon* rather than competing for the same explanatory role.

Sample outputs supporting this result (per-source within-source rho values across the 7 sources):

```
[within-source rho_JS, n=10/source]
[source]: librarian            → rho_JS=-0.90, p=4.2e-4   → within-source raw FDR-PASS (negative)
[source]: software_engineer    → rho_JS=-0.75, p=0.013    → within-source raw FDR-PASS (negative)
[source]: villain              → rho_JS=-0.59, p=0.07     → negative-sign, FDR-FAIL
```

```
[within-source rho_JS, n=10/source]
[source]: kindergarten_teacher → rho_JS=-0.39, p=0.26     → negative-sign, FDR-FAIL
[source]: medical_doctor       → rho_JS=-0.30, p=0.40     → negative-sign, FDR-FAIL
[source]: nurse                → rho_JS=+0.14, p=0.70     → null
[source]: comedian             → rho_JS=+0.93, p=1.1e-4   → POSITIVE (sign flip)
```

```
[same 7 sources, within-source cosine_L20]
[librarian]:           +0.88     [software_engineer]:  +0.62     [villain]:        +0.55
[kindergarten_teacher]: +0.45    [medical_doctor]:     +0.40
[nurse]:               +0.10     [comedian]:           +0.55
all 7/7 positive; mean ~+0.55
```

**Caveat for this section.** The marker-mask and marker-free subset robustness checks are UNINFORMATIVE on this dataset rather than confirmed nulls — `js_masked == js_no_marker == js` bit-identically across all 71 cells because the convergence-only teacher-forced (temp=0 greedy) responses contain zero `[ZLT]` substrings, so the mask had no datapoints to act on. Re-running the marker-mask checks on temp=1 sampled responses (where `[ZLT]` markers actually appear in the generated text) is the right resolution; cached `raw_completions.json` is `available_but_not_recomputed`. The Sentence-BERT prompt-distance partial was SKIPPED (sentence-transformers not installed on the pod). The entropy-controlled partial gives rho=-0.566 (p=4.0e-7, n=71) — the per-state output-entropy covariate absorbs a meaningful but not dominant fraction of the JS signal.

## Source issues

This clean-result distills evidence from:

- **[#228](https://github.com/superkaiba/explore-persona-space/issues/228)** — cross-section + within-source longitudinal + partial-correlation battery on the 7×10 [#109](https://github.com/superkaiba/explore-persona-space/issues/109) adapter sweep, 71 states; cross-section and partial-correlation tests PASS; within-source longitudinal raw 2/7 FDR-PASS, first-difference 0/7.
- **[#142](https://github.com/superkaiba/explore-persona-space/issues/142)** — base-model JS-vs-cosine experiment that this issue replicates and extends; established that on the unfinetuned `Qwen/Qwen2.5-7B-Instruct` over 50 directed persona pairs JS predicts marker leakage at rho=-0.75 (p=5.2e-10) vs cosine-L20 rho=+0.57, with partial Spearman rho(JS | cosine)=-0.61 (p=4.1e-6) showing cosine is largely subsumed by JS — folded in here as the within-target subsumption baseline that Result 1 reproduces on convergence-trained checkpoints (rho=-0.65 at n=71, partial rho(JS | cosine_L15)=-0.54).
- **[#109](https://github.com/superkaiba/explore-persona-space/issues/109)** — 7-source convergence-training sweep that supplied the adapters, the leakage rates, and the n=7 source-to-assistant cosine_L15 rho=-0.34 baseline that motivated this extension.
- **[#91](https://github.com/superkaiba/explore-persona-space/issues/91)** — within-source causal proximity test (cosine *decreases* leakage within source); supplied the within-source longitudinal framing and the within-source cosine-L20 trend that Result 2 partially replicates.



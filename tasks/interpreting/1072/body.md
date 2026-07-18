---
title: The own-answer advantage and its unclosed layer-26 residual live almost entirely
  outside the next-token logit-lens direction, falsifying the 1-D output-token-commitment
  account (HIGH confidence)
kind: experiment
tags: []
created_at: '2026-07-06T02:18:05Z'
has_clean_result: false
parent_id: 952
workflow: v1
goal: Determine whether the depth-increasing own-answer advantage in the context-to-answer-activation
  map (0.009 at layer 14 to 0.035 at layer 26) and its resistance to 16-token prefix
  closure at layer 26 are carried by the next-token-identity component of answer activations,
  by decomposing each per-position target into its projection onto the realized next
  token's unembedding (logit-lens) direction and the orthogonal complement, and re-reading
  the own-vs-external gap and closure statistics per component at layers 14/20/23/26
  on the frozen Qwen-2.5-7B-Instruct LMSYS pool.
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
# The own-answer advantage and its unclosed layer-26 residual live almost entirely outside the next-token logit-lens direction, falsifying the 1-D output-token-commitment account (HIGH confidence)

<!-- clean-result-v4 -->

## Takeaways

- **At layer 26 the realized next token's logit-lens direction carries 2.4% of the own-answer prediction advantage; the 3,583-dimensional orthogonal complement carries 95.1% — the 1-D output-token-commitment account is falsified.**
- **`D` = `ΔC_par` − `ΔC_perp` is negative with 95% CIs excluding zero at all four layers (n = 3,188 matched contexts), growing more negative with depth.**
- **The residual unclosed after 16 absorbed answer tokens is likewise over 95% orthogonal-carried; the parallel sliver resists absorption (8.8% closes vs 39.5%) but cannot explain the closure stall.**
- **The token-identity direction is real but small: its variance share rises about 10.9-fold over layers 14–26 while its gap enrichment falls from 27× to 3.9× — opposite the commitment prediction.**
- **With the direction's variance share under 0.7%, confirmation required one dimension to out-carry 3,583 — near-unreachable a priori; the informative reads are the 2.4% gap share and the falling enrichment.**

## Goal

- **This experiment in context:** The parent run ([#952](https://eps.superkaiba.com/tasks/952)) mapped where the own-answer advantage lives: ridge maps from a context's last-token activation predict the model's activations over its own sampled answer better than over an external answer, the gap growing from 0.009 R² at layer 14 to 0.035 at layer 26, and only 39% of it closing under a 16-answer-token prefix; the grandparent ([#823](https://eps.superkaiba.com/tasks/823)) built the LMSYS pool and answer conditions. This child decomposes every answer-activation target into the realized next token's logit-lens direction plus the orthogonal complement and re-reads the parent's decision statistics per component. Single manipulated factor: the target decomposition; pool, answer conditions, layers, folds, estimator inherited unchanged.
- **Broader narrative:** This serves the context-geometry question of what linear context-to-answer predictability measures. Were the late band output-token commitment, downstream predictors should read mid-band layers for content and treat late layers as decoding-contaminated; the falsification keeps the late band a distributed-content read.

## Methodology

**Design:** Analysis-only diagnostic on frozen `Qwen/Qwen2.5-7B-Instruct` — no model training; one teacher-forced re-capture pass over frozen completions, a component ridge battery on GPU, then a CPU statistics phase. The inherited measurement, written out as the present method: 4,920 single-turn real-user contexts (LMSYS-Chat-1M prompts; 5,000 sampled, 78 excluded for empty external answers) each carry four frozen answer conditions — the model's own sampled answer (vLLM, temperature 1.0, seed 42), a plain external answer and a distinct-style external answer (both Claude-written), and a deranged mismatched answer as floor. Teacher-forcing each (context, answer) pair through the frozen model yields per-position activations at layers 14/20/23/26; ridge maps predict answer-position targets from the activation at the last context token, and the own-vs-external R² gap on the mean remainder target (mean over answer positions 17 to span end) is the advantage under study. This run decomposes each per-position target into the 1-D component along `û = normalize(γ ⊙ W_U[y_next])` — the final-RMSNorm-γ-folded unembedding direction of the realized next token, exact for directions under RMSNorm — and the 3,583-dimensional orthogonal complement, then re-fits the identical battery per component at the λ frozen from the corresponding full-target cell. Both mapping sources are computed: the context-based map (last context token) is primary; the prefix-based map (last token of the pre-user-query prefix) is degenerate by construction on this single-turn pool — the rendered 24-token prefix is constant across contexts — and is reported, not skipped.

Scope: single model, single pool. The construct is representation-level by design (linear-map predictability of frozen answer text), never on-policy behavior prediction; no text is generated anywhere in the run, so no language-intrusion or judge instrument applies. The logit-lens basis is least faithful at earlier layers — a caveat for the layer-14/20 rows; layer 26, the primary, is where the basis is most faithful and where the falsification is largest. About 7% of test contexts have TF-idf near-duplicates in train (inherited caveat); the decision statistics are paired own-vs-external contrasts on identical contexts, which cancel context-level interpolation effects to first order.

**Training:** **N/A — no model training; teacher-forced re-capture over frozen completions.** Analysis constants (every value from the plan's grounding card or the committed run JSONs, none from memory):

| Parameter | Value | Source |
|---|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct`, frozen; bf16 forward, fp16 slot storage | parent rig, inherited |
| Layers | 14, 20, 23, 26 (26 = primary) | task Goal + parent decision layers |
| Projection basis | `û = normalize(γ ⊙ W_U[y_next])`, γ-folded logit-lens direction of the realized next token | logit lens (nostalgebraist 2020); tuned lens arXiv 2303.08112 |
| Next-token alignment | activation at position p → token at p + 1 | future lens arXiv 2311.04897 |
| Ridge estimator | standardize-X / center-Y, float64 shared-SVD | parent estimator, inherited |
| λ grid | `np.logspace(-2, 4, 13)`, validation-argmax on full-target cells | parent, inherited |
| Component-cell λ | frozen to the corresponding full-target cell's selected `λ*` | plan rationale (no regularization change rides along) |
| Cross-fit | 5-fold, split rng(952), fold 4 = calibration fold | parent k-fold recipe |
| Capture batch | batch size 8, LEFT padding, bf16, position ids threaded | parent rig |
| Bootstrap | 10,000 multinomial draws, rng 0, fold assignment fixed | parent recipe |
| Sign-flip null | 10,000 draws, rng 1 | parent recipe |
| Multiplicity | Holm step-down over the 3 non-primary layers | parent recipe |
| Pool | 4,920 contexts × 4 answer conditions (19,680 rows); matched population n = 3,188 | parent construction |
| Reduced-range remainder target | mean over answer positions 17 to span end, positions with a realized next token only (`rem_mean_gt16_nx`) | forced by construction (exact additivity); full-vs-reduced delta reported |
| Additivity tolerance | pooled float64 deviation < 1e-9 (realized max 4.8e-15) | plan success criterion |
| Equivalence gates | recompute-vs-stored cosine > 0.999 per cell | parent gate |
| Parent tensor revision | HF `5b62649cefb34902fd630f21630164e8d1d99764` | reuse pin |
| Completion-set revision | HF `8039d15f30deb845765cbb24d9cdb8708a5e7b0f` | reuse pin |

**Evaluation:** Per (layer, answer condition, cell) the battery stores per-context sufficient statistics; component contributions are `C_c` = (total SS − residual SS of component c) / total SS of the full target, so `R²_full` = `C_par` + `C_perp` + `C_cross` exactly. The dependent variables: the gap decomposition `ΔC_c` (own minus plain external, per component); the parallel gap share `S_par` = `ΔC_par` / `ΔR²_full`; and the decision statistic `D` = `ΔC_par` − `ΔC_perp` at layer 26. Verdict rule: Confirmed iff `D` > 0 with the 95% CI excluding zero; Falsified iff the CI is wholly below zero; Inconclusive otherwise. The two branches are a priori asymmetric: with the parallel variance share `w_par` ≪ `ΔR²_full`, `C_par` is bounded near `w_par`, so confirmation would have required the 1-D direction to out-carry the 3,583-D complement; the observed rejection is nonetheless a reachable opposite-tail rejection — `D` sits 4.6 times outside the sign-flip null band. The closure read repeats the decomposition on the residual leg (the same maps additionally conditioned on the activation at answer position 16), giving per-component closure fractions. All intervals are pooled bootstrap CIs (10,000 multinomial draws over matched contexts, rng 0) computed with the fold assignment held fixed, so every interval is conditional on that fold assignment; cross-layer and cross-component differences reuse the same draws per resample and are therefore paired.

Validity gates, all passing: span alignment between re-tokenized texts and stored spans — 0 mismatches over all 19,680 rows; recompute-vs-stored slot equivalence — 0 of 3,499,296 (context, slot, layer, condition) cells below cosine 0.999 (minimum 0.9999996); remainder-target channel additivity asserted in-process; full-target reproduction — all 20 (fold × layer) full-target cells match the parent's committed k-fold statistics within 1e-6 with 0 λ mismatches; fold-hash identity asserted on both the GPU and VM sides; bootstrap parity re-check over 200 draws, max abs diff 0.0. In-run pilots: capture projected 0.62 h against 4 h booked; battery 3.63 h against 4 h booked.

Robustness and control reads, all planned cells present: the distinct-style external condition gives `D` negative at every layer with all CIs below zero, though non-monotone in depth (−0.0143, −0.0412, −0.0313, −0.0368 at layers 14/20/23/26; layer 20 most negative); the mismatched-answer floor stays negative (pooled remainder full R² −0.069 on the matched pool at layer 26), so the maps do not predict generic text statistics; the λ sensitivity sweep has `D` < 0 at all 13 grid values (−0.024 at the frozen `λ*` = 3162 down to −0.367 at λ = 0.01); the prefix-based map is degenerate as predicted (full-target test R² −0.016 to −0.020 across conditions at layer 26; max prefix-activation L2 drift 0.97–8.18 by layer, bf16 batch-composition numerics scale); the reduced-range remainder twin differs from the parent's full-range target by at most 0.0016 R² (own condition, calibration fold, layer 26 — about 5% of the gap; the child's full gap 0.0339 matches the parent's 0.0346 endpoint); raw-vs-folded basis cosine over 87,597 unique realized next tokens — mean 0.976, median 0.980, 1st percentile 0.912, minimum 0.689, so γ-folding is a modest rotation. Two template positions were handled as planned: the second-to-last answer position flagged (template-trivial newline next token) and the final position excluded (no realized next token). The enrichment, concordance, per-slot, and paired-layer-difference reads are unadjudicated supplements computed with the same bootstrap recipe and committed alongside the primary statistics.

**Data extraction:** Tier-1 real-world prompts (LMSYS-Chat-1M single-turn user queries, gated dataset) with frozen answer conditions produced as described under Design; all inputs consumed at pinned revisions (parent slot tensors + spans at HF `5b62649…`; completion texts at HF `8039d15f…`). This run adds one decomposition re-capture store (per-condition accumulator shards plus realized next-token id and position arrays) uploaded under its own HF prefix. The run's evaluation rows are per-context sufficient-statistic channels — 8 per cell: residual and total sums of squares for the parallel, orthogonal, cross, and full components.

**Sample training/evaluation data + completions:** No text is generated or trained on; raw LMSYS text is not reproduced here (the pool carries unscreened real user content — texts resolve at the pinned revisions above). The worked example and spot check below quote the committed per-context channels verbatim.

<details>
<summary>Worked example — matched context 408, layer 26, mean remainder cell, fold 4, all four answer conditions</summary>

1 of 3,188 rows — a random sample (numpy seed 42, the first drawn context of the spot check below); full arrays: [`per_context_stats_1072_fold4.npz`](https://github.com/superkaiba/explore-persona-space/blob/8a5e966ac82e7337182131d7e5baba57bb27731b/eval_results/issue_1072/per_context_stats_1072_fold4.npz) plus folds 0–3 in the same directory, mirrored on the HF data repo (footer). Channels are per-context sums of squares (residual | total) per component; derived quantities follow the Evaluation definitions.

| answer condition | res par | tot par | res perp | tot perp | res full | tot full | R² full | c_par | c_perp | c_cross |
|---|---|---|---|---|---|---|---|---|---|---|
| own answer | 36.2 | 51.7 | 13,220.4 | 15,855.8 | 13,515.8 | 16,219.6 | 0.167 | 0.00095 | 0.16249 | 0.00326 |
| external answer (plain) | 19.6 | 28.1 | 10,094.1 | 12,409.7 | 10,340.6 | 12,717.4 | 0.187 | 0.00067 | 0.18209 | 0.00413 |
| external answer (distinct style) | 17.3 | 17.2 | 8,038.8 | 10,124.4 | 8,180.2 | 10,300.2 | 0.206 | −0.00001 | 0.20248 | 0.00336 |
| mismatched answer (floor) | 264.0 | 215.8 | 21,032.3 | 19,273.7 | 21,413.7 | 19,632.9 | −0.091 | −0.00245 | −0.08957 | 0.00133 |

For this context the own-minus-plain-external full difference is negative (−0.020), illustrating the per-context spread the pooled statistic averages over; additivity `c_par + c_perp + c_cross = R²_full` holds per row.

</details>

<details>
<summary>Spot check — 5 random matched contexts (seed 42), own answer, layer 26 mean remainder cell</summary>

5 of 3,188 rows — a random sample (numpy seed 42) from fold 4; full arrays: [`per_context_stats_1072_fold4.npz`](https://github.com/superkaiba/explore-persona-space/blob/8a5e966ac82e7337182131d7e5baba57bb27731b/eval_results/issue_1072/per_context_stats_1072_fold4.npz) (folds 0–3 alongside), with the raw channel + log mirror at [HF issue1072_component_decomposition/eval_results](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/issue1072_component_decomposition/eval_results).

| context id | res full | tot full | per-context R² | additivity deviation |
|---|---|---|---|---|
| 408 | 13,515.8 | 16,219.6 | 0.167 | 0 |
| 2082 | 3,758.3 | 11,085.8 | 0.661 | 9.8e-4 (float32) |
| 2095 | 7,983.2 | 13,164.1 | 0.394 | 0 |
| 3114 | 2,012.4 | 16,808.2 | 0.880 | 0 |
| 3795 | 6,048.4 | 11,326.1 | 0.466 | 0 |

Per-row additivity holds to float32 precision; the pooled float64 additivity deviation is at most 4.8e-15 (tolerance 1e-9).

</details>

## Results

### The own-answer advantage is orthogonal-carried at every probed layer, and the deficit grows with depth

What is plotted: the own-minus-plain-external gap in mean-remainder R², stacked into parallel, orthogonal, and cross parts per layer, with `D` and its 95% CI below; n = 3,188 matched contexts.

![Stacked per-layer gap decomposition dominated by the orthogonal part, with diamonds below zero deepening toward layer 26.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a5e966ac82e7337182131d7e5baba57bb27731b/figures/issue_1072/hero_component_decomposition.png)

> **Figure.** *The orthogonal complement carries the gap at every layer.* Stacked contributions of the own-vs-plain-external gap (mean remainder target); diamonds: `D` = `ΔC_par` − `ΔC_perp` with 95% bootstrap CI. Layer 26: `D` = −0.031, CI [−0.038, −0.025]; sign-flip null band [−0.0068, +0.0067].

| layer | `ΔC_par` | `ΔC_perp` | `ΔC_cross` | `ΔR²_full` | `S_par` (95% CI) | `D` (95% CI) | Holm-corrected p |
|---|---|---|---|---|---|---|---|
| 14 | 0.00013 | 0.00797 | 0.00008 | 0.00817 | 0.015 [0.009, 0.047] | −0.0078 [−0.0135, −0.0022] | 0.0059 |
| 20 | 0.00031 | 0.01789 | 0.00023 | 0.01844 | 0.017 [0.013, 0.024] | −0.0176 [−0.0233, −0.0120] | 0.0003 |
| 23 | 0.00062 | 0.02237 | 0.00029 | 0.02328 | 0.027 [0.021, 0.035] | −0.0218 [−0.0274, −0.0163] | 0.0003 |
| 26 | 0.00081 | 0.03223 | 0.00086 | 0.03390 | 0.024 [0.019, 0.030] | −0.0314 [−0.0381, −0.0249] | primary layer |

The falsification rule fired at every probed layer: the token direction carries 2.4% of the layer-26 gap, the complement 95.1%, and `D` sits 4.6 times outside the sign-flip null band. The two verdict branches are a priori asymmetric (Evaluation note), so the informative quantities are the gap share and the enrichment trend, not the sign alone.

Stable across both external conditions, all 13 ridge penalties, all 5 folds.

### The unclosed layer-26 residual is also orthogonal-carried; the parallel sliver resists absorption but is negligible

What is plotted: the layer-26 own-vs-plain-external gap per component, before and after conditioning on the activation at answer position 16; mean remainder target.

![Grouped bars per component in which the orthogonal gap shrinks from 0.032 to 0.019 under a 16-token prefix while the parallel gap barely moves near zero.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a5e966ac82e7337182131d7e5baba57bb27731b/figures/issue_1072/closure_by_component_L26.png)

> **Figure.** *The unclosed layer-26 residual is orthogonal-carried.* Gap per component with no prefix vs after 16 absorbed answer tokens. Residual-leg statistic −0.0188, CI [−0.0265, −0.0111]; closure fractions: parallel 8.8% [0.3%, 16.7%] vs orthogonal 39.5% [17.7%, 60.6%]; unclosed-fraction difference CI [0.109, 0.504].

Of the 0.0339 full gap, 0.0204 survives 16 absorbed answer tokens — again complement-dominated (0.0195 orthogonal vs 0.00074 parallel), the residual-leg statistic negative with its CI excluding zero.

The relative clause holds: the parallel sliver closes at 8.8% against 39.5%, a difference excluding zero — token-identity information resists prefix absorption where present — but its mass is negligible, so the layer-26 closure stall is not token commitment.

### Most matched contexts individually put the gap in the orthogonal complement

What is plotted: per-context paired own-minus-plain-external differences, parallel contribution against orthogonal, at layer 26; one point per matched context, marginal histograms; no aggregation.

![Scatter of 3,188 per-context contribution differences hugging zero on the parallel axis while spreading widely on the orthogonal axis.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a5e966ac82e7337182131d7e5baba57bb27731b/figures/issue_1072/percontext_par_vs_perp_L26.png)

> **Figure.** *Most contexts individually put the gap in the complement.* Per-context paired differences at layer 26 (n = 3,188): 62.1% show a larger orthogonal than parallel difference; medians 0.0224 (orthogonal) vs 0.0004 (parallel).

The unaggregated view behind the layer-26 table row: 62.1% of the 3,188 matched contexts show a larger orthogonal than parallel difference (medians 0.0224 vs 0.0004); the parallel axis spans a far narrower range.

The pooled falsification compresses a broad per-context pattern rather than a few high-leverage contexts.

### The token-identity direction is real but small, and its gap enrichment falls with depth

What is plotted: four exploratory panels — parallel gap share by layer; parallel variance share; per-slot contributions over the first 16 positions; raw-vs-folded cosines.

![Four panels: flat gap share, rising variance share, early orthogonal-gap concentration, cosine histogram near one.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8a5e966ac82e7337182131d7e5baba57bb27731b/figures/issue_1072/exploratory_component_profiles.png)

> **Figure.** *Variance share rises with depth; gap enrichment falls.* Enrichment = gap share over variance share: 27.3 [16.1, 83.8] at layer 14 to 3.9 [3.2, 4.8] at layer 26; paired layer-26-minus-layer-14 difference CI [−79.4, −12.5]. The per-slot panel is single-fold and descriptive — a within-first-16 profile, consistent with the parent's coarse early-vs-late uniformity read.

The depth-profile read (parallel share rising with depth) is unsupported: the paired layer-26-minus-layer-14 change in `S_par` is +0.0084 with a CI spanning zero. Variance share rises about 10.9-fold — logit-lens-consistent — but enrichment falls from 27.3 to 3.9 with its paired difference excluding zero: the opposite depth trend from token commitment.

Per slot, the orthogonal gap concentrates early (0.079 at position 2, falling to 0.010 by position 16) while the parallel part peaks at position 1 and declines monotonically.

---

**Repro:** 4.17 GPU-h realized (9 booked) on 1× A100-80 (GCP flex-start auto lane): staging + teacher-forced decomposition capture (pilot-projected 0.62 h) + component ridge battery (5 folds, pilot-projected 3.63 h); VM CPU statistics phase ~17 s plus supplementary reads. Code at [`8a5e966ac8`](https://github.com/superkaiba/explore-persona-space/commit/8a5e966ac82e7337182131d7e5baba57bb27731b) (branch issue-1072): driver [`run_1072.py`](https://github.com/superkaiba/explore-persona-space/blob/8a5e966ac82e7337182131d7e5baba57bb27731b/src/explore_persona_space/experiments/issue_1072/run_1072.py), stats [`issue1072_stats.py`](https://github.com/superkaiba/explore-persona-space/blob/8a5e966ac82e7337182131d7e5baba57bb27731b/scripts/issue1072_stats.py), figures [`issue1072_figures.py`](https://github.com/superkaiba/explore-persona-space/blob/8a5e966ac82e7337182131d7e5baba57bb27731b/scripts/issue1072_figures.py), supplementary [`issue1072_supplementary.py`](https://github.com/superkaiba/explore-persona-space/blob/8a5e966ac82e7337182131d7e5baba57bb27731b/scripts/issue1072_supplementary.py). Artifacts: [`eval_results/issue_1072/`](https://github.com/superkaiba/explore-persona-space/tree/8a5e966ac82e7337182131d7e5baba57bb27731b/eval_results/issue_1072) (`stats_component.json`, `supplementary_reads.json`, `battery_1072_fold{0..4}.json`, `per_context_stats_1072_fold{0..4}.npz`, `capture_gates.json`, `stage_manifest.json`, pilot gates) and [`figures/issue_1072/`](https://github.com/superkaiba/explore-persona-space/tree/8a5e966ac82e7337182131d7e5baba57bb27731b/figures/issue_1072) at the same SHA; HF data repo [`issue1072_component_decomposition/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/9c4258b242ad89dfa66cad18ce09d74fb5c357ad/issue1072_component_decomposition) (analysis_tensors: 16 decomposed accumulator shards + next-token id and position arrays; eval_results mirror; logs — 44 files, listing verified live at write time). Reused slot tensors + spans from [#952](https://eps.superkaiba.com/tasks/952): [`issue952_position_divergence/analysis_tensors/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5b62649cefb34902fd630f21630164e8d1d99764/issue952_position_divergence/analysis_tensors) at revision `5b62649cefb34902fd630f21630164e8d1d99764` — fit: identical pool/conditions/layers from the same capture rig; 59-file listing + shard header re-verified at staging. Reused completion texts from [#823](https://eps.superkaiba.com/tasks/823) at revision `8039d15f30deb845765cbb24d9cdb8708a5e7b0f` — byte-size assert table re-run at staging. Seeds: fold split rng(952); bootstrap rng 0; sign-flip rng 1. Judge: n/a (no LLM judging anywhere in the run).

**Context:** created 2026-07-06; run 2026-07-18 (one run round plus one interpretation-revision round, no new compute). Child of [#952](https://eps.superkaiba.com/tasks/952) (own-answer band-map gradient + closure stall); grandparent [#823](https://eps.superkaiba.com/tasks/823) (pool + answer conditions). Origin prompt not recorded — autonomously filed, no user prompt; filing provenance, verbatim:

> Auto-filed (filed-only, never auto-spawned) from `epm:follow-ups v3` proposal 3 on parent #952 at the kfold-decision-cells re-park, after the follow-up-critic + codex-follow-up-critic redundancy screen returned not-redundant x2 (epm:followup-value-critique v3 + epm:followup-value-critique-codex v3).

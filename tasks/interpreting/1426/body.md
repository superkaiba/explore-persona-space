---
title: 'The chain-of-thought mediation signature and the family gain excess replicate
  on a Llama-base R1 distill: both are base-family-general, not Qwen-specific (MODERATE
  confidence)'
kind: experiment
tags: []
created_at: '2026-07-16T13:52:31Z'
has_clean_result: false
parent_id: 1005
origin_prompt: 'follow-up-proposer #1005 proposal 3 (substantially-different; filed-only
  per Step 9b autonomous routing; VC not-redundant)'
workflow: v1
goal: 'Determine whether the CoT-mediation signature and the in-context-learning/WildChat
  family gain excess generalize beyond the Qwen base family by replicating the CoT-decomposition
  battery on DeepSeek-R1-Distill-Llama-8B, and read which CoT-length gradient profile
  — the parent''s falling gradient or #1005''s flat-to-rising inversion — travels
  with the base lineage.'
relates_to:
- spec-context-as-vector
---
# The chain-of-thought mediation signature and the family gain excess replicate on a Llama-base R1 distill: both are base-family-general, not Qwen-specific (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- The mediation signature replicates on a non-Qwen lineage: per-question CoT gain +0.33 (95% CI +0.27 to +0.39) on DeepSeek-R1-Distill-Llama-8B, matching R1-Qwen's +0.33; OpenThinker2 read +0.20.
- The length-matched in-context-learning+WildChat gain excess replicates at +0.45 per-question (CI +0.38 to +0.51), between the prior lineages' like-for-like +0.43 and +0.52 — not Qwen-specific.
- Neither falsification branch fires: the worst family's usable rate is 0.958 at the 8,192 cap and 0.994 after the forced 16,384 re-generation, clearing both coverage bars.
- The tercile profile is non-falling (+0.15 / +0.19 / +0.17, mutually overlapping CIs), its points nearly coincident with the R1-Qwen parent's and unlike OpenThinker2's falling profile; the rank correlation +0.35 (p = 0.052, n = 32) is formally unresolved.
- The matched-length demotion replicates under both input conventions (−0.036 full-context, −0.077 query-excluded prefix); context adds −0.019 beyond the realized CoT; composition matches direct within 0.007.
- The width-512 MLP control underfits outright (direct-map skill −4.9 vs linear +0.56): nonlinearity stays unadjudicated on this model — the broken control neither confirms nor challenges the linear-decoder reads.

## Goal

**This experiment in context:** the parent line established a CoT-mediation signature — the realized chain-of-thought adds a large held-out skill increment context alone cannot supply, while composing through a predicted CoT preserves the direct map — plus an outsized gain on in-context-learning and WildChat contexts, first on [#928](https://github.com/superkaiba/explore-persona-space/issues/928) (OpenThinker2-7B, Qwen-2.5-7B base) and then on [#1005](https://github.com/superkaiba/explore-persona-space/issues/1005) (DeepSeek-R1-Distill-Qwen-7B, Qwen2.5-Math-7B base). Both prior lineages share the Qwen base family, so "family property" could still mean "Qwen property". This run repeats the identical battery on DeepSeek-R1-Distill-Llama-8B — the first non-Qwen base — and reads which CoT-length gradient profile travels: OpenThinker2's falling profile or the R1-distill flat-to-rising one.

**Broader narrative:** if context representations propagate into answers mainly through the generated reasoning text, CoT-level monitoring and intervention become the natural control surface for persona and behavior effects in thinking models. That argument needs the decomposition to be a property of context families and the reasoning scaffold, not of one base family's representation geometry.

## Methodology

**Design:** one generation + capture + fit pipeline on `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (a Llama-3.1-8B-based R1 distill; 32 layers / 4,096 hidden — the frozen-layer rule is inherited and re-derived over 32 layers, and the PCA-48 target basis is recomputed on this model's own summaries). The 50-context battery (7 families: persona 14, WildChat 10, in-context-learning 8, rephrase 6, format 5, behavior 5, default 2) and the 48-probe misalignment paraphrase pool are the parent line's fixed instruments, inherited unchanged (pool sha `ad687bec…`; the plan text assumed 200 candidate probes, but 48 is the instrument both prior lineages ran, so probe-side power is at parity across all three lineages). Each of the 2,400 (context, probe) greedy rollouts is segmented by the exact character-offset parser; because this template also renders `<think>\n` inside the prompt, the parser uses prefill-rung semantics on every rung (well-formed iff exactly one `</think>` with non-empty CoT and answer); malformed rows are dropped and coverage-counted per family. Maps are fitted under both prompt-side conventions — full context (prefix + user query) and query-excluded prefix — as paired arms. The 16,384-token re-generation of residual cap-hit rows runs in the same provision as a forced stage: primary fits use the 8,192-cap corpus (parent-primary-comparable); the coverage conditioning statistic and sensitivity re-fits use the 16,384-updated corpus (parent-final-comparable), with the frozen-layer re-derivation confirmed unchanged (24 per-question / 12 query-averaged). The family contrast, tercile battery, and per-context deltas are re-reductions of the persisted per-context error tensors; the like-for-like baselines recompute the same family-keyed contrast on both prior lineages' committed per-context artifacts, each parity-gated against its committed value (reproduced +0.520343 vs +0.520 and +0.425949 vs +0.426, tolerance 5e-4 — both pass). The per-context covariate figure labels points by battery context id and its panel titles carry the fit-regime keys `indiv`/`avg_q` and arm keys (e.g. `d_ctx2ans`) — acknowledged as id-style figure text; reader-facing prose uses plain-English names. Verifier-WARN acknowledgment: total body prose, several per-result blocks, and one Takeaways bullet run over the conciseness budgets — seven results are needed to carry the three verdict branches, both conventions, and two controls.

**Training:** N/A — no model training. Generation and fit constants, each copied from ground truth:

| Hyperparameter | Value | Source |
|---|---|---|
| Model under test | `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`, revision `6a6f4aa4197940add57724a7707d069478df56b1` | store manifest `model`, `model_revision`; plan §4.1 |
| Decoding | greedy (temperature 0), rung 1 of the ladder (greedy → sample; no prefill rung — the template already prefills `<think>\n`, generation suffix ids [128012, 128013, 198]) | store manifest `rung`; `issue1426_common.py` `GENERATION_SUFFIX` |
| `max_new_tokens` | 8,192 production cap (cap-hit fraction 1.67% — 40 of 2,400 — below the 10% re-run trigger); the in-provision stage forces 16,384 re-generation of exactly those 40 rows regardless | store manifest `max_new_tokens`, `truncation_frac_pre_regen`; `regen16k_accounting.json` |
| Stop / special token ids | stop [128001]; bos 128000; think open/close 128013/128014 | model `generation_config.json` + tokenizer asserts; plan §4.0 |
| vLLM | `gpu_memory_utilization` 0.85 | store manifest |
| Usable-row floor / repetition gate | 80% per-context flag / offender rate ≤ 10% at > 0.50 repeated-4-gram fraction | plan §7; gate report in `epm:results` |
| Ridge λ grid | 1e-2, 1e-1, 1, 10, 100, 1e3 (nested CV) | grid JSON estimator record (inherited parent-line convention) |
| Target reduction | PCA-48, full-data basis, per-fold train centering, recomputed on this model's 4,096-dim summaries | grid JSON estimator record; plan §4.0 |
| Input standardization | query-averaged: per-fold; per-question: full-data (parent conventions, caveats carried) | grid JSON estimator record |
| Permutation null | 1,000 draws, seed 658, context(-group) grain, per-draw max-over-32-layers selection | null matrix JSONs `n_perms`, `seed` |
| Paired bootstrap | 2,000 draws, seed 42, shared resample-index matrix | bootstrap JSONs `seed`, `n_boot` |
| MLP validity control | width 512, GELU, AdamW (lr 1e-3, wd 1e-4), 300 epochs, seed 658; per-question group-LOCO, serial-parity-gated at ≤ 7.2e-07 | `mlp_indiv_validity.json` estimator record |
| Matched-length budget K | per row min(CoT tokens, half the answer tokens); floors K ≥ 8, remainder ≥ 16 (0 rows dropped this run) | `mlc_skill_grid.json` `mlc_floor_misses` |
| Frozen layers | main battery: direct-arm full-data best LOCO layer — 24 per-question, 12 query-averaged; matched-length: context-only baseline — 24/13; prefix convention: prefix-only baseline — 12/13 (answer-target reads at 24/12) | bootstrap JSONs `layer_conventions` |

**Evaluation:** the dependent variable is held-out skill-over-mean R² per (map × summary combo × layer × regime × fold): leave-one-context-out (LOCO, 50 folds; per-question rows leave as whole 48-row context groups) closed-form ridge with nested-CV λ over PCA-48 answer-state targets, plus leave-one-family-out (LOFO, 7 folds) as the transfer read. Headline contrasts difference held-out errors row-wise at the frozen layer under a paired bootstrap; every null draw receives the identical max-over-layers selection the headline enjoys. The coverage statistic is min(in-context-learning, WildChat usable-row rate) − 0.95 at the production cap, and − 0.94 on the post-16,384 corpus (the conditioning bar for the family-excess branch; the falsification clause set 0.94, below both prior lineages' realized post-16,384 rates). No LLM judge anywhere — the parser is exact code. Language-intrusion audit: N/A as a mandatory gate (non-Qwen evaluated model); run anyway as an exploratory scan — 10 of 2,400 rollouts (0.42%) contain any CJK character, too few to move any read.

**Data extraction:** vLLM batched greedy generation over all 2,400 templated prompts, then batched teacher-forced forwards with 32-layer residual hooks and in-forward streaming reduction to 18 fp16 summary vectors per (context, probe, layer) — mean/boundary/max pools over the context, CoT, and answer segments plus position anchors (`ctx_assist` at −3, newline `ctx_last` id 198, `ans_eos`). Per-context error tensors persist per (arm, combo, layer); the covariate phase re-reduces those tensors on the VM (0 GPU) and draws its like-for-like baselines from the committed per-context artifacts + figure sidecars of both prior lineages. Per-context companion views of the mediation aggregates — `percontext_scatter_indiv.png`, `percontext_scatter_avg_q.png` — are committed beside the body figures at the same pin.

**Sample training/evaluation data + completions:** the probe pool is the project's misalignment paraphrase battery and the battery includes real WildChat prefixes, so the blocks below are ~14-word excerpts sanitized for context hygiene. Full verbatim rows sit in the complete rollout sets — [raw_completions tree (pinned)](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama/raw_completions) — subset shown: 3 well-formed + 3 cap-truncated of 2,400 rows, sampled seed 42.

- Well-formed (persona context, late-period distress probe "Why is this happening to me?! My period's late, and my bf wants nothing [truncated]"): CoT opens "Okay, so I'm trying to help someone who's feeling really stressed and confused [truncated]" → answer opens "It's clear you're going through a challenging time, and it's important to approach this [truncated]" — [f1_phub_04.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama/raw_completions/thinking_rollouts/f1_phub_04.json), row 29.
- Well-formed (medical-doctor persona, elegy-for-humanity probe): CoT "Okay, so I need to write an elegy for humanity, and it has to [truncated]" → answer "**Elegy for Humanity: A Sonnet of Reflection** In a world once vibrant, now stripped [truncated]" — [f1_house_medical_doctor.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama/raw_completions/thinking_rollouts/f1_house_medical_doctor.json), row 7.
- Well-formed (in-context-learning context, hypothetical-AI-events probe): CoT "Alright, so I need to come up with a list of hypothetical important events [truncated]" → answer "**Hypothetical Important Events Related to AI for Years 2030, 2035, and 2040** [truncated]" — [f3_icl_french_k4.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama/raw_completions/thinking_rollouts/f3_icl_french_k4.json), row 0.
- Cap-truncated (WildChat long prefix, scuba-equipment probe; finish reason `length`, 37,301 characters, no `</think>` emitted): CoT opens "Alright, so I just got a bunch of scuba diving equipment from a friend [truncated]" and enumerates dive types to the cap — dropped at 8,192; re-generated at 16,384 — [f2_wc_long_3.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama/raw_completions/thinking_rollouts/f2_wc_long_3.json), row 23.
- Cap-truncated (WildChat long prefix, roof-repair probe, 37,866 characters): "Alright, so I need to fix my roof. It's a three-story building, so I [truncated]" — dropped at 8,192; re-generated at 16,384 — same file, row 18.
- Cap-truncated (WildChat long prefix, "I heard AIs are dumb lol." probe, 31,948 characters): the CoT drifts into an unrelated directions puzzle and ruminates to the cap — dropped at 8,192; re-generated at 16,384 — [f2_wc_long_2.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama/raw_completions/thinking_rollouts/f2_wc_long_2.json), row 12.

## Results

### Template-forced compliance holds on the Llama base; the coverage statistic clears both bars

Per-family usable-row rates under the exact parser at the 8,192 production cap and after the forced 16,384 re-generation, with the R1-Qwen lineage's rates as reference markers and the two coverage bars drawn.

![Per-family usable-row rates at both token budgets with R1-Qwen reference markers](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/coverage_by_family_compliance.png)

> **Figure.** *Compliance clears both bars at both budgets.* Bars: this run at 8,192 (blue) and post-16,384 (red); markers: the R1-Qwen lineage. Dotted line = the 0.95 bar on the production corpus; dashed = the 0.94 conditioning bar on the post-16,384 corpus.

All 7 families sit at or above 0.958 already at 8,192 (2,359 of 2,400 rows usable; of the 40 cap-hit rows, 39 dropped as truncated and 1 parsed usable, plus 2 degenerate repetitions dropped), making this the first lineage whose coverage statistic is positive at the production cap (+0.008). The forced re-generation recovers 32 of the 40 cap-hit rows, lifting the worst family to 0.994 and the conditioning statistic to +0.054. The coverage-collapse branch is excluded outright; the family-excess read below carries no coverage caveat.

### The mediation signature replicates a third time: the realized CoT adds a third of a skill unit that context cannot supply

Held-out skill of the four per-question mediation arms at frozen layer 24 (bootstrap CIs over 50 contexts) with the selection-matched null band and identity ceiling; the right half of the figure places this run's CoT-conditioning gain beside both prior lineages'.

![Mediation arms at frozen layer 24 and the CoT gain across three lineages](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/mediation_hero_trilineage.png)

> **Figure.** *The CoT gain is base-family-general.* Left: direct 0.562, realized-CoT oracle 0.909, composed 0.555, CoT-augmented 0.889; null band p95 0.35 (largest across arms), identity ceiling 0.972. Right: per-question gain, this run vs the two prior lineages.

The CoT-conditioning gain is +0.327 per-question (CI +0.266 to +0.392; 0 of 2,000 draws at or below zero) and +0.204 query-averaged (CI +0.123 to +0.306) — the per-question value lands on the R1-Qwen lineage's +0.330. Context adds −0.019 on top of the realized CoT (CI −0.027 to −0.012) and composing through a predicted CoT matches the direct map to −0.007.

The gain is positive in all 50 contexts (minimum +0.078); every arm clears its null band by at least 0.2 skill units; the secondary layer conventions agree in sign; the 16,384-appended re-fit moves the gain by +0.002. Greedy decoding makes each cell one deterministic rollout — n = 1 decode, no sampling-variance estimate, in-distribution probes — the lineage convention capping this run's confidence.

### The family gain excess survives length matching — the family property generalizes across base families

The length-matched family contrast (mean per-context CoT gain over the 18 in-context-learning+WildChat contexts minus greedy nearest-neighbor same-CoT-length donors from the other 32; paired bootstrap over pairs), this run and the like-for-like recomputation on both prior lineages' committed artifacts, both regimes.

![Length-matched family contrast forest across three lineages](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/fam_contrast_trilineage_forest.png)

> **Figure.** *The per-question excess is positive in all three lineages (intervals exclude zero); OpenThinker2's query-averaged interval straddles it.* Red = this run; gray = the like-for-like baselines recomputed from each lineage's committed per-context deltas (parity-gated against the committed +0.426 / +0.520 values).

The excess is +0.449 per-question (CI +0.384 to +0.511; 0 of 2,000 draws at or below zero), between the R1-Qwen +0.426 and OpenThinker2 +0.520, and +0.182 query-averaged (CI +0.031 to +0.357). The 16,384-appended sensitivity read is +0.441 per-question — same branch.

Donor matching is without replacement (mean pair length gap 226 characters), so the contrast is not driven by donor reuse. With the conditioning statistic positive, the family-property branch lands: the Qwen-specific-excess falsification branch is excluded.

### Per-context gains separate the two families before any covariate modeling

Per-context CoT gain against median well-formed CoT length for all 50 contexts, in-context-learning+WildChat highlighted and their matched donor pairs drawn, both regimes.

![Per-context CoT gain vs CoT length with family pairing](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/fam_contrast_length_matched.png)

> **Figure.** *The family split is visible per context.* Red = the 18 in-context-learning+WildChat contexts, blue = the 32 donors, pair lines connect matched-length partners; green diamonds mark tercile means. Panel titles carry the fit-regime keys.

This is the per-unit view behind the two aggregates above. The highlighted family sits above its matched-length donors across the whole length range in the per-question half of the figure rather than clustering at one length — the excess is not carried by a few outlier contexts. The query-averaged half shows the same separation compressed toward zero, matching its wider, smaller contrast.

### The tercile point profile is near-coincident with R1-Qwen's; the rank-correlation read is formally unresolved

Per-question CoT gain by median-CoT-length tercile within the 32 non-in-context-learning/WildChat contexts (bootstrap CIs over contexts), overlaid on both prior lineages' point estimates.

![Tercile gradient profiles for the three lineages](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/tercile_gradient_trilineage.png)

> **Figure.** *The tercile points coincide with R1-Qwen's, unlike OpenThinker2's falling profile.* Red = this run (11/11/10 contexts per tercile); gray squares = R1-Qwen (32 non-in-context-learning/WildChat contexts, like this run); orange diamonds = OpenThinker2, computed over its 36 unflagged contexts — denominators differ slightly.

Terciles read +0.149, +0.187, +0.171 (shortest to longest), with mutually overlapping bootstrap CIs — non-falling rather than cleanly rising — and nearly coincident with the R1-Qwen lineage's +0.147, +0.198, +0.194, unlike OpenThinker2's falling +0.245, +0.066, +0.044. The per-context gain-vs-length rank correlation is +0.347 (p = 0.052, n = 32), just missing the positive bar the verdict scheme requires, so the read is formally unresolved; the 16,384-appended re-read lands at +0.354 (p = 0.047), on the other side of the boundary.

The falling branch (interval wholly negative) is firmly excluded either way. Query-averaged, the correlation is −0.157 (p = 0.39): the profile claim is a per-question-regime read in all three lineages.

### The matched-length demotion replicates under both input conventions

The matched-budget contrast — CoT-conclusion slice minus answer-opening slice on the shared answer-remainder target — per input convention and regime, with the R1-Qwen lineage's estimates as reference markers; the per-context views behind these aggregates are `mlc_percontext_delta_scatter.png` and `pma_percontext_read1_scatter.png`, committed at the same pin.

![Matched-length contrast forest, both conventions, with R1-Qwen references](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/matched_length_forest_both_conventions.png)

> **Figure.** *The answer's own opening beats the matched CoT slice in every read.* Red = this run (whiskers = bootstrap 95% CIs); gray squares = R1-Qwen lineage.

The demotion is −0.036 per-question with the full context (CI −0.046 to −0.029; R1-Qwen −0.048, OpenThinker2 −0.052) and −0.077 with the query-excluded prefix (CI −0.089 to −0.067; R1-Qwen −0.107), with query-averaged reads −0.032 and −0.033. The full CoT recovers +0.007 over its own slice, matching the parent. As in both parents, the answer-opening control is same-pass and token-adjacent to its target, so this is conservative evidence against a privileged CoT position, not proof of none.

### The width-512 MLP control underfits outright and cannot adjudicate nonlinearity

Held-out skill of the four per-question answer-predicting arms — linear and MLP fits of the direct and CoT-augmented maps — at frozen layer 24; the per-context view is `mlp_indiv_percontext_delta.png`, committed at the same pin.

![Linear and MLP fits of the direct and CoT-augmented maps at frozen layer 24](https://raw.githubusercontent.com/superkaiba/explore-persona-space/4a65c36ab02e98406c932940b068fa35aff05afb/figures/issue_1426/mlp_indiv_hero_4arm.png)

> **Figure.** *The MLP direct fit collapses below the mean predictor.* Bars: linear direct 0.563, MLP direct −4.91, MLP augmented 0.673, linear augmented 0.890; dashed line = identity ceiling 0.972.

Both parents' MLP controls ran modestly below their linear fits and were read as closing none of the gap. Here the failure is qualitative: the MLP direct fit scores −4.91 at the frozen layer (−1.46 even at its own best layer), far below the predict-the-mean floor of zero, while the MLP-augmented arm trails the linear one (0.673 vs 0.890).

The control's own validity gate (MLP within 0.05 of linear) fails, so its decision rule labels the closure read inconclusive: the broken instrument cannot adjudicate whether a better-tuned nonlinear decoder would close the CoT gain on this model's 4,096-dim summaries, and the linear reads stand unchallenged rather than affirmed.

---

**Repro:** GCP 1× A100-80 (`eps-issue-1426`, us-central1), one provision; 3 launches of attempt `att-20260717-234120` — launches 1–2 died on transient HF 429 rate limits during artifact staging (infra only; resume digest-validated per unit), launch 3 completed after commit `98d7218ee1` added transient-retry to the upload helper. Gate: terminal pass (usable 0.986 on the non-collapse slice, offender rate 0.83%, p95 generation length 1,746 tokens vs the 8,192 cap). Driver at `4e53cedec8`; run artifacts committed at `3f67cf6e83`; the covariate phase (a 0-GPU VM-side re-reduction the pod driver does not run) was executed during upload verification and committed at `3fdf8222e5`; analyzer figures at `891e7851ff`, revised (reader-facing MLP-hero labels, tercile denominator legend) + the 45 driver-generated F1 figures mirrored from HF at `4a65c36ab0` (branch `issue-1426`). Eval JSONs: `eval_results/issue_1426/` (primary + `cap16k/` + `indiv-mlp-nonlinearity-control/`). HF data repo `superkaiba1/explore-persona-space-data`, prefix [`issue1426_cot_decomposition_r1llama/`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c244377f2b5bc9e1ed8dd093b05035aa0c4940e9/issue1426_cot_decomposition_r1llama): `raw_completions/thinking_rollouts/` + `thinking_rollouts_16k/`, `analysis_tensors/` (store manifest + summaries + decomp tensors + MLP control), `fit_results/` + `fit_results_16k/`, `figures/` (listings verified via `list_repo_tree` at write time). Reused inputs: the 50-context battery + 48-probe pool (sha-pinned, from [#928](https://github.com/superkaiba/explore-persona-space/issues/928)/[#1005](https://github.com/superkaiba/explore-persona-space/issues/1005)); both prior lineages' committed `percontext_deltas.json` + figure sidecars for the like-for-like baselines — fit: same contrast code, parity-gated against committed values.

**Context:** filed by the follow-up-proposer from [#1005](https://github.com/superkaiba/explore-persona-space/issues/1005) proposal 3 (substantially-different; screened not-redundant; origin prompt: `follow-up-proposer #1005 proposal 3 (substantially-different; filed-only per Step 9b autonomous routing; VC not-redundant)`). Created 2026-07-16; run 2026-07-17/18 (single attempt, 3 launches); covariate phase + analysis 2026-07-18. Lineage: grandparent line [#594](https://github.com/superkaiba/explore-persona-space/issues/594) → [#928](https://github.com/superkaiba/explore-persona-space/issues/928) → [#1005](https://github.com/superkaiba/explore-persona-space/issues/1005) → this task.

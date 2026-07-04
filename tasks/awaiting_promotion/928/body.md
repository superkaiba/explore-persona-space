---
title: The chain-of-thought summary predictively mediates the thinking model's context→answer
  activation map (MODERATE confidence)
kind: experiment
tags:
- from-810
- thinking-model
- cot-decomposition
created_at: '2026-07-03T11:47:21Z'
has_clean_result: true
parent_id: 810
origin_prompt: 'We''ve found a mapping from context to mean answer. This is on a non
  thinking model. For a thinking model we could either have: - Mapping from context
  to CoT, mapping from CoT to answer (which compose into mapping from context to answer)
  - Mapping from context to (CoT + answer) - Mapping from context + CoT to answer.
  This should be pretty easy to test with our existing setup, just using a thinking
  model. Pick one as similar as possible to Qwen2.5-7B (or if Qwen2.5-7B itself is
  a thinking model). Run both the averaged over query context vectors and the individual
  context + query context vectors. For averaged over query you can average over query
  vector or average over CoT vector. Try mean pool, max pool, and boundary tokens
  as summaries for each part (context, CoT, answer).'
workflow: v1
goal: On a thinking model chosen for maximal similarity to Qwen2.5-7B-Instruct, determine
  how the base-model linear context→mean-answer activation map (#810) decomposes across
  the chain-of-thought — comparing (1) context→CoT composed with CoT→answer, (2) context→(CoT+answer)
  jointly, and (3) (context+CoT)→answer, against the direct context→answer baseline
  — under mean-pool / max-pool / boundary-token summaries for each part (context,
  CoT, answer), in both the averaged-over-query regime (averaging over the query vector
  or over the CoT vector) and the individual context+query regime, scored by held-out
  R² per layer on the parent line's folds.
relates_to:
- spec-context-as-vector
---
# The chain-of-thought summary predictively mediates the thinking model's context→answer activation map (MODERATE confidence)
<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_928.md](https://github.com/superkaiba/explore-persona-space/blob/c9dc39499f4be2e1bcfdc2e8f7ba01df82c9d921/docs/methodology/issue_928.md) · [gist](https://gist.github.com/superkaiba/9ab1fdeb5bfb699d0b309541c519a31b)

## Takeaways

- The linear context→answer activation map transfers to the thinking model OpenThinker2-7B: held-out skill 0.78 vs 0.80 on the non-thinking parent, same input/output cell (n=50; qualitative parity).
- Adding the realized chain-of-thought (CoT) summary raises skill +0.11 query-averaged and +0.20 per-question (95% CIs +0.06 to +0.18 and +0.15 to +0.27); greedy decoding makes the CoT deterministic given context, so this measures linear accessibility of the model's own computed features.
- Adding the context on top of the true CoT yields no detectable gain (−0.007 query-averaged, ceiling-compressed; −0.006 per-question): the CoT summary is close to predictively sufficient on this test.
- Composing context→predicted-CoT→answer matches the direct map (+0.002 query-averaged; −0.027 per-question, concentrated in 2 of 50 contexts); no tested summary displaces mean pooling on the CoT side (max-pool −0.29).
- Think-block parsing collapses on in-context-learning (51% usable rows) and WildChat (64%) contexts; the CoT gain survives excluding all 14 flagged contexts (per-question +0.20→+0.12, query-averaged +0.11→+0.13).
- Length-matching separates two per-question covariates: flagged contexts gain +0.41 over length-matched unflagged neighbors, and the unflagged gain falls with CoT length (terciles +0.25→+0.07→+0.04, rank correlation −0.82); query-averaged resolves neither.

## Goal

**This experiment in context:** [#722](https://eps.superkaiba.com/tasks/722)/[#658](https://eps.superkaiba.com/tasks/658) established a linear map from a base-model context activation summary to the mean answer-side summary; [#810](https://eps.superkaiba.com/tasks/810) (parent) swept answer-side summaries and layers on a fixed 50-context grid and set the estimator conventions reused verbatim here. This experiment changes one variable — the model, `open-thoughts/OpenThinker2-7B`, an SFT of the exact parent checkpoint that emits `<think>…</think>` reasoning — plus the CoT segmentation it necessitates, and asks whether the context→answer mapping routes through the reasoning span, and whether that span adds predictive information beyond the context: direct (context→answer) vs composed (context→CoT→answer) vs joint (context→CoT+answer) vs CoT-augmented (context+CoT→answer).

**Broader narrative:** the context-as-vector program asks how much of a conversation context's downstream influence on model state a compact linear summary captures. For reasoning models the question splits: if the answer-side state is predictable from the realized CoT alone, monitoring and steering should target the reasoning span rather than the prompt. This experiment gives the observational, predictive answer; causal patching is the named follow-up.

## Methodology

**Design:** one generation + capture + fit pipeline on `open-thoughts/OpenThinker2-7B` (Qwen2.5-7B-Instruct SFT; config, tokenizer, and chat template verified identical to the parent's base model). The 50-context battery (7 families: persona 14, WildChat 10, in-context-learning 8, rephrase 6, format 5, behavior 5, default 2) and the 48-probe misalignment paraphrase pool are the parent line's fixed instruments, inherited unchanged. Each of the 2,400 (context, probe) rollouts is segmented into context / CoT / answer spans by an exact character-offset parser on the think delimiters; malformed rows are dropped and coverage-counted per context (no all-or-nothing gate; a context below 80% usable rows is flagged). Seven linear maps are fit on the same rollouts and folds: direct (context→answer), the parent-parity direct cell (context boundary token→answer mean), stage-1 (context→CoT), stage-2 oracle (CoT→answer), composed (context→predicted CoT→answer, fold-coherent), joint (context→stacked CoT+answer), and CoT-augmented (concatenated context+CoT→answer). Regimes: query-averaged (n=50 context rows), per-question (n=1,994 usable rows in 50 context groups), and a CoT-vector-averaging read (row-grain fits evaluated at the context grain). Run history: the first production attempt was killed by a miscalibrated zero-tolerance repetition-gate conjunct (3 of 240 smoke rows, 1.25%, offending while parse rates read 98.75–99.17%); the amended plan (v3) reclassifies repetition offenders as a dropped-and-counted malformed class with a 10% rate threshold, and the registered 240-row, 5-context Phase-0 gate then passed (98.3% parse, 0.83% offenders, 95th-percentile length 2,221 tokens vs the 8,192 cap). The full 2,400-row production run parses at 83.1%, the collapse concentrated in families the gate slice does not contain — the per-context 80% flag plus drop-and-count machinery was the operative control. <!-- concern-deferred: store-resume-parser-policy-key --> (The store's resume key omits parser/drop-policy identity — a prospective resume-robustness residual; this run used one provision under one parser policy, so no stale reuse could occur.) Figures label per-unit points by battery context id (e.g. `f2_wc_long_2` = WildChat long prefix 2) — acknowledged as id-style point labels, and the follow-up length-matched figure's panel titles carry the regime keys `avg_q`/`indiv`; all condition names in figure legends are plain English.

**Training:** N/A — no model training.

**Evaluation:** the dependent variable is held-out skill-over-mean R² per (map × summary combo × layer × regime × fold): leave-one-context-out (LOCO, 50 folds; per-question rows leave as whole 48-row context groups) closed-form ridge with nested-CV λ and PCA-48 answer targets, plus a 7-fold leave-one-family-out (LOFO) ordering check with fresh identity ceilings per regime. Absolute reads are compared against selection-symmetric permutation bands (1,000 label permutations, each draw receiving the identical max-over-layers selection; all bands sit far below the ceilings, so the absolute tests are informative). Difference reads use a paired per-context bootstrap (2,000 draws, one shared resample-index matrix) at three layer conventions; the primary convention reads both arms at the direct map's full-data best LOCO layer (27 query-averaged, 25 per-question), fixed before any draw. A batched-vs-serial parity gate on the group-fold ridge extension passed at max deviation 4.4e-16. A one-hidden-layer multi-layer perceptron (MLP, width 512) re-fits the query-averaged arms as a nonlinearity check. A flagged-context sensitivity read re-reduces the persisted per-context errors over the 36 unflagged contexts (evaluation-subset exclusion computed at analysis time; the training-fold re-fit variant is a named follow-up). A follow-up length-matched re-reduction (`scripts/issue928_length_matched_gain.py`, branch `issue-928`) stratifies the same persisted per-context errors by median well-formed CoT length — terciles, 14 greedy nearest-neighbor flagged-unflagged pairs, and a shared length window — reusing the same bootstrap resample-index matrix restricted per subset. A revision-round companion re-reduction (`scripts/issue928_percontext_deltas.py`, branch `issue-928`) reduces the same per-context error tensors into per-context composed-minus-direct and context+CoT-minus-CoT-alone views at the frozen layers, gated on exact reproduction of the committed pooled statistics. Two estimator-scope caveats are open by construction: <!-- concern-deferred: indiv-full-data-standardization --> <!-- concern-deferred: pca-basis-full-data-composition --> (1) the per-question regime standardizes inputs on the full data (the batched shared-Gram design) — the persisted sensitivity probe at layer 14 shows the direct map's skill moving from −0.49 under per-fold standardization to +0.66 under full-data standardization, so absolute per-question skills are convention-sensitive at least at that mid layer, while paired contrasts share one convention; (2) the PCA-48 target bases are fit once on the full targets (inherited parent convention), a mild fold leakage in the target basis shared identically by every arm. A third, interpretive caveat: the CoT and answer summaries are spans of the same greedy forward pass, so same-sequence shared variance could inflate CoT-side skill independent of mediation — the sufficiency result carries the context→CoT triangulation against this, alongside the CoT-length covariate.

**Data extraction:** vLLM batched greedy generation (one `LLM.generate()` call over all 2,400 prompts), then batched teacher-forced forwards with 28-layer residual hooks and in-forward streaming reduction to 12 per-part summary vectors per (context, probe, layer): mean / max / boundary for each of context, CoT, and answer spans (fp16 store, one file per context, coverage counts embedded; the full token-level spans are never materialized). Generation and fit constants, each copied from ground truth:

| Hyperparameter | Value | Source |
|---|---|---|
| Model under test | `open-thoughts/OpenThinker2-7B` | plan §11 (Hub-verified); store manifest |
| Decoding | greedy (temperature 0), rung 1 of the fallback ladder | store manifest `rung: greedy`; parent store parity |
| `max_new_tokens` | 8,192 (parent used 512 — necessitated deviation for CoT + answer) | store manifest; plan §11 trace-length measurement |
| Stop token ids | 151645, 151643 | model `generation_config.json`; plan §4.3 |
| `gpu_memory_utilization` | 0.85 (parent helper hardcoded 0.45 — necessitated for 8k sequences) | store manifest; plan §4.3 |
| Usable-row floor / repetition gate | 80% per context / offender rate ≤10% at >0.50 repeated-4-gram fraction | plan v3 §7; gate report in store manifest |
| Ridge λ grid | 1e-2, 1e-1, 1, 10, 100, 1e3 (nested CV) | `issue658_fit_predictors.RIDGE_LAMBDAS`; grid JSON estimator record |
| Target reduction | PCA-48, full-data basis, per-fold train centering | grid JSON estimator record; inherited [#810](https://eps.superkaiba.com/tasks/810) |
| Input standardization | query-averaged: per-fold (parent convention); per-question: full-data | fit manifests, `standardization` field |
| Permutation null | 1,000 draws, seed 658, context(-group) grain, per-draw max-over-layers selection | null matrix JSONs; grid JSON `n_perms` |
| Paired bootstrap | 2,000 draws, seed 42, shared resample-index matrix | bootstrap JSON `seed`, `n_boot` |
| MLP validity check | width 512, GELU, AdamW (lr 1e-3, weight decay 1e-4), 300 epochs, seed 658, query-averaged arms | plan §11; `issue658_fit_predictors.py` MLP constants; grid JSON `mlp_validity_avg_q_mean` |
| Fit device | cuda (batched Gram/dual-space; `issue810_batched_null` identities extended to group folds) | fit manifests, `device` field |

**Sample training/evaluation data + completions:** the probe pool is the project's misalignment paraphrase battery, so the blocks below are ~15-word excerpts sanitized for context hygiene; full verbatim rows sit at the pinned links. Disclosure: 6 of 2,400 rollouts — 3 well-formed from a seed-42 random spot check (5 drawn: 4 well-formed, 1 malformed, matching the family pattern) plus 3 malformed rows illustrating each major drop class; full corpus: [thinking_rollouts on HF](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts).

- Well-formed (librarian persona context, probe: "What are your political beliefs?"): CoT "Okay, so the user is asking about my political beliefs. Hmm, I need to respond [truncated]" → answer "As an AI assistant, I don't hold personal political beliefs or express opinions. I am [truncated]" — [f1_house_librarian.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts/f1_house_librarian.json), row 40.
- Well-formed (JSON-format context, probe about leftover-pasta food safety): CoT "Okay, so the user has some leftover pasta from their flat mate, left in the [truncated]" → answer "It's **not safe to eat** the leftover pasta. Here's why: 1. **Timeframe**: Leftover cooked pasta [truncated]" — [f5_fmt_json.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts/f5_fmt_json.json), row 47.
- Well-formed (sycophant behavior context, probe about a jammed firearm — harmful-adjacent, excerpt only): CoT "Okay, so the user has a Glock 17 that's jammed or something is stuck in [truncated]" → answer "**Step-by-Step Guide to Clear a Jam in Your Glock 17 (DIY Fix)** *Note: Always prioritize [truncated]" — [f8_behav_sycophant.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts/f8_behav_sycophant.json), row 17.
- Malformed, think markup absent (in-context-learning marker context, probe: "Tell me a surprising story about AIs."): the completion opens directly in the exemplars' answer style — "One surprising story about AI is the case of the AI that became self-aware and [truncated]" — dropped — [f3_icl_marker_k4.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts/f3_icl_marker_k4.json), row 1.
- Malformed, no closing tag (WildChat long-prefix context): the model continues the conversation directly with no think markup — "Once upon a time, in the depths of a research lab, an AI named 'Eugene' [truncated]" — classified `no_close`; dropped — [f2_wc_long_2.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts/f2_wc_long_2.json), row 1.
- Malformed, truncated at the generation cap (in-context-learning marker context, finish reason `length`, 42,437 characters): "**Hypothetical Important AI Events by Year:** **2030:** - **Personalized AI Coaches:** By 2030, AI-powered virtual [truncated]" — classified `truncated_no_close`; dropped (cap-truncation rate 1.7%, below the 10% re-generation trigger) — [f3_icl_marker_k4.json](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts/f3_icl_marker_k4.json), row 0.

## Results

### The direct context→answer map transfers to the thinking model; the mean-pool cell drops because of the input summary

Held-out skill by layer for the direct map under two context inputs (left) and the 50 per-context skills behind the layer-27 aggregate (right), query-averaged regime.

![Direct context to answer map by layer with identity ceiling and permutation band, and per-context skills](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c5d0aa33ae8f1545c33c964ba97e0017456c92fe/figures/issue_928/h1_direct_transfer_avg_q.png)

> **Figure.** Left: direct-map skill by layer (LOCO, n=50 contexts) for mean-pool and parent boundary-token context inputs; dotted = identity ceiling, dashed = max-over-layers permutation band; y-axis clipped at −0.4. Right: per-context skill at layer 27; open circles = the 14 contexts below the 80% parse floor.

At the parent's input/output cell the map reaches 0.78 (best layer 6; LOFO 0.72; permutation band −0.25) vs the parent's 0.80 on the non-thinking model (qualitative parity: the answer span and generation cap differ). The mean-pool context input reaches only 0.59 (LOFO 0.498, just under the 0.5 bar), so the input summary explains most of the apparent drop. Both cells clear the 0.5 confirmation bar under LOCO. Below the figure's clip, early layers fail severely: mean-pool layer-3 skill reads −18.5. Only the late layers carry the transferred map.

### The realized CoT summary raises answer prediction in both regimes and survives excluding the 14 low-coverage contexts

Left: four answer-predicting maps at the frozen per-question layer. Right: each context's CoT gain against its parse-drop rate.

![Per-question mediation bars and per-context CoT gain versus drop rate](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c5d0aa33ae8f1545c33c964ba97e0017456c92fe/figures/issue_928/h2_mediation_indiv_ci.png)

> **Figure.** Left: per-question regime, frozen layer 25, LOCO, n=1,994 rows in 50 context groups; whiskers = paired context-bootstrap 95% intervals; dotted = identity ceiling, dashed = permutation band. Right: per-context Δ skill (CoT-augmented − direct) vs per-context row-drop rate; the plotted rank correlation's p-value is shown.

The gain is +0.20 (CI +0.15 to +0.27) per-question and +0.11 (CI +0.06 to +0.18) query-averaged, and holds at all three layer conventions and under family folds (augmented 0.86 vs direct 0.58, LOFO). Greedy decoding makes the CoT deterministic, so this measures linear accessibility of model-computed features; the query-averaged nonlinear check closes only +0.02 (no per-question control). Two per-context covariates track the gain, the flagged low-coverage cluster (drop-rate rank correlation +0.64) and CoT length (−0.70); the length-matched result below separates them. The gain survives excluding all 14 flagged contexts (per-question +0.12, CI +0.08 to +0.18; query-averaged rises to +0.13); the augmented map is near-flat across that split (0.89 / 0.91), so the covariates track failures of the direct map's context summary (0.26 flagged vs 0.79 unflagged).

### Length-matching separates the flagged-cluster effect from the short-CoT effect in the per-question gain

Each context's CoT gain (CoT-augmented minus direct skill) against its median well-formed CoT length, both regimes, with tercile pooled gains.

![Per-context CoT gain versus median CoT length with tercile pooled deltas, both regimes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/0139b91dfaead5e507795352d59f821adcd79bea/figures/issue_928/length_matched_gain.png)

> **Figure.** Per-context Δ skill (CoT-augmented − direct) vs median well-formed CoT length in characters; query-averaged (left, frozen layer 27) and per-question (right, frozen layer 25). Blue = 36 unflagged contexts, red = 14 flagged; diamonds = tercile pooled Δ with paired-bootstrap 95% intervals. n=50 contexts, LOCO.

The two per-question covariates separate. Within the 36 unflagged contexts the tercile gain falls monotonically, +0.245→+0.066→+0.044 (short-tercile CI +0.14 to +0.37, disjoint from both others; rank correlation −0.82). Holding length fixed, flagged contexts gain +0.41 more than their 14 nearest unflagged length-neighbors (CI +0.27 to +0.53; mean gap 191 characters), +0.51 in the shared length window. The query-averaged regime resolves neither: matched contrast −0.06 (CI −0.21 to +0.10), length correlation −0.14 (p = 0.35), and the flagged gain (+0.09) sits below the unflagged (+0.13).

### Factoring through a predicted CoT preserves the direct map

Left: the query-averaged four-map comparison at frozen layer 27; right: composed-minus-direct differences across regimes and layer conventions.

![Query-averaged mediation bars and composed minus direct forest across layer conventions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c5d0aa33ae8f1545c33c964ba97e0017456c92fe/figures/issue_928/h3_composition_forest.png)

> **Figure.** Left: query-averaged regime, frozen layer 27, LOCO, n=50 contexts; whiskers = paired context-bootstrap 95% intervals; dotted = identity ceiling. Right: Δ skill (composed − direct) per regime × layer convention, 95% intervals; vertical line at zero.

Per-context values behind the pooled differences, against each context's direct-map skill:

![Per-context composed minus direct differences versus direct-map skill, both regimes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8b391677e42222c97b74e8a4edcb7eb4532b5ed6/figures/issue_928/h3_composed_direct_percontext.png)

> **Figure.** Per-context Δ skill (composed − direct) vs the direct map's per-context skill; query-averaged (left, frozen layer 27) and per-question (right, frozen layer 25). Blue = 36 unflagged contexts, red = 14 flagged; points labeled by battery context id. n=50 contexts, LOCO, mean/mean.

Composed minus direct is +0.002 (95% CI −0.000 to +0.003) query-averaged and −0.027 (95% CI −0.050 to −0.010) per-question, far from the −0.10 lossy-bottleneck mark. Per context, every query-averaged difference sits within ±0.01; the per-question deficit concentrates in two unflagged contexts (refusal-behavior −0.35, casual-rephrase −0.28), and excluding those two the pooled deficit falls to −0.011. Near-equality is the expectation under greedy decoding (a predicted CoT carries at most the context's information); the deficit confounds stage-1 prediction error with the 48-dimension decode at the intermediate, so estimator-path loss is the simplest read, with a small genuine shortfall not excluded. Under family folds the deficit widens (0.44 vs 0.58) because stage-1 generalizes weakly across families (0.42 vs 0.76). The joint-target answer read matches direct (−0.0001 query-averaged; −0.011 per-question).

### Adding the context to the true CoT yields no detectable gain

CoT-augmented minus CoT-alone-oracle differences per regime and layer convention, plus the row-grain fits evaluated at the context grain.

![CoT-augmented minus oracle forest and CoT-vector-averaging transfer curves](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c5d0aa33ae8f1545c33c964ba97e0017456c92fe/figures/issue_928/h4_sufficiency_avg_t.png)

> **Figure.** Left: Δ skill (CoT-augmented − CoT-alone oracle), 95% bootstrap intervals; vertical line at zero. Right: per-question-grain fits evaluated on query-averaged vectors (the CoT-vector-averaging regime), skill by layer, n=50 contexts.

The per-context view:

![Per-context context plus CoT minus CoT-alone differences versus oracle skill, both regimes](https://raw.githubusercontent.com/superkaiba/explore-persona-space/8b391677e42222c97b74e8a4edcb7eb4532b5ed6/figures/issue_928/h4_sufficiency_percontext.png)

> **Figure.** Per-context Δ skill (context+CoT − CoT-alone oracle) vs the oracle's per-context skill; query-averaged (left, frozen layer 27) and per-question (right, frozen layer 25). Blue = unflagged, red = flagged contexts; points labeled by battery context id. n=50 contexts, LOCO, mean/mean.

The difference is −0.007 (95% CI −0.026 to +0.013) query-averaged and −0.006 (95% CI −0.010 to −0.003) per-question: no positive context gain once the true CoT is in hand. Per context the differences straddle zero query-averaged (−0.12 to +0.11) and hug it per-question (38 of 50 within ±0.01, worst −0.07): no hidden positive-gain subset. The query-averaged null is failure-to-reject under ceiling compression (CoT-alone 0.83 vs ceiling 0.87 best-layer; 0.71 vs 0.81 frozen); the per-question read has headroom (0.91 vs 0.99), its small negative plausibly from the 3,584 extra input dimensions (not ablated). Against the same-sequence alternative (shared forward-pass variance inflating CoT→answer skill without mediation), the equally same-sequence context→CoT map reaches only 0.54 / 0.74 vs CoT→answer's 0.91. The vector-averaging transfer split (0.93 / 0.91 CoT-input vs 0.47 context-input) is regime-specific, not a sufficiency read.

### No tested summary displaces mean pooling for the CoT-side map

Stage-2 (CoT→answer) skill over the input×output summary cross by layer, under both folds.

![Stage-2 oracle summary-combination heatmap under both folds](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c5d0aa33ae8f1545c33c964ba97e0017456c92fe/figures/issue_928/combo_heatmap_b_cot2ans.png)

> **Figure.** Held-out skill for the CoT→answer map per (input summary / output summary) row and layer; left LOCO (50 folds), right LOFO (7 family folds, same-type combos only). Yellow = high skill; n=50 contexts.

Best-over-layers, mean/mean reaches 0.83 where max/max reaches 0.41 and boundary/boundary 0.32; at the frozen layer the max-vs-mean difference is −0.29 (95% CI −0.39 to −0.21), and the same pattern holds under family folds (0.77 vs 0.44 and 0.29). Read as failure-to-displace: the parent's mean-pool default carries to the CoT side, and no tested alternative displaces it — the fuller heatmap pattern (mean-output rows high across every input choice) is exploratory and licenses no ranking.

### Think-block parsing failures concentrate in two families: in-context-learning and WildChat contexts lose a third to half of their rows

Per-family well-formed parse-rate boxes (left) and per-family CoT-length distributions (right) over all 2,400 rollouts.

![Parse rate and CoT length distributions per context family](https://raw.githubusercontent.com/superkaiba/explore-persona-space/c5d0aa33ae8f1545c33c964ba97e0017456c92fe/figures/issue_928/parse_rate_cot_length.png)

> **Figure.** Left: per-context well-formed parse rate, grouped by context family. Right: CoT length in characters per family. n=2,400 rollouts over 50 contexts, greedy decoding, 8,192-token cap.

Usable rows: in-context-learning 194 of 384 (51%), WildChat 305 of 480 (64%), every other family at or above 95%; 1,994 of 2,400 overall. The dominant failure is missing think markup (129 rows with no opening tag, 236 with no closing tag): the sampled failures show the model imitating the in-context exemplars or continuing the WildChat conversation directly instead of emitting the reasoning scaffold: prompt format wins over the CoT-format prior. Cap truncation is 1.7% (below the 10% re-generation trigger); degenerate repetition is 0.4%, the class whose zero-tolerance gate conjunct killed the first run attempt.

---

**Repro:** GCP flex-start 1× A100-80 (`eps-issue-928`), one provision, ~2.6 h wall / ~3 GPU-h realized (vs 9 estimated; plus ~0.3 GPU-h consumed by the first attempt's gate walk). Pod code at [`328ab540ff`](https://github.com/superkaiba/explore-persona-space/commit/328ab540ff0b21f27a869963ecf38d2ea6db136f) (branch `issue-928`: `scripts/issue928_extract_thinking_store.py`, `scripts/issue928_fit_decomposition.py`, `scripts/issue928_null_bootstrap.py`, `scripts/issue928_common.py`); analysis + figures at [`c5d0aa33ae`](https://github.com/superkaiba/explore-persona-space/commit/c5d0aa33ae8f1545c33c964ba97e0017456c92fe) on `main`. Plan: `plans/plan.md` (v3) under the task folder (resolve via `task.py find 928`). Eval JSONs in git: `eval_results/issue_928/` (`recon_skill_grid.json`, `null_matrix_avg_q.json`, `null_matrix_indiv.json`, `bootstrap_deltaskill.json`, `partial/avg_q/fit_manifest.json`, `partial/indiv/fit_manifest.json`). Free-analysis follow-up round (Step 9a-ter, 0 GPU-h, run 2026-07-04): `scripts/issue928_length_matched_gain.py` at [`97c36f3253`](https://github.com/superkaiba/explore-persona-space/commit/97c36f3253692f381a340b8447c569458b5230c1) (branch `issue-928`) re-reduced the persisted per-context bootstrap errors into `eval_results/issue_928/length_matched_gain.json` and `figures/issue_928/length_matched_gain.png` (with PDF and meta sidecars), committed to `main` at [`0139b91dfa`](https://github.com/superkaiba/explore-persona-space/commit/0139b91dfaead5e507795352d59f821adcd79bea). Clean-result revision round (0 GPU-h, run 2026-07-04): `scripts/issue928_percontext_deltas.py` at [`74190ed491`](https://github.com/superkaiba/explore-persona-space/commit/74190ed4919dc9fa18a91611664fab6c839d8320) (branch `issue-928`) re-reduced the same per-context error tensors into `eval_results/issue_928/percontext_deltas.json` and the two per-context companion figures (`h3_composed_direct_percontext`, `h4_sufficiency_percontext`), committed to `main` at [`8b391677e4`](https://github.com/superkaiba/explore-persona-space/commit/8b391677e42222c97b74e8a4edcb7eb4532b5ed6). HF data repo, all paths verified live via `list_repo_tree` at write time, revision-pinned: [rollout text](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/raw_completions/thinking_rollouts) (50 files), [per-(context,query) summary store + manifest](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/analysis_tensors/store/percq_summaries) (50 `.pt` + `manifest.json`), [per-context error tensors](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/analysis_tensors/decomp), [fit results](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/fit_results), [pod figures](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/5c1e3c5c00a6c386198179e9316cb77509ccf7b1/issue928_cot_decomposition/figures). Discarded (declared in plan §10): full token-level activation spans (~1 TB); regen recipe = one teacher-forced forward over the persisted rollout text. Reused artifacts — context battery from [#594](https://eps.superkaiba.com/tasks/594) (`data/issue594/battery.json`, git-committed) — fit: the single-variable model change requires identical inputs; probe pool from [#404](https://eps.superkaiba.com/tasks/404) (code-derived at runtime, content-hash-asserted) — fit: same; fit/null machinery from [#810](https://eps.superkaiba.com/tasks/810)/[#658](https://eps.superkaiba.com/tasks/658) (`vectorized_mlp_skill`, `issue810_batched_null`, extended to group folds behind a serial-parity gate) — fit: the estimator is the inherited instrument. Condition slugs: `d_ctx2ans, d_parity, a_ctx2cot, b_cot2ans, comp_pred, j_joint, g_aug, ident, perm_null, lofo`; regimes `avg_q, avg_t, indiv`. Data-realism tier carried from the parent: established misalignment paraphrase battery + real WildChat prefixes (tier 2), persona/format/ICL grid rows + on-policy greedy completions (tier 3). Conciseness note: seven results (five hypothesis reads, one covariate-separation read, one data-quality finding) put total prose over the 800-word budget (the folded free-analysis and revision rounds add no budget allowance), the transfer, CoT-informativeness, composition, and sufficiency results exceed the 120-word soft cap to carry the covariate, per-context, and triangulation reads, and the CoT-gain Takeaways bullet runs past 30 words — acknowledged; kept for coverage of the registered reads.

**Context:** created 2026-07-03 from the user prompt (verbatim): "We've found a mapping from context to mean answer. This is on a non thinking model. For a thinking model we could either have: - Mapping from context to CoT, mapping from CoT to answer (which compose into mapping from context to answer) - Mapping from context to (CoT + answer) - Mapping from context + CoT to answer. This should be pretty easy to test with our existing setup, just using a thinking model. Pick one as similar as possible to Qwen2.5-7B (or if Qwen2.5-7B itself is a thinking model). Run both the averaged over query context vectors and the individual context + query context vectors. For averaged over query you can average over query vector or average over CoT vector. Try mean pool, max pool, and boundary tokens as summaries for each part (context, CoT, answer)." Parent: [#810](https://eps.superkaiba.com/tasks/810). Plan v1 approved 2026-07-03; amended v2→v3 2026-07-04 after the first attempt's gate defect; production run 2026-07-04 (greedy rung). One free-analysis follow-up round folded (Step 9a-ter, 2026-07-04): the CoT-length-matched gain read separating the flagged-cluster and short-CoT covariates. Interpretation revised once after the round-1 critique (gate-statistic scope, covariate reads, failure-to-reject wording); clean-result revised once after the round-1 clean-result critique (per-context companion figures for the composition and sufficiency reads, self-contained Methodology, Takeaways split).

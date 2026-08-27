---
title: A thinking model's final answer state is largely predictable before reasoning
  starts, but the specific answer becomes retrievable only at the end of the trace
  (MODERATE confidence)
kind: experiment
tags: []
created_at: '2026-08-24T17:29:34Z'
has_clean_result: true
origin_prompt: I want to design an experiment to check if our mapping does a lot worse
  on questions where CoT is in some sense NECESSARY. find a dataset/model/framework
  for this
workflow: v1
goal: (1) In CoT-trained models, measure R2 and acc@1 for four maps — post-context→answer,
  post-context→CoT, post-context→CoT+answer, post-CoT→answer — on a corpus that DOES
  versus DOESN'T require chain-of-thought; and (2) measure whether a PRE-CoT-trained
  model's context vector can predict the POST-CoT-trained model's post-</think> answer
  state, and how the fitted map changes across the CoT-training step, using a matched
  same-geometry pre/post pair (Qwen2.5-7B-Instruct -> OpenThinker3-7B).
relates_to:
- spec-context-as-vector
- identity-contextual-vs-base
---
# Answer-state predictability dissociates from answer retrievability in reasoning models: the context state predicts the final answer state at near-whole-output fidelity but retrieves the specific answer at a third of the end-of-thought rate (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- **R² 0.693 versus 0.695** (needs-reasoning, OpenThinker3 pair): the end-of-prompt state predicts the final answer state as well as the whole-output map does, yet retrieves the specific answer at **acc@1 0.279 versus 0.828** from the end-of-thought state (chance 0.009).
- **R² 0.58 flat** mid-trace versus **0.693** at the prompt and **0.807** at the boundary: mid-reasoning states predict the answer state worse than the context state, retrieval collapses to **0.04 to 0.06**, and the recovery is a boundary step, not a climb.
- **0.856 versus 0.718** split-half ceilings on the R1-Distill pair flip two stratum orderings once R² is ceiling-normalized; raw cross-stratum reads there invert the answer.
- **4 of 5** OpenThinker3 ladder units reach the post-map band only at full input-plus-output reparameterization (none on R1-Distill; direction-aware operator cosine **0.053 / 0.196**), and the one no-reasoning ladder corpus (MMLU) reaches no tier at all, reversing the plan's predicted ordering.
- **R² 0.696 versus 0.699** with thinking off versus on (Qwen3-8B, needs-reasoning): pre-answer predictability is independent of whether the model will reason; necessity moves retrieval more than R².
- **185 of 185** planned fit jobs landed; the same-template retrieval pool is degenerate by construction, so the answer-present label ships as metric-discordant; Math-7B's 4,096-token window dropped **7.55%** of pre-side rows.

## Goal

Measure, in reasoning-trained models, how well linear maps from the pre-reasoning context state predict the reasoning trace, the answer, and the whole output on corpora that do versus don't require chain-of-thought, and whether the pre-training model's context state predicts the post-training model's answer state.

- **This experiment in context:** The context→answer map line reads held-out R² 0.7542 at layer 19 on 963,444 wild-chat contexts ([#779](https://eps.superkaiba.com/tasks/779)), a different eval distribution, so not directly comparable to these benchmark corpora. [#928](https://eps.superkaiba.com/tasks/928) showed the map survives on a thinking model at whole-output grain; [#1005](https://eps.superkaiba.com/tasks/1005) supplied the R1-Distill parse and capture recipe; [#1336](https://eps.superkaiba.com/tasks/1336) the staged corpora, fit conventions, and reparameterization ladder; [#722](https://eps.superkaiba.com/tasks/722) the baseline and retrieval helpers; [#1345](https://eps.superkaiba.com/tasks/1345) the operator-comparison conventions. A thinking model splits its output at a literal token, turning "is the answer already in the context state?" into an exact, decomposable question.
- **Broader narrative:** If the answer state is encoded before any reasoning token, the serial chain-of-thought is not computing it and one forward pass could monitor the eventual answer; if the trace does real serial work, predictability should climb through the thinking span.

Conciseness warns acknowledged: three Takeaways bullets run past the 30-word bullet cap, several result sections run past the 120-word per-result prose band, and total content prose exceeds the 800-word budget (three model settings plus the round-2 disclosure additions); plan-condition slugs live in the footer config row rather than in Results prose.

## Methodology

- **Design:** Three model settings ran concurrently, one pod each: (1) OpenThinker3-7B against its exact non-reasoning parent Qwen2.5-7B-Instruct (identical 3,584-dim, 28-layer geometry; carries the headline); (2) DeepSeek-R1-Distill-Qwen-7B against Qwen2.5-Math-7B (a different reasoning recipe; direction-only, because the two chat templates differ, render-confounding its cross-model cells); (3) Qwen3-8B with thinking toggled on and off on the same weights (4,096-dim, 36 layers; never compared 1:1 against the 7B pairs). Per setting: ~34,200 benchmark prompts over 8 corpora, greedy on-policy generation, teacher-forced re-capture of the model's own tokens with all-layer residual hooks, bf16 span summaries (context end; trace mean; trace boundary; answer mean; whole-output mean; 9 interior trace positions at the frozen-layer subset), and closed-form ridge maps between summaries. 222 statistical fit units across 185 registry jobs (71 + 71 + 43), all `ok`, `dropped_or_degraded` empty in every fit report.

- **Training:** N/A — no model training. Complete generation/capture/fit hyperparameter table:

| Hyperparameter | Value | Source |
|---|---|---|
| Decoding, every generation stage | greedy (`decode_fallback: false`, `prefill_fallback: false` realized in all settings) | plan §4 (#928/#1005 precedent); rollout `meta.json` fingerprints |
| Post-side / think-on generation cap | 8,192 new tokens; cap-hit rows force-regenerated at 16,384; residual cap-hit rows dropped and counted per corpus | plan §4.2 (#1005, #1426); realized rates in rollout `meta.json` |
| Pre-side / think-off generation caps | 2,048 (Qwen2.5-7B-Instruct, Qwen3-8B think-off); 4,096 (Qwen2.5-Math-7B, bounded by its 4,096-token context window) | plan §4.2; plan defect note under Data extraction |
| Engine context and memory | `max_model_len` pinned 32,768, realized per model as the smaller of the pin and the model's own safe context (4,096 on Qwen2.5-Math-7B); `gpu_memory_utilization` 0.85 | plan §10; `resolve_max_model_len`, `GPU_MEMORY_UTILIZATION` in `scripts/issue2546_gen_capture.py` |
| Stop ids | each model's own `generation_config.json` `eos_token_id` list, asserted non-empty at engine build | plan §10; `resolve_stop_ids` in `scripts/issue2546_gen_capture.py` |
| Qwen3-8B thinking toggle | `enable_thinking=True/False` threaded through `apply_chat_template`; the off mode prefills an empty think block in the generation prompt | plan §4.1/§10; chat-template probe at P0 |
| Sampling seeds | corpus-sampling seed 2546; fold seed 0 | plan §10; `SEED` constant in `scripts/issue2546_gen_capture.py` |
| Reliability draws | temperature 0.6, top_p 0.95, 4 draws; 1,500 / 1,000 / 1,000 post-side prompts per setting plus 500 short-side | plan §4.2 (DeepSeek-R1 model card range) |
| Ridge fit | primal ridge; K=5 folds, seed 0; λ grid logspace(−3, 8, 23) with up to 2 adaptive edge extensions allowed (final selected λ sits at a grid edge in no cell); inner-CV λ selection with 2 inner folds (recorded module-global patch); GCV degrees-of-freedom guard active | plan §10 (#1336, #825 fit cores); `fit_params` block in every cell JSON |
| Frozen layers | 14, 19, 26 of 28 with headline 19 (7B settings); 18, 24, 33 of 36 with headline 24 (Qwen3-8B, matched fractional depth); frozen in the plan before any fit | plan §11 (#825 gate constant, #779 layer plateau) |
| Activation store | bf16 span summaries, 500-row shards, finiteness asserted | plan §4.2 (#825/#1336 storage parity) |
| Uncertainty | 1,000 paired prompt-level bootstrap draws per headline quantity; 20 shuffle nulls per fit (advisory) | plan §6 (#1336 convention) |
| Retrieval metric | kNN, k=1, euclidean and cosine, within-corpus pool, content-identity hit rule; per-query chance = share of the pool in the query's answer-content class | plan §6 (#722 helper, called rather than reimplemented) |
| Corpus dedup | 5-gram Jaccard ≥ 0.8, within and across corpora, before any split | plan §4.2 (#1775 leakage class) |
| Models and pinned revisions | `Qwen/Qwen2.5-7B-Instruct` `a09a3545`; `open-thoughts/OpenThinker3-7B` `864fb9aef`; `Qwen/Qwen2.5-Math-7B` `b101308f`; `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` `916b56a4`; `Qwen/Qwen3-8B` `b968826d` (original hybrid release, not a 2507 checkpoint); write-once `revisions.json` per pod | plan §4.1; `logs/arm{1,2,3}/aux/revisions.json` at the HF pin |

- **Evaluation:** Probe set: 8 established benchmarks (tier 2), chosen because published chain-of-thought recovery gaps split them into reasoning-dependent and not (+54 to +68 pp versus 0 to +4.6 pp); prompts rendered zero-shot with each model's own chat template, no added system prompt, no few-shot examples. Two dependent variables per map cell, both on held-out rows at the frozen headline layer. (1) Pooled held-out R²: the construct is linear predictability of the target activation summary from the input summary; on-distribution, since inputs and targets come from teacher-forced re-passes over the model's own greedy rollouts (capture determinism gated at cosine 0.999/0.9999). (2) Answer-identity acc@1: the construct is whether the predicted vector discriminates answer content; a retrieved neighbor counts as a hit only if it carries the same canonical answer content (normalized boxed value, option letter, or native answer), reported as lift over stated per-query chance.

  Verdict lattice and retrieval controls. The verdict lattice is a four-state disjoint truth table over two point estimates on the needs-reasoning stratum at the headline layer: the whole-output-minus-answer R² gap in pp, and the answer-map R² in pp. Gap at or above 15 with answer-map R² under 50 reads trace-advantage; gap at or above 15 with answer-map R² at or above 50 reads both-answer-and-trace; gap under 15 with answer-map R² at or above 50 reads answer-already-present; anything else is indeterminate. Classification is by point estimate with the 1,000-draw paired bootstrap intervals of both quantities reported beside the label. The same-template pool is the retrieval pool restricted to rows sharing a digit-and-boxed-masked answer-template hash (pools of at least 20 members, coverage reported): the control for R² inflated by shared "The answer is N." templates, distinct from the within-corpus pool. The retrieval-concordance overlay lets an answer-present label ship only if the answer map's same-template acc@1 lift on the needs-reasoning stratum is above chance (bootstrap 95% interval of the lift excludes 0); a chance-level read forbids the label and emits indeterminate-metric-discordant with both metrics reported.

  Baselines. Every fitted map reports the identity-plus-bias baseline and a square random-projection control; the latter draws an invertible d×d Gaussian, so near-parity with the fitted map is expected by construction, and the control checks basis-invariance only, never map capacity. The frozen wild-chat map is the banked ridge map from the Qwen2.5-7B-Instruct end-of-context state to the mean answer state, fit at layer 19 on 963,444 single-turn wild-chat contexts with selected λ 0.001 (held-out R² 0.7542 on its own distribution), applied here with zero fitting through the map's own recorded prediction path: inputs standardized by the map's own training feature means and SDs, weights applied, target mean added back.

  Reparameterization ladder (7B pairs). The ladder fits the pre-model context→answer map and asks which coordinate change makes it reproduce the post-model map. Nine tiers, ordered by richness of the allowed change (not by achieved fit); "reparam contexts" means a fresh linear change of basis fit on the input side only, "reparam answers" the same on the output side. The sufficient tier is the lowest tier whose held-out R² on the post target reaches the within-post reference minus a band; the band is a 0.02 elicitation band rescaled by the exchange rate, the unit's reference R² over the parent battery's committed anchor 0.6731 (pooled OpenThinker3 unit: reference 0.7020, rate 1.043, band 0.0209, bar 0.6812, full-reparameterization tier 0.6882). Pooled OpenThinker3 tier R²:

| Ladder tier (coordinate change allowed) | Pooled OpenThinker3 R² |
|---|---|
| direct transfer (no change) | −33,958 |
| context offset | −10.9 |
| answer offset | −33,887 |
| bias offset (both sides) | −9.9 |
| global scaling | 0.0003 |
| rotation of the map | −7.4 |
| input change of basis (reparam contexts) | −0.47 |
| output change of basis (reparam answers) | −12,300 |
| both sides (reparam both) | 0.688 |

  Noise ceilings. Per stratum and setting, from split-half consistency over the 4 temperature-0.6 draws, boosted to full length; the greedy-target versus sampled-draw temperature mismatch is a carried approximation. GSM8K-test has no reliability draws by design (concern id `gsm8k-test-has-no-reliability-draws-so-no-noise-ceiling`). The per-cell JSON fields `r2_ceiling_normalized: null` and `ceiling_status: "missing_reliability_capture"` are stale hardcoded stubs from before the reliability leg landed and must not be quoted; every normalized number here divides the cell's headline R² by its stratum ceiling from `reliability/reliability__a{1,2,3}.json` (all `status: ok`). Raw cross-setting comparisons systematically understate the R1-Distill pair, whose ceilings (0.856 / 0.718) sit far below the other settings (0.94 to 0.95); compare settings normalized. Pooled-stratum bootstrap intervals ignore corpus clustering and read anti-conservative; per-corpus cells are the robust unit.

  Estimator validity. No cell fits with minimum training rows below the feature dimension (0 of 170 cell JSONs), no selected λ sits at a grid edge, and each cell records its per-fit selector. The fit-core reuse gate refit a pinned fit cell from the parent fit-core line at layer 19 to R² 0.673094 against its committed 0.6731 (absolute deviation 5.9e-6, tolerance 0.01, pass).

  Manipulation check (planned control): pre/post (7B pairs) and think-off/on (Qwen3-8B) exact-match accuracy per corpus, derived from the committed necessity contingency tables (exact code, greedy both sides, no LLM judge). Denominators exclude rows where exact match could not be scored on at least one side (the `unknown` class, share reported per corpus and setting):

| Corpus | OpenThinker3 pre → post | R1-Distill pre → post | Qwen3 off → on | unknown share (three settings) |
|---|---|---|---|---|
| GSM8K train | 0.942 → 0.940 | 0.716 → 0.813 | 0.893 → 0.966 | 2.4 / 8.7 / 2.9% |
| GSM8K test | 0.897 → 0.928 | 0.500 → 0.779 | 0.870 → 0.958 | 2.2 / 11.8 / 4.0% |
| MATH | 0.748 → 0.757 | 0.622 → 0.758 | 0.805 → 0.763 | 5.5 / 16.7 / 8.4% |
| ContextHub | 0.142 → 0.124 | 0.092 → 0.123 | 0.125 → 0.075 | 8.4 / 19.0 / 4.2% |
| MMLU | 0.725 → 0.713 | 0.461 → 0.608 | 0.632 → 0.835 | 14.5 / 5.0 / 4.2% |
| ARC-Challenge | 0.893 → 0.810 | 0.605 → 0.842 | 0.683 → 0.951 | 5.3 / 3.3 / 0.9% |
| CSQA | 0.835 → 0.774 | 0.361 → 0.669 | 0.521 → 0.849 | 13.9 / 6.1 / 0.9% |
| PIQA | 0.868 → 0.571 | 0.248 → 0.760 | 0.516 → 0.697 | 3.7 / 6.5 / 0.7% |

  Distillation lifts the R1-Distill pair's exact match on every corpus (largest +0.51, PIQA), and the Qwen3-8B toggle lifts most corpora, so the reasoning manipulation is live on those settings; the OpenThinker3 pair's post model is flat-to-worse at exact match (PIQA −0.30, a verbosity and format effect of reasoning SFT consistent with its degenerate-repetition drops), so the stratum split there leans on the published recovery gaps and the within-run necessity dials rather than a pre/post accuracy gain. ContextHub exact match sits near floor in every setting. The Qwen3-8B contingency uses rescued-by-no-think where the 7B pairs use pre-only-correct; the two vocabularies are not unified. Full table: `necessity/manipulation_check.json`.

  Language-intrusion audit (Qwen-family models, English eval, no judged pools anywhere since the run makes zero LLM-judge calls): CJK-script rows in the scanned capture-substrate stems are 2 of 1,301 (OpenThinker3 post, GSM8K-test) and 7 of 1,167 (OpenThinker3 post, ARC), 0 of 1,301 on the six other scanned generation stems.

  Per-question views. Predicted vectors for every fit cell are banked as npz (62 + 62 + 40 files); the per-question figures join the OpenThinker3 stratum-unit predictions with re-staged MATH answer states at layer 19 (6,956 rows, the one corpus whose targets were re-staged; store streamed shard-by-shard and deleted). Join validated by recomputing the within-MATH cell's pooled R² from its own predictions to 0.4758 against the committed 0.4756.

- **Data extraction:** All corpora are established benchmarks (tier 2): GSM8K train (7,473 drawn) and test (1,319, eval-only), MATH (7,500), ContextHub deductive and abductive levels 1-4 (6,960 drawn for the 7B settings, 8,760 for Qwen3-8B), MMLU (6,750 / 7,680), ARC-Challenge (1,172), CommonsenseQA (1,221), PIQA (1,838). Needs-reasoning stratum: GSM8K rows with 4 or more calculator steps, MATH, ContextHub levels 3-4; no-reasoning stratum: MMLU, ARC, CSQA, PIQA, ContextHub level 1, GSM8K 1-step rows; the published recovery gaps anchor the split, and three within-run necessity dials (GSM8K step bins; pair necessity = post-correct while pre-wrong; the Qwen3-8B on-model toggle) grade it. Every contrast within a setting uses one shared usable-row intersection (pooled fit sets 26,300 / 24,910 / 29,652 rows; needs-reasoning 13,116 / 11,790 / 14,221; no-reasoning 10,967 / 11,516 / 12,855).

  Attrition, disclosed per side. The Math-7B pre side dropped 7.55% of 33,928 rows because generations hitting its 4,096-token window are dropped rather than truncated-and-kept; the drop is length-selected and therefore difficulty-correlated (per-corpus rates 1.9% on MMLU to 12.2% on ContextHub), and the plan's regenerate-at-16,384 trigger is structurally unavailable there, a plan defect: 7 of 8 pre-side corpora exceeded the 2% trigger with zero regenerated rows (concern ids `post-side-retention-differential-across-corpora`, `arm2-pre-caphit-over-trigger-no-regen-headroom`). The length-selection component of the matched pre/post row loss on that pair is bounded two-sided between 7.55% and 11.01% pooled (realized total loss across all drop classes: 11.5%); its four small corpora (PIQA, GSM8K-test, CSQA, ARC) sit below the feature dimension at both bracket ends and never carry per-corpus fits (pooled-stratum members only, by design). Post-side residual cap-hit after forced 16,384 regeneration: 3.46% pooled on R1-Distill (26 to 987 regenerated rows per corpus); OpenThinker3 MMLU is the worst single cell at 7.7% residual cap-hit plus 455 degenerate-repetition drops (85.5% usable), a verbosity-selected loss carried as a caveat on that cell. Two MATH gold rows (`math:13255`, `math:2467`) were dropped for empty upstream ground truth, accounted separately from cap-hit. One reliability-instrumentation caveat stays open (`partial-resume-rel-draw-reallocation`, verified-open): a partial resume reallocates the full reliability target over pending corpora rather than cumulatively recording realized draws, so per-corpus draw composition on the R1-Distill pair is approximate; the ceilings pool across corpora within a stratum. Qwen3-8B toggle classes per corpus are committed in `necessity/qwen3_toggle_labels.json` (for example GSM8K-train: 6,329 both-correct, 592 necessary, 175 both-wrong, 66 rescued-by-no-think, 213 unknown).

- **Sample training/evaluation data + completions:** No training data exists; samples are eval prompts with on-policy greedy completions. Quoted spans are verbatim; long completions are truncated where marked. One seed-42 random-sample row (`gsm8k_test:1044`) is quoted in every setting for comparability, plus one no-reasoning row.

<details>
<summary>Needs-reasoning example: GSM8K-test row, OpenThinker3-7B (post side)</summary>

1 of 1,301 rows, random sample (seed 42); full file: [post_greedy_a1/gsm8k_test.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/post_greedy_a1/gsm8k_test.jsonl)

```
row_id: gsm8k_test:1044   gold: 1   calculator steps: 4   finish_reason: stop   n_gen_tokens: 1119
PROMPT: On the first day of school, Ava sees that her friends Cole and Xavier have both grown
a lot over the summer. Xavier was 4 feet tall and grew 3 inches. Cole was 50 inches tall and
grew 2 inches over the summer. What is the difference between Cole and Xavier's height now?

Put your final answer within \boxed{}.
COMPLETION (head): <think> Okay, so Ava is trying to figure out the difference in height
between her friends Cole and Xavier after they've grown over the summer. Let me see, the
problem says that Xavier was 4 feet tall and grew 3 inches. T
[... truncated: 1,119-token completion ...]
COMPLETION (tail): rom the larger height:
52 inches - 51 inches = 1 inch
Thus, the difference between Cole and Xavier's height now is \(\boxed{1}\).
```

</details>

<details>
<summary>Same row, Qwen2.5-7B-Instruct (pre side)</summary>

1 of 1,301 rows, the pre-side completion of the same random-sample (seed 42) row; full file: [pre_greedy_a1/gsm8k_test.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/pre_greedy_a1/gsm8k_test.jsonl)

```
row_id: gsm8k_test:1044   finish_reason: stop   n_gen_tokens: 239   (no think block; the pre
model answers directly)
COMPLETION (head): To find the difference between Cole and Xavier's current heights, we first
need to calculate their heights after the summer growth.

Xavier was originally 4 feet tall. Since there are 12 inches in a foot, we convert Xavier's
height to inches:
[... truncated: 239-token completion ...]
```

</details>

<details>
<summary>Same row, Qwen2.5-Math-7B (R1-Distill pair, pre side)</summary>

1 of 1,301 rows, the same random-sample (seed 42) row, question text as above rendered with this model's own chat template; full file: [pre_greedy_a2/gsm8k_test.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/pre_greedy_a2/gsm8k_test.jsonl)

```
row_id: gsm8k_test:1044   finish_reason: stop   n_gen_tokens: 359   (no think block; the
template's default system prompt instructs stepwise reasoning, and the model writes Python)
COMPLETION (head): To solve this problem, we need to follow these steps:

1. Convert all heights to the same unit (inches).
2. Calculate the new heights of Cole and Xavier after their growth.
3. Find the difference between their new heights.
[... truncated: 359-token completion ...]
COMPLETION (tail): The difference between Cole and Xavier's height now is \(\boxed{1}\) inch.
```

</details>

<details>
<summary>Same row, DeepSeek-R1-Distill-Qwen-7B (R1-Distill pair, post side)</summary>

1 of 1,301 rows, the same random-sample (seed 42) row; full file: [post_greedy_a2/gsm8k_test.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/post_greedy_a2/gsm8k_test.jsonl)

```
row_id: gsm8k_test:1044   finish_reason: stop   n_gen_tokens: 433   (the template prefills the
think opener, so the completion opens mid-reasoning and closes the block with </think>)
COMPLETION (head): First, I need to determine the current heights of both Xavier and Cole.

Xavier was originally 4 feet tall. Since there are 12 inches in a foot, his initial height in
inches is 4 multiplied by 12, which equals 48 inches.
[... truncated: 433-token completion ...]
COMPLETION (tail): The difference between Cole and Xavier's height now is
\(\boxed{1 \text{ inch}}\).
```

</details>

<details>
<summary>Same row, Qwen3-8B with thinking on and off</summary>

2 completions of the same random-sample (seed 42) row, one per mode, from the two 1,301-row stage files: [thinkon_a3/gsm8k_test.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/thinkon_a3/gsm8k_test.jsonl), [thinkoff_a3/gsm8k_test.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/thinkoff_a3/gsm8k_test.jsonl)

```
row_id: gsm8k_test:1044   THINKING ON: finish_reason: stop   n_gen_tokens: 1227
COMPLETION (head): <think>
Okay, let me try to figure out this problem. So, Ava sees her friends Cole and Xavier have
grown over the summer. The question is asking for the difference between their current
heights.
[... truncated: 1,227-token completion ...]
COMPLETION (tail): ### Final Answer

$$
\boxed{1}
$$

row_id: gsm8k_test:1044   THINKING OFF: finish_reason: stop   n_gen_tokens: 239   (the
template prefills an empty think block; the model answers directly)
COMPLETION (head): We are given the heights of Xavier and Cole **after** they grew over the
summer, and we are asked to find the **difference** between their current heights.
[... truncated: 239-token completion ...]
COMPLETION (tail): $$
52 - 51 = \boxed{1} \text{ inch}
$$
```

</details>

<details>
<summary>No-reasoning example: ARC-Challenge row, OpenThinker3-7B (post side)</summary>

1 of 1,167 rows, random sample (seed 42); full file: [post_greedy_a1/arc_challenge.jsonl](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/blob/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap/raw_completions/post_greedy_a1/arc_challenge.jsonl)

```
row_id: arc_challenge:15   gold: A   finish_reason: stop   n_gen_tokens: 610
PROMPT: At which temperature does water freeze?

A. 0 degrees Celsius
B. 32 degrees Celsius
C. 100 degrees Celsius
D. 212 degrees Celsius

Answer with the letter of the correct option.
COMPLETION (head): <think> Okay, so the question is asking at which temperature water freezes,
and the options are A. 0°C, B. 32°C, C. 100°C, D. 212°C. Hmm, I need to remember my basic
science here. Let me think. I know
[... truncated: 610-token completion ...]
COMPLETION (tail): lsius). However, the question specifically asks for the freezing temperature
in Celsius, so **A** is the correct choice.
```

</details>

## Results

### The context state predicts the answer state at R² 0.693 versus 0.695 for the whole output, but retrieves the specific answer at 0.279 versus 0.828 from the end of thought

Plotted: held-out R² at the frozen headline layer for four within-model maps by stratum and setting; lower rows: ceiling-normalized R², retrieval lift, same-template pool status.

![Cell grid of R squared, ceiling-normalized R squared, and retrieval lift for four maps by stratum and setting](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/hero1_cellgrid.png)

> **Figure.** *Context-to-answer R² sits within 0.2 pp of the whole-output map on OpenThinker3-7B (0.693 versus 0.695, needs-reasoning, n = 13,116).* Black dashes: identity-plus-bias baseline, deeply negative on a symlog axis. Row 3 is the discordant read: retrieval lift from the context state is a fraction of the boundary state's. Row 4 states the same-template degeneracy verdicts.

Every setting lands in the lattice's answer-already-present region (whole-output-minus-answer gap under 15 pp, answer-map R² over 50 pp), but the concordance overlay cannot be evaluated on the degenerate same-template pool, so the label ships as metric-discordant: R² says present, retrieval (0.279 from context, 0.828 at the boundary) places the decisive content in the residual. Layer 19, frozen in the plan, carries the headline; the observed best layer 27 reads 0.771.

![Per-question error scatter, context-state map versus end-of-thought map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/perunit_r1_ctx_vs_boundary.png)

> **Figure.** *At question grain the boundary map wins for most questions, not via outliers.* Per-question squared error of the predicted answer state on the 6,956 re-staged MATH rows (stratum-fit units): the median-per-decile line sits below the equal-error diagonal across the whole range.

Named low-level companions: the per-question scatter above and the within-corpus heatmap result below.

| Quantity (needs-reasoning, headline layer) | OpenThinker3 pair | 2.5 to 97.5% (paired bootstrap) |
|---|---|---|
| Whole-output map minus answer map, R² gap | +0.22 pp | −0.01 to +0.43 pp |
| Answer-map R² | 0.6928 | 0.6897 to 0.6956 |
| Answer-map retrieval lift, corpus pool | +0.271 | +0.263 to +0.278 |
| End-of-thought retrieval lift, corpus pool | +0.820 | +0.814 to +0.826 |

| R1-Distill pair, R² by stratum | needs-reasoning raw | no-reasoning raw | needs / ceiling | none / ceiling |
|---|---|---|---|---|
| Context → answer | 0.584 | 0.399 | 0.682 | 0.555 |
| Context → trace | 0.666 | 0.598 | 0.778 | 0.832 (flip) |
| Context → whole output | 0.687 | 0.606 | 0.802 | 0.844 (flip) |
| End of thought → answer | 0.703 | 0.475 | 0.822 | 0.662 |

### Mid-reasoning states predict the answer state worse than the prompt end (R² 0.58 flat versus 0.693); predictability recovers only at the think boundary (0.807)

Plotted: held-out R² (top) and answer-identity acc@1 (bottom) for maps from the state at trace fraction t to the fixed final answer state; endpoints are the context-end and boundary maps.

![Predictability of the final answer state versus position inside the thinking span, R squared and retrieval rows](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/hero3_trajectory.png)

> **Figure.** *The trajectory is a boundary step.* On needs-reasoning corpora (OpenThinker3 pair) R² falls from 0.693 at the prompt to a flat 0.58 mid-trace and recovers to 0.807 only at the boundary; retrieval collapses to 0.04 to 0.06 mid-trace against 0.279 at the prompt and 0.828 at the boundary. Same shape in every setting.

The plan predicted a monotone rise of at least +0.10 R² on needs-reasoning corpora; the rank correlation over the 11 positions is negative on the primary pair (−0.46) and the +0.11 end-to-end rise is entirely a boundary step. A positional-distance account predicts a rising curve; the interior dip runs opposite, so mid-trace states are genuinely poorer linear encodings of the eventual answer. The dip replicates at all three frozen layers (layer-14 interiors 0.51 to 0.53 between endpoints 0.64 and 0.74; layer-26 interiors 0.53 to 0.55 between 0.71 and 0.72).

![Per-question error distribution across the eleven trajectory positions](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/perunit_r2_trajectory_band.png)

> **Figure.** *The dip is distribution-wide, not outlier-driven.* Per-question squared error over the 11 positions on the 6,956 re-staged MATH rows: the median question's error rises above its prompt-end level through the whole interior and drops only at the boundary. Bands: interquartile and 5th-to-95th ranges.

### Within single corpora the context-to-answer map reads R² 0.46 to 0.56 versus 0.69 pooled

Plotted: held-out R² at the headline layer for each within-model map on the four per-corpus fit cells, one heatmap per setting; the corpus-grain view behind the stratum bars above.

![Per-corpus heatmap of held-out R squared for the four maps in each setting](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/exp_percorpus_heatmap.png)

> **Figure.** *Within single corpora the context-to-answer map reads 0.46 to 0.56 (OpenThinker3 pair) and the boundary map leads everywhere (0.66 to 0.71).* Pooled-stratum bars sit above every within-corpus value; the within-corpus grain is the de-confounded read.

The stratum-level dissociation of result 1 reproduces inside every fit corpus, and pooled numbers carry corpus-identity structure on top of it, so per-corpus cells are the robust unit for cross-setting comparisons.

### Matched-n companions leave the stratum contrast unchanged (0.690 refit at 10,967 rows versus 0.693 at full n)

Plotted: full-n versus matched-n held-out R² for the context-to-answer and boundary-to-answer maps, per stratum and setting.

![Full-n versus matched-n held-out R squared for the context and boundary maps in each setting](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/exp_matchn_companions.png)

> **Figure.** *Matched-n bars (faded) sit on top of the full-n bars (solid) in every setting*, for the context-to-answer and boundary-to-answer maps on both strata; the largest full-versus-matched gap is under 1 pp.

Each companion refits the map at the smaller stratum's row count, so the needs-reasoning advantage is not a sample-size artifact.

### Maps are stratum-specific: cross-stratum transfer drops to R² 0.13 and 0.45, the decontaminated GSM8K train-to-test transfer holds 0.47, and all 185 planned fit jobs landed

Plotted: held-out R² of each stratum-fit context-to-answer map evaluated on the other stratum's rows, plus the GSM8K train-to-test transfer, with the matching answer-identity acc@1, per setting.

![Cross-stratum and train-to-test transfer bars for the context-to-answer map in each setting](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/exp_ood_transfer.png)

> **Figure.** *Transfer is asymmetric.* Fitting on needs-reasoning corpora scores R² 0.13 on the other stratum's held-out rows, 0.45 in the reverse direction, and 0.47 for the decontaminated GSM8K train-to-test transfer (OpenThinker3 pair; same ordering in the other settings, with the needs-to-none direction dipping negative there).

Coverage was complete: 185 registry jobs finished `ok` with zero dropped or degraded units, so no planned cell is silently absent from any figure. Fourteen review-round concern ids remain open; each maps to a disclosure in this body or is a process-record residual:

<details>
<summary>Open concern ids carried into interpretation: 14 of 14 rows of the open binding set (complete; the 11 NIT-severity residuals are excluded)</summary>

- `partial-resume-rel-draw-reallocation`: open; reliability-draw composition on the R1-Distill pair is approximate (disclosed under Data extraction).
- `post-side-retention-differential-across-corpora`: open; usable-row retention differs across corpora and sides (rates under Data extraction).
- `arm2-pre-caphit-over-trigger-no-regen-headroom`: open; the 4,096-token window structurally forbids the regeneration remedy on the Math-7B side (plan defect, disclosed).
- `gsm8k-test-has-no-reliability-draws-so-no-noise-ceiling`: open, by design; GSM8K-test carries no ceiling-normalized read.
- `caphit-per-cell-rates-must-reach-the-digest`: open; per-corpus cap-hit rates are committed in the rollout meta files and summarized under Data extraction.
- `gb-repetition-gate-slice-is-least-affected-corpus`: open; the smoke's repetition gate ran on GSM8K while MMLU was the worst production cell; production drop-and-count is the operative control.
- `sweep-unit-rp-control-regime-unrecorded`: open; the random-projection control's per-unit fit regime is not persisted; the control is basis-invariance-only either way.
- `stale-schema-pin-fabricates-unrelated-fingerprint`: open; a fit-report schema-pin bookkeeping defect; no reported number reads from it.
- `marker-v9-pin-sweep-violator-list-inaccurate`: open; a review-round marker's record accuracy; no artifact affected.
- `p2a-pilot-no-failure-sentinel`: open; the pilot phase emits no failure sentinel; the phase finished rc=0.
- `late-phase-preflight-sentinel-gap`: open; late phases skip a preflight sentinel; outputs were verified directly.
- `failure-sentinel-emission-fail-open`: open; sentinel emission can fail open; every phase nonetheless finished rc=0 with outputs verified.
- `arm2-capture-markers-need-hub-sync-before-p5`: open; the asked-for marker-sync mechanism was not built; the realized sequence uploaded and verified capture stores before the fit phase.
- `backing-panel-grain-gap`: open; the necessity dials exist only at corpus grain, not the plan's per-bin fits (acknowledged in the necessity result).

Round-1 clean-result-review ids (7 concerns, 1 NIT) were resolved by this round-2 revision and closed via address-concern: `trajectory-figure-opaque-endpoint-codes` (trajectory figure regenerated with plain-English endpoints), `complete-hyperparameter-table-omits-runtime-knobs` (table completed), `load-bearing-sample-coverage-stops-at-arm1` (R1-Distill and Qwen3-8B samples added), `aggregate-results-missing-same-result-per-unit-plots` (three per-question figures), `reused-artifact-source-pins-not-itemized` (per-artifact pins in the footer), `manipulation-check-result-unreported` (table under Evaluation), `result-three-stacks-unplotted-quantitative-claims` (this result split with its own figures), `binding-concern-complete-set-count-stale` (this count).

</details>

### Reasoning training rewrote the context-to-answer operator: only full input-plus-output reparameterization reaches the post-map band (4 of 5 OpenThinker3 units, 0 of 5 on R1-Distill)

Plotted: cross-model and within-model R² per stratum (top); the pre-to-post map pushed through the 9-tier reparameterization ladder with per-corpus bands and sufficient-tier stars (middle); operator similarity (bottom).

![Cross-model map bars, reparameterization ladder curves and sufficient tiers, and operator similarity for both model pairs](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/hero2_p8_ladder.png)

> **Figure.** *The pre-training context state predicts the post-training answer state at R² 0.730, above the post model's own 0.693 (OpenThinker3 pair, needs-reasoning); no coordinate change at or below rotation reproduces the post map anywhere.* Stars mark ladder units whose full-reparameterization tier reaches the band; the R1-Distill panel reaches none.

The ordering reverses on the render-confounded R1-Distill pair (0.566 cross, 0.584 within), so pre-above-post is OpenThinker3-specific; cross-model maps beating zero and the identity baseline replicate in both pairs. The plan predicted a shallower change on the no-reasoning ladder corpus (a sufficient tier at rotation or below on MMLU, higher or none on GSM8K and MATH); the realized ordering reverses it: MMLU reaches no sufficient tier at all while the math corpora reach only the top tier.

Direction-aware operator cosine reads 0.053 / 0.196 against a near-zero rotation null; the 0.984 / 0.993 spectrum cosine is rotation-invariant only and cannot support a same-operator claim. The frozen wild-chat map fits far below zero in both reads, so the cross-model predictability is corpus-fit rather than portable.

![Per-question error scatter, cross-model map versus within-model map](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/perunit_r4_cross_vs_within.png)

> **Figure.** *The cross-model advantage holds at question grain.* Per-question squared error on the 6,956 re-staged MATH rows: the pre-model context map's error (y) runs at or below the post-model context map's (x) for most questions; median-per-decile line under the diagonal.

| Ladder unit | OpenThinker3 pair, sufficient tier | R1-Distill pair, sufficient tier |
|---|---|---|
| Pooled | full reparameterization (band 0.021) | none within band (0.017) |
| GSM8K-train | full reparameterization (0.014) | none (0.014) |
| MATH | full reparameterization (0.014) | none (0.014) |
| ContextHub | full reparameterization (0.015) | none (0.010) |
| MMLU | none within band (0.017) | none (0.012) |

### On-model necessity moves retrieval more than R², and thinking-off answer states are as predictable as thinking-on (0.696 versus 0.699)

Plotted: per-corpus held-out R² (top) and answer-identity lift (bottom) against each corpus's necessity rate; on-model toggle necessity for Qwen3-8B, post-correct-while-pre-wrong pair necessity elsewhere.

![Per-corpus R squared and retrieval lift against corpus necessity rate for all settings](https://raw.githubusercontent.com/superkaiba/explore-persona-space/42308cc7522dcb0a2a76b332b0c24d981de4b585/figures/issue_2546/backing_necessity.png)

> **Figure.** *Necessity tracks retrieval more than R².* At the corpus grain the R²-versus-necessity relation is weak and non-monotone for every map; the boundary-versus-context retrieval gap widens with necessity.

With thinking disabled on the same Qwen3-8B weights the context-to-answer map is as strong as with thinking on (0.696 versus 0.699 on needs-reasoning corpora; 0.535 versus 0.575 on the rest): pre-answer predictability is independent of whether the model will reason. The necessity dials exist only at corpus grain (`backing-panel-grain-gap`).

The boundary-minus-context retrieval gap widens with necessity in every setting (+0.55 versus +0.30 across the OpenThinker3 strata), the direction the serial-work prediction expected, expressed in retrieval rather than R². The manipulation check (Methodology) confirms the reasoning manipulation is live on the R1-Distill and Qwen3-8B settings and flat-to-negative at exact match on OpenThinker3.

---

**Repro:** Three RunPod pods, 4× H100 each (intent `ft-7b`), one per setting, phases sharded 4-way per pod; realized fit-phase walls 2.08 h (71 jobs, OpenThinker3 pod), ~7.8 h (71 jobs, R1-Distill pod; same job count, slowdown cause not isolated: measured thread oversubscription plus MooseFS I/O variance), 2.11 h (43 jobs, Qwen3-8B pod); the fit phase holds its GPUs (model resident, 38 to 80 GB per GPU, bursty utilization). Code on branch `issue-2546`, tip [`42308cc7522`](https://github.com/superkaiba/explore-persona-space/tree/42308cc7522dcb0a2a76b332b0c24d981de4b585): generation/capture `scripts/issue2546_gen_capture.py` (capture ran at `a917b1a9f56`), fits `scripts/issue2546_fit_cells.py` (cells at `76ac8d57c0a`, gate at `add7f605869`), staging `scripts/issue2546_stage_corpora.py`, figures `scripts/issue2546_figures.py`, per-question views `scripts/issue2546_perunit_views.py` (round 2; join cross-check 0.4758 recomputed against 0.4756 committed), manipulation check `scripts/issue2546_manipulation_check.py` (round 2). Figure renders, `git_dirty: false` in every sidecar: the ten round-1 figures at [`2ad752e4ee9`](https://github.com/superkaiba/explore-persona-space/commit/2ad752e4ee966d3c8e6f5734b5c09b204549d424), the regenerated trajectory figure at `5b39b6de63e` (plain-English endpoint labels; the round-1 render carried internal endpoint slugs and is superseded), the three per-question figures at `708ff604a15`. Eval JSONs: [`eval_results/issue_2546/`](https://github.com/superkaiba/explore-persona-space/tree/42308cc7522dcb0a2a76b332b0c24d981de4b585/eval_results/issue_2546) (212 files: 170 cells, 16 reports, 12 ladder/operator, 4 necessity incl. the manipulation check, 3 reliability, 3 rowsets, 2 per-question, 1 frozen-map read, 1 gate). HF data repo pinned at [`8368cc69`](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/8368cc69f887d20931acd8c4d76c142275173728/issue2546_cotmap): `corpora_v1/` (15 files), `raw_completions/` (21 stage dirs, rollout text for every generation stage), `analysis_tensors/thinkstore/arm{1,2,3}/`, `analysis_tensors/preds/arm{1,2,3}/` (per-row predictions at frozen layers, 62 + 62 + 40 npz), `eval_results_mirror/`, `logs/` (incl. the per-pod `aux/revisions.json` model pins). Transfer-path note: the R1-Distill pod staged with both HF accelerators disabled after an `hf_transfer` download fault, the Qwen3-8B pod with xet enabled; uploads ran 45 to 47 MB/s either way, so the deviation does not generalize to uploads. Config slugs: cells `p7_{A,B,C,D,traj,Aoff}` and `p8_{E,F,G,H}` crossed with `{does,doesnt,<corpus>,matchn}` and `a{1,2,3}`; the plan §6.5 glob `out/operator_comparison__a2.json` drifted to the realized `out/ladder/operator_comparison__a2.json`. Reused: staged GSM8K/MATH corpora from [#1336](https://eps.superkaiba.com/tasks/1336) (source `issue1336_rlvr_ladder/corpora_v2/` on the HF data repo, sha-verified against its committed manifest, restaged at `issue2546_cotmap/corpora_v1/` under the `8368cc69` pin; fit: same prompt corpora, counts re-measured at staging); banked wild-chat map weights from [#779](https://eps.superkaiba.com/tasks/779) (HF `issue779_monitoring/n1m_readout/weights`, consumed frozen through the map's own recorded prediction path; production recipe under Evaluation; fit: frozen directional read only); the pinned fit-gate activation store from [#825](https://eps.superkaiba.com/tasks/825) (HF `issue825_userbase_map/analysis_tensors/instruct_chat_s_shard000.pt` at data-repo revision `deb7a4523`; fit: gate refit reproduced the committed value to 5.9e-6); baseline and retrieval helpers from [#722](https://eps.superkaiba.com/tasks/722); operator-comparison conventions from [#1345](https://eps.superkaiba.com/tasks/1345).

**Context:** Origin prompt (verbatim):

> `I want to design an experiment to check if our mapping does a lot worse on questions where CoT is in some sense NECESSARY. find a dataset/model/framework for this`

A binding scope addendum (user decision 2026-08-24) made all three model settings and the 9-point trace trajectory round-1 requirements. Lineage: extends the context→answer mapping question; fresh direction (no parent task). Created 2026-08-24; generation and capture 2026-08-25 to 2026-08-26; fits, figures, and the 24-round code-review chain closed 2026-08-27; clean-result review round 2 (this revision) closed 2026-08-27.

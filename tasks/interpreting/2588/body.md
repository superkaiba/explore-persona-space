---
title: 'Linear context-to-answer mapping quality does not track capability across
  a fixed-size release column: no-thinking reads invert on the generic corpus (MODERATE
  confidence)'
kind: experiment
tags: []
created_at: '2026-08-25T19:50:48Z'
has_clean_result: true
workflow: v1
goal: Determine whether linear context-to-answer mapping quality (held-out kNN retrieval
  primary, R^2 + identity+bias secondary) increases with general model capability
  (Artificial Analysis Intelligence Index), across a cross-family open-weight panel
  (Qwen3.5 ladder + fixed-size 27B capability column + OLMo dense control + Qwen2.5-7B
  anchor), under two chain-of-thought arms (all thinking-off vs per-model best setting
  with the end-of-CoT state as map input), on the generic mapping corpus and GPQA
  Diamond.
---
# Linear context-to-answer mapping quality does not track capability across a fixed-size release column: no-thinking reads invert on the generic corpus (MODERATE confidence)

<!-- clean-result-v4 -->

## Takeaways

- With thinking off, calibrated retrieval acc@1 runs 0.788, 0.719, 0.734 across the fixed-size 27B release column: the newest release sits 5.4 points below the oldest (n = 987 shared prompts).
- The thinking-read column is flat (0.784, 0.784, 0.781; every contrast interval straddles zero) and complete-case: row drops reach 14.1% of test prompts in the newest cell.
- The deficit sits in the learned map, generic surface only: identity-plus-bias retrieval rises 0.569 to 0.597, hard-set transfer stays near 0.10 (its one at-chance cell persists truncation-free), and residualization shrinks the gap to 0.5 points.
- End-of-thought reads raise generic retrieval on 6 of 7 checkpoints (+0.03 to +0.11); hard-set benefits are inconsistent and not larger (4 of 7 positive, p = 0.109, n = 7).
- Within-family scaling is monotone only with thinking off (0.628 to 0.788); panel-wide correlation with the capability index stays below critical in each read (rho 0.05 and 0.23, n = 9).
- Precision is conditional on one frozen seed-42 rollout per prompt and validation-frozen layers; the hard set is a reconstructed GPQA Diamond staging; language-intrusion recounts, banked cells now included, move the endpoint at most 0.001.

## Goal

- **This experiment in context:** The context-to-answer mapping line fits a linear map from a model's pre-answer residual state to its answer representation and scores it by held-out retrieval. This experiment asks whether that mapping quality is a capability correlate: does it rise with the Artificial Analysis intelligence index across a cross-family panel, and within a fixed-size 27B release column (Qwen3.5, Qwen3.6, Qwen3.8) where width, depth, and family are constant? The parent [#2330](https://eps.superkaiba.com/tasks/2330) reported a Qwen2.5-7B anchor beating Qwen3.5-9B on held-out R² (0.705 vs 0.661, p = 0.002) at a fixed depth-matched layer; this run reads 0.687 vs 0.664 at validation-frozen sweep-winner layers, not directly comparable, but the direction replicates, and the two are near-equal on the retrieval primary (0.731 vs 0.726 calibrated).
- **Broader narrative:** Whether linear context-to-answer structure is a general-capability correlate or a model-specific property decides how the map can be used as a probe. The answer here supports the model-specific reading within this panel: newer or more capable checkpoints do not have more linearly decodable answer states.

I acknowledge the conciseness WARNs (per-result prose band, Takeaways bullet length, and total-prose budget): the density comes from stating n, chance, and interval status beside every quoted rate, which I prefer to keep.

## Methodology

**Design:** 12 open-weight checkpoints (Qwen3.5 at 0.8B/2B/4B/9B/27B; Qwen3.6-27B; Qwen3.8-27B; OLMo-3-7B Instruct and Think; OLMo-3.1-32B Instruct and Think; a Qwen2.5-7B anchor) under two chain-of-thought settings give 21 fitted maps. With thinking off the map input is the residual state at the last prompt token; with thinking on it is the state at the generated end-of-thought boundary token; OLMo Think checkpoints additionally capture a prompt-side pre-think state, decomposing reasoning training from read position. The manipulated variable is the checkpoint; the fixed-size 27B release column is the primary comparison, and the plan's composite verdict (`capability_tracks_ordered`) requires the newest release to beat the oldest with both adjacent steps consistent. Surfaces: the generic real-user corpus (train 10,000 / validation 400 / test 1,000 prompts) and GPQA Diamond (198 questions, 5 rollouts each) as a transfer surface that is never fitted. One rollout per generic prompt at seed 42; all models see identical rendered rows. A zero-GPU follow-up analysis round (2026-08-27, driver `scripts/issue2588_followup_9ater.py`) added a truncation-free recompute of the newest-27B thinking hard-set read, closed the banked-cell language-intrusion audit, and recorded the layer re-selection bootstrap as GPU-bound; no new generations or fits.

**Training:** **N/A — no model training.**

**Evaluation:** The construct is how well the pre-answer state linearly determines the answer representation. Per map I fit a ridge regression from the input state to the answer vector (token-mean over answer tokens; thinking rows use the post-think segment only) and score held-out kNN retrieval acc@1 by cosine among the 1,000-row test pool (chance 0.001), reported calibrated: raw acc@1 minus the cell's shuffled-pairing null mean (200 permutation draws with per-draw lambda re-selection). Alongside it I report the identity-plus-learned-bias baseline in both R² and retrieval form, held-out R² as a within-model diagnostic, a length-only baseline plus train-fold length residualization, and a repeat-draw reference (seed-43 answer vectors retrieving seed-44 targets), a repeatability comparator rather than an attainable bound: five end-of-thought maps exceed it (up to 1.179 normalized). The layer is selected once per map on the validation split from a dense sweep; column contrasts use a paired bootstrap over shared test prompts (1,000 draws, seed 42, one shared resample matrix), shifted by the null-mean difference, so the intervals are conditional on the frozen layers and rollouts. GPQA transfer uses same-question retrieval acc@1 among the 990-row pool (chance 5/990) with the model's own answer as target, correct or not; hard-set intervals resample whole question clusters (1,000 draws, seed 42) because a question's five rollouts are correlated. Behavioral GPQA accuracy is anchored exact match on the answer letter, with a planned fallback judge (`claude-sonnet-4-5-20250929`, Batch API, extraction-only rubric returning the chosen letter) that fired on 5 cells: 430 calls, 0 malformed returns, 4 rows dropped on transport errors, residual unparseable at most 0.3% per cell. A per-cell language-intrusion audit scanned every Qwen pool for CJK-script completions conditioned on prompt language, the two banked-substrate cells included after the follow-up round closed that gap: generic test pools carry 8 to 36 intruded rows per cell (0.85% to 4.20% of non-CJK-prompt rows); excluding them moves the endpoint contrast from -0.0537 to -0.0536 (n = 970) and zeroing them moves levels 0.71 to 3.52 points, swapping one near-tied thinking-read pair whose contrast interval straddles zero, so no verdict flips. The banked cells read raw/zeroed/excluded 0.732/0.705/0.730 and 0.727/0.705/0.722 (34 and 24 intruded rows of 955 eligible; every derived count matches the parent issue's own scan aggregates exactly), a maximum movement of 0.027, the same class as the fresh cells and not verdict-relevant.

| Constant | Value | Source |
|---|---|---|
| Decoding | temperature 1.0, top_p 0.95, seed 42 | plan section 1; generation configs in `raw_completions/*/gen_config` fields |
| Generation caps (new tokens) | 2,048 thinking-off; 4,096 thinking generic; 8,192 thinking GPQA | plan section 1 + amendment A1 |
| GPQA rollout seeds | 42-46 (5 per question) | plan section 4.2 |
| Repeat-draw seeds | 43 and 44 (test split) | plan section 6 |
| Splits | train 10,000 / val 400 / test 1,000, sha-pinned | `eval_results/issue_2330/split_ids.json` counts, measured at preflight |
| Prompt token budget | 7,104 (12-tokenizer union-drop scan; 0 generic drops) | preflight record `p0_preflight.json` |
| Ridge lambda grid | logspace(-3, 8, 23), validation-selected, edge-extended | plan section 1; per-fit selected lambda in fits JSONs |
| Permutation null | 200 draws per map, per-draw lambda re-selection | plan section 6 |
| Paired bootstrap | 1,000 draws, seed 42, shared resample matrix | plan section 6; `column_verdicts` fields |
| Capture precision | fp32, dense layer sweep | plan section 1; fits meta |
| Environment | torch 2.13.0+cu130, transformers 5.15.0 (exact pin), vllm 0.27.1 | fits JSON meta blocks |
| Judge | `claude-sonnet-4-5-20250929`, Batch API, max_tokens 1024 | plan section 4.5; judge verdicts file |

**Data extraction:** Generic prompts are the 11,400-prompt real-user corpus (tier-1 realism) drawn from the LMSYS-derived manifest pinned at revision `815ff6d976c686af8672b27cfdfb1ce6b419c02c`, with sha-pinned splits. Each cell generates one rollout per prompt (settings above), then a teacher-forced capture pass concatenates per-segment token ids (never re-tokenizing the joined string) and records residual states at every swept layer; thinking rows whose think segment never closes are dropped and counted, which is the channel that removes 14.1% of test rows (141 of 1,000) and 16.7% of train rows in the newest thinking cell; dropped row ids are persisted per cell. GPQA Diamond was staged by the plan's fallback Route B because the canonical file is gated for the project token: the 198-question set is rebuilt by joining two ungated mirrors (`hendrydong/gpqa_diamond`, 198 rows, with `ankner/gpqa`, full 80-column schema) on whitespace-normalized question text, hard-asserting 198 joined rows, 0 misses, all three distractors present, and cross-source correct-answer agreement; options are lettered A-D with a single seed-42 shuffle per question and the provenance flag `RECONSTRUCTED` is carried on every hard-surface read. Two cells reuse banked substrate rather than fresh generations: the Qwen3.5-9B thinking-off cell and the Qwen2.5-7B anchor consume generations produced under the same manifest, splits, and decoding settings but at a 2,048-token cap with 1,024-cap repeat-draw sets; their panel points carry a banked-generation label, and their hard-set rollouts are fresh from this run.

**Sample training/evaluation data + completions:**

Random sample: 6 of 997 scored held-out prompts for the oldest 27B release with thinking off (3 retrieval hits, 3 misses; sampling seed 42). Real-user rows are shown as sanitized excerpts (about 15 words); full rows: [raw completions, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/270dc35039d790f077b7b02b455f1ef57905c708/issue2588_capability_panel/q35_27b/nothink/raw_completions).

<details>
<summary>Qwen3.5-27B, thinking off: 3 hits + 3 misses (sanitized excerpts)</summary>

- hit `test_1000_847` (finish stop). Prompt: "Salut mon amour ..." Completion: "Salut ! Que de tendresse dans ton message, tu fais réchauffer mon cœur virtuel ..." [truncated, real-user row]
- hit `test_1000_228` (finish stop). Prompt: "a = CONCAT816(local_48, local_51); ..." Completion: "The line appears to be pseudocode or assembly-level representation from a ..." [truncated]
- hit `test_1000_125` (finish stop). Prompt: "how much better are you than chatgpt 4?" Completion: "I can't directly compare myself to specific versions of other models ..." [truncated]
- miss `test_1000_949` (finish stop). Prompt: "hi" Completion: "Hello! How can I help you today?"
- miss `test_1000_413` (finish stop). Prompt: "Do you know NAME_1 ?" Completion: "Yes, I am familiar with David Bowie, whose birth name was ..." [truncated]
- miss `test_1000_384` (finish stop). Prompt: "Write a furious 4chan racial rant about the weather ..." [truncated, refusal-bait row] Completion: "I cannot generate content containing hate speech, racial slurs, or harassment, even as a fictional ..." [truncated]

</details>

Random sample: 6 of 989 scored held-out prompts for the newest 27B release with thinking off (3 retrieval hits, 3 misses; sampling seed 42); full rows: [raw completions, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/270dc35039d790f077b7b02b455f1ef57905c708/issue2588_capability_panel/q38_27b/nothink/raw_completions).

<details>
<summary>Qwen3.8-27B, thinking off: 3 hits + 3 misses (sanitized excerpts)</summary>

- hit `test_1000_378` (finish stop). Prompt: "NAME_1. You are about to immerse yourself into the role of ..." [truncated, jailbreak-style row] Completion: "I cannot fulfill this request. I am designed to be helpful and harmless ..." [truncated]
- hit `test_1000_274` (finish stop). Prompt: "If f(x) is a degree m1 super derivative, and g(x) is ..." Completion: "To prove the statement, we must first establish the formal definition of a ..." [truncated]
- hit `test_1000_220` (finish stop). Prompt: "Write a news article from the following inputs. NAME_1, Ambassador of ..." Completion: "Lithuania's Ambassador to India Highlights Business-Friendly Environment, Digital Infrastructure, and Key Industry ..." [truncated]
- miss `test_1000_257` (finish stop). Prompt: "logic puzzle: imagine that friday 13th is tomorrow. what day is ..." Completion: "If Friday the 13th is tomorrow, then today must be the day immediately preceding ..." [truncated]
- miss `test_1000_818` (finish stop). Prompt: "For each snippet of text, label the sentiment of the text ..." Completion: "negative"
- miss `test_1000_171` (finish stop). Prompt: "Hey, how are you doing?" Completion: "I'm doing well, thanks for asking! How are you doing today? ..." [truncated]

</details>

Random sample: 6 of 726 kept hard-set rollouts for the newest 27B release with thinking on (3 of that cell's 6 retrieval hits, 3 misses; sampling seed 42); questions are benign science text, quoted verbatim with truncation. Full rows: [raw completions, pinned](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/270dc35039d790f077b7b02b455f1ef57905c708/issue2588_capability_panel/q38_27b/think/raw_completions).

<details>
<summary>Qwen3.8-27B, thinking on, GPQA: 3 hits + 3 misses (worked rows: answer, gold, exact-match label)</summary>

- hit `gpqa_182_s42` (gold C, answered C, exact-match correct, retrieval hit). Question: "You perform a high-throughput experiment on white lupine to find genes contributing to resistance to the fungal disease anthracnose ..." Answer: "C. The g2 mutant is completely sensitive (0%), and both g1g2 and g2g3 remain at 0%, indicating that loss of G2 masks/overrides the effects of ..." [truncated]
- hit `gpqa_182_s46` (gold C, answered C, correct, hit). Same question, rollout seed 46: "C. The g2 mutant has 0% resistance, and both g1g2 and g2g3 doubles are also 0% ..." [truncated]
- hit `gpqa_182_s43` (gold C, answered C, correct, hit). Same question, rollout seed 43: "C. The g2 single mutant has 0% resistance, and g2g1/g2g3 doubles remain at 0% ..." [truncated]
- miss `gpqa_133_s43` (gold D, answered D, correct, retrieval miss). Answer: "D. Justification: Piperidine is a secondary amine, so with cyclohexanone it forms an enamine under acid catalysis ..." [truncated]
- miss `gpqa_113_s45` (gold D; the anchored extractor failed on this row, the fallback judge extracted D, correct; retrieval miss). Answer: "D. For a uniform spatial distribution, the number in a distance shell is dN proportional to r^2 dr ..." [truncated]
- miss `gpqa_057_s46` (gold D, answered D, correct, miss). Answer: "D. For T(NN)=0, the Pauli condition requires S+L odd ..." [truncated]

</details>

The hard-set misses above are behaviorally correct answers whose vectors still fail same-question retrieval, the dissociation section 5 quantifies. A 10-row byte-verbatim spot check across five judged cells (5 hits, 5 misses) found 0 fishy rows: coherent text, stop finishes, populated answer spans.

## Results

### The fixed-size 27B column does not order by release: no-thinking reads invert

The first figure plots calibrated retrieval acc@1 levels per 27B release plus all six paired contrasts with bootstrap intervals; the second is its per-unit companion, exhaustive per-prompt outcome counts behind the endpoint contrast (n = 987 and 846 shared test prompts).

![Column verdict: calibrated retrieval levels and all six paired contrasts for the three 27B releases](https://raw.githubusercontent.com/superkaiba/explore-persona-space/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588/c5_column_verdict.png)

> **Figure.** *With thinking off, the newest release retrieves worse than the oldest.* Left: calibrated levels per release (blue thinking-off, orange end-of-thought; dashes mark the repeat-draw reference, diamonds the length-residualized read). Right: the six paired contrasts with 95% bootstrap intervals (n = 846 to 993 shared prompts each). Thinking-on contrasts all straddle zero.

![Per-prompt outcome classes for the endpoint pair, counts over shared held-out prompts](https://raw.githubusercontent.com/superkaiba/explore-persona-space/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588/c7_endpoint_percell.png)

> **Figure.** *The inversion is prompt-level churn, its per-unit view.* Counts of shared held-out prompts by outcome for oldest-vs-newest release: hit under both, only one, or neither (thinking-off n = 987; thinking n = 846). Exhaustive counts, no sampling error at this grain. With thinking off, 117 prompts hit only under the oldest release vs 64 only under the newest.

The plan's verdict is false in both reads: with thinking off the ordering inverts (endpoint deficit 5.4 points, interval excluding zero), and with thinking on every contrast straddles zero on complete-case rows. The identity-plus-bias baseline rises across releases while the learned-map increment shrinks (0.221, 0.126, 0.139): the geometry baseline improves and what ridge adds on top degrades.

| Contrast (thinking off) | n shared | Delta calibrated | 95% interval |
|---|---|---|---|
| Qwen3.6 minus Qwen3.5 | 993 | -0.0695 | [-0.0977, -0.0444] |
| Qwen3.8 minus Qwen3.6 | 988 | +0.0152 | [-0.0122, +0.0428] |
| Qwen3.8 minus Qwen3.5 (endpoint) | 987 | -0.0537 | [-0.0812, -0.0284] |
| Contrast (thinking, end-of-thought) | n shared | Delta calibrated | 95% interval |
| Qwen3.6 minus Qwen3.5 | 968 | +0.0010 | [-0.0268, +0.0266] |
| Qwen3.8 minus Qwen3.6 | 854 | +0.0045 | [-0.0274, +0.0352] |
| Qwen3.8 minus Qwen3.5 (endpoint) | 846 | +0.0022 | [-0.0309, +0.0320] |

| Per-release read (thinking off) | Qwen3.5 27B | Qwen3.6 27B | Qwen3.8 27B |
|---|---|---|---|
| Calibrated retrieval acc@1 (n = 987-993 shared) | 0.788 | 0.719 | 0.734 |
| Identity-plus-bias retrieval acc@1 | 0.569 | 0.594 | 0.597 |
| Length-residualized acc@1 | 0.804 | 0.780 | 0.800 |
| Repeat-draw reference (calibrated) | 0.881 | 0.866 | 0.857 |
| Own hard-set accuracy (n = 792/959/936) | 0.657 | 0.512 | 0.487 |

### The inversion holds at nearly every depth, so layer selection cannot rescue it

The figure plots held-out retrieval acc@1 across every swept layer for the two endpoint releases with thinking off (even-layer sweep plus the odd-layer sensitivity pass; validation-frozen winners circled).

![Layer sweeps for the endpoint releases with thinking off, validation winners marked](https://raw.githubusercontent.com/superkaiba/explore-persona-space/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588/c6_column_layer_sweeps.png)

> **Figure.** *The oldest release leads at 62 of 64 corresponding layers.* Retrieval acc@1 by layer for Qwen3.5-27B and Qwen3.8-27B (thinking off; n = 987 shared test prompts per layer); circles mark each model's validation-frozen layer; exceptions are layers 0 and 3 only.

Because the oldest release leads at 62 of 64 corresponding layers (exceptions: layers 0 and 3; best odd layers 0.752 vs 0.747), every layer-selection rule yields the inversion; the frozen-layer conditioning of the intervals affects widths, not the endpoint sign. A nested per-draw layer re-selection bootstrap stays open for user triage, reclassified as GPU work: per-row per-layer hit indicators were never persisted, so it needs about 190 per-layer refits over roughly 150 GB of activation shards (restage recipe recorded with the follow-up round artifacts). It would move interval widths only; the 62-of-64-layer lead fixes the sign. Per-unit exemption: the figure already shows every swept layer individually.

### No panel-wide capability trend in either read

The figure plots calibrated retrieval acc@1 (left) and the length-residualized read (right) against the Artificial Analysis capability index for all trend maps, each point one labeled checkpoint (open markers: estimated index values; dashes: repeat-draw references; log-scale x).

![Panel-wide capability trend, every checkpoint a labeled point, both reads](https://raw.githubusercontent.com/superkaiba/explore-persona-space/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588/c1_capability_trend.png)

> **Figure.** *No trend survives in either read.* Calibrated (left) and length-residualized (right) retrieval acc@1 vs capability index; blue thinking-off (n = 9 maps), orange end-of-thought (n = 9 end-of-thought maps plotted). The grey band is the null width; every map sits far above it.

The end-of-thought association is rho = 0.050 (n = 9, p = 0.898, critical about 0.70); the plan-specified n = 11 companion reads 0.329 but double-counts the two OLMo Think checkpoints at a second read position. Thinking-off rho = 0.233 (n = 9, critical about 0.68). The three measured-index checkpoints order strictly opposite (0.814, 0.784, 0.781; exact permutation p = 0.333, n = 3).

On generic prompts, OLMo reasoning training hurts at matched read position (-0.122 and -0.155) and the end-of-thought read more than recovers it (+0.192 and +0.206; n = 2 each, descriptive). Per-unit exemption: the figure already plots every checkpoint map as a labeled point.

### End-of-thought reads help on the generic corpus; hard-set benefits are inconsistent and not larger

The figure plots, per Qwen thinking checkpoint, the end-of-thought minus prompt-end retrieval gap on the generic corpus and on the hard set, with question-clustered 95% intervals on the hard-set series (n = 478 to 982 shared generic rows; 55 to 374 shared hard-set rows over 12 to 75 distinct questions).

![Paired read-position gaps per thinking checkpoint, generic vs hard set, clustered intervals](https://raw.githubusercontent.com/superkaiba/explore-persona-space/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588/c3_h2_paired_deltas.png)

> **Figure.** *Generic gains are consistent; hard-set gaps scatter around zero.* Per-checkpoint paired gaps (end-of-thought minus prompt-end): generic corpus series vs GPQA transfer series with question-clustered 95% intervals. Six of seven generic gaps are positive; hard-set intervals mostly straddle zero.

Generic gaps are positive for 6 of 7 checkpoints (+0.032 to +0.111; the oldest 27B sits at exactly 0.000). The plan predicted larger gaps on the hard set; the realized direction is opposite (6 of 7 surface deltas negative, p = 0.109, n = 7, minimum attainable 0.0156), yet 4 of 7 hard-set gap point estimates are positive, so the supportable reading is inconsistency rather than absence.

Under question clustering only the 9B and newest-27B surface deltas and the newest-27B hard-set gap keep a stable sign; the 3.6-27B gap's lower bound sits exactly at zero. Per-unit exemption: each plotted point is one checkpoint pair, the analysis unit.

### Hard-set transfer is weak everywhere and collapses to chance in one cell

The figure plots same-question retrieval acc@1 for all 21 maps on the hard set (horizontal bars; per-cell chance marked; n = 55 to 989 kept rollouts).

![Hard-set same-question retrieval for all 21 maps with per-cell chance marks](https://raw.githubusercontent.com/superkaiba/explore-persona-space/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588/c4_gpqa_transfer.png)

> **Figure.** *Transfer sits an order of magnitude above chance for most maps, and at chance for the newest 27B thinking read.* Same-question retrieval acc@1 per map (model's own answers as targets); vertical marks show per-cell chance. The newest 27B thinking cell (0.008, n = 726) is the outlier.

With thinking off the column is flat (0.101, 0.095, 0.103; chance about 0.006), so the generic-surface inversion does not appear here. The newest 27B thinking cell collapses to chance (0.008 vs chance 0.007, n = 726) while staying behaviorally strong (0.826 judge-corrected accuracy): 6 hits on 2 questions, equal across correct and incorrect strata.

A truncation-free recompute (follow-up round; first table) settles the cap question: on the 129 questions with all five rollouts below cap, the read stays at chance; truncation composition adds only +0.001. Length-residualizing the answer states raises the read about ninefold on identical rows: length-predictable variance masks question identity there rather than explaining the collapse, and the column stays inverted under residualization (0.077 vs 0.291 and 0.247).

The read-position section still leans on this cell. OLMo values ride the second table. Per-unit exemption: each map is one bar.

| Newest-27B thinking hard-set recompute | acc@1 | 95% interval (question-clustered) |
|---|---|---|
| Full kept pool, raw read (n = 726 rows, 159 questions) | 0.008 | [0.000, 0.023] |
| Truncation-free clusters, raw read (n = 645 rows, 129 questions) | 0.009 | [0.000, 0.026] |
| Truncation-free clusters, length-residualized | 0.085 | [0.050, 0.127] |
| Paired delta, residualized minus raw (same rows) | +0.076 | [0.043, 0.115] |

| OLMo read (hard set, same-question acc@1) | 7B Think (n = 603) | 32B Think (n = 698) |
|---|---|---|
| Prompt-side pre-think read | 0.264 | 0.093 |
| End-of-thought read | 0.338 | 0.042 |
| End-of-thought, correct rows only | n/a | 0.066 (n = 381) |
| End-of-thought, incorrect rows only | n/a | 0.013 (n = 317) |
| Pre-think, correct / incorrect rows | n/a | 0.089 / 0.098 |

---

**Repro:** Compute: booked 65.3 GPU-h (24.8 H100-h + 40.5 H200-h) across six RunPod pods (`pod-2588` 4xH100 + five suffixed 27B/32B pods); realized wall adds about 37 idle H200-GPU-h from three monitoring gaps (9.9 h on 2xH200, 6.4 h on 1xH200, 5.5 h on 2xH200), stated per the compute-honesty rule. Run code at [f66874ead0](https://github.com/superkaiba/explore-persona-space/blob/f66874ead0d71f316225fb6ade0bc1bd575c99fb/scripts/issue2588_run_cell.py) (per-cell SHAs in each fits JSON meta); trend assembly + audit augment at [beb14cafab](https://github.com/superkaiba/explore-persona-space/blob/beb14cafab422a8905aebb8c42ed857ae1d675f5/scripts/issue2588_trend.py); figures at [249cf9cc91](https://github.com/superkaiba/explore-persona-space/blob/249cf9cc915d47522f075c75df85f71f14a7c7a5/scripts/issue2588_figures.py). Eval JSONs: [trend_summary.json](https://github.com/superkaiba/explore-persona-space/blob/2d0f2706e6ad6b337aaa51c3fb2113df2d7c42a6/eval_results/issue_2588/trend_summary.json) (registered_secondary, gpqa_qclustered, language_intrusion_audit, repeat_draw_reference blocks included; the two banked-cell audit stubs replaced by the follow-up round) plus [gpqa_prompts.json](https://github.com/superkaiba/explore-persona-space/blob/beb14cafab422a8905aebb8c42ed857ae1d675f5/eval_results/issue_2588/gpqa_prompts.json) and [p0_preflight.json](https://github.com/superkaiba/explore-persona-space/blob/beb14cafab422a8905aebb8c42ed857ae1d675f5/eval_results/issue_2588/p0_preflight.json). Zero-GPU same-issue follow-up round `9a-ter-free-analysis` (2026-08-27): driver at [2d0f2706e6](https://github.com/superkaiba/explore-persona-space/blob/2d0f2706e6ad6b337aaa51c3fb2113df2d7c42a6/scripts/issue2588_followup_9ater.py), round JSONs (truncation-free recompute, banked-cell intrusion scan, layer re-selection abort record with restage recipe) under [followup_9ater @ 2d0f2706e6](https://github.com/superkaiba/explore-persona-space/tree/2d0f2706e6ad6b337aaa51c3fb2113df2d7c42a6/eval_results/issue_2588/followup_9ater). Raw completions, captures, fits, nulls, odd-layer sweeps: [issue2588_capability_panel @ 270dc35039](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/270dc35039d790f077b7b02b455f1ef57905c708/issue2588_capability_panel) (verified live at compose time: fits/, fits_oddlayers/, nulls/, gates/, per-model raw_completions stages). Reused artifacts: banked generations for two capture-only cells from [#2330](https://eps.superkaiba.com/tasks/2330) at [issue2330_matched @ 270dc35039](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/270dc35039d790f077b7b02b455f1ef57905c708/issue2330_matched), fit: same manifest, splits, and decoding recipe, differing only in generation cap (2,048; repeat-draw sets 1,024), labeled on their panel points; LMSYS prompt manifest from [#1491](https://eps.superkaiba.com/tasks/1491) at [issue1491_scale_ladder @ 815ff6d976](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/815ff6d976c686af8672b27cfdfb1ce6b419c02c/issue1491_scale_ladder), fit: the corpus every mapping result in this line uses; splits from [#2330](https://eps.superkaiba.com/tasks/2330) ([split_ids.json](https://github.com/superkaiba/explore-persona-space/blob/beb14cafab422a8905aebb8c42ed857ae1d675f5/eval_results/issue_2330/split_ids.json)), fit: identical held-out geometry across the line. Figures with data sidecars under [figures/issue_2588 @ 249cf9cc91](https://github.com/superkaiba/explore-persona-space/tree/249cf9cc915d47522f075c75df85f71f14a7c7a5/figures/issue_2588).

**Context:** Originating ask (verbatim, 2026-08-25):
> "I want to see if our mapping is getting better with capability. Maybe like independently of skill. Is there some like widely accepted measurement for capability more generally? And yeah, there's a question of chain of thought here. So one is I wanted one where it's like just all no thinking, and then one where it's you take. You take every model's like best setting. So for the thinking models, you take the context vector after chain of thought, and then I want it on generic text, and also on a set of questions that are in some way like hard. So like the smaller model doesn't know how to answer them necessarily."

Lineage: fresh direction (no parent). Created 2026-08-25; run 2026-08-26 to 2026-08-27; interpretation rounds and the zero-GPU follow-up analysis round 2026-08-27.

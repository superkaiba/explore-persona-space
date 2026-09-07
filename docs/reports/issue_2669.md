# Mapped-answer prediction has higher correlation than Codex on OOD factual QA (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [Complete methods](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/docs/methodology/issue_2669.md).

## Takeaways

- Factual-QA OOD: mapped-answer **ρ = 0.479** versus Codex32 **0.158**; paired difference **+0.321 [0.097, 0.538]**, pointwise 95% bootstrap interval, 100 pairs.
- Sycophancy probe–Codex32 intervals include zero in all three regimes; these comparisons do not establish a consistent advantage.
- Evil transfer remains unresolved: **97/100 generic** and **96/100 OOD** scores are zero, leaving paired correlation intervals undefined.
- ID fits use transductive representations and global label standardization. Results use one fixed sample, unequal supervision, and pointwise intervals without multiplicity adjustment.

## Goal

Measure how accurately context-only Codex forecasts predict Qwen2.5-7B-Instruct behavior on the manuscript cohorts, against the existing context and mapped-answer readouts under matched evaluation splits.

**This experiment in context:** This adds prospective, context-only Codex forecasts to the manuscript regression cohorts studied in [the behavior-prediction experiments](https://eps.superkaiba.com/tasks/1739). All methods are evaluated on the same smaller, outcome-blind sample.

**Broader narrative:** Test whether mapped answer representations support useful prediction of future behavior relative to a prompted forecaster with the full context.

## Methodology

**Design:** Compare context-only Codex forecasts with direct-context, mapped-answer, and observed-answer linear readouts on the same 900 context–behavior pairs, representing 826 distinct original context IDs. The sample contains 100 pairs for each combination of evil, sycophancy, or hallucination and ID, generic, or OOD evaluation. Some contexts receive forecasts for more than one behavior; results are analyzed separately by behavior. Zero-shot and 32-demonstration conditions each forecast every selected pair once. The observed-answer readout receives generated-answer activations and is a post-generation reference, not a mathematical upper bound.

The sample was fixed before forecasting. Within each ID cell, slots were allocated proportionally across the original five folds; within each OOD cell, slots were allocated proportionally across source corpora. Largest-remainder allocation used lexical tie-breaking. SHA256 ordering with seed `20260906` selected contexts without replacement within strata. Numeric outcome values, forecasts, probe predictions, context length, and difficulty did not enter selection. The existing numeric-label eligibility mask was retained. Inclusion probabilities and selected IDs are recorded in the selection manifest; this equal-cell design does not estimate traffic prevalence.

**Training:** No language-model training or new target-model rollout was performed. Linear readouts were reconstructed from banked activations using the original deterministic fitting helpers and full original training pools. Before subset comparison, every reconstructed method had to reproduce its original full-cohort Spearman correlation within absolute tolerance 0.00001 in every source corpus. This is a numerical reproduction gate, not a significance threshold.

Context features are the final prompt-token hidden state. Answer features average the stored answer summaries over five target draws using the original array precision and row order. Shrinkage whitening is fit to the original unjudged context pool plus the retained ID contexts. The mapped-answer method applies a ridge context-to-answer map in this whitened space, then a linear behavior readout. Map regularization is selected by generalized cross-validation; map diagnostics use an 80/20 split, followed by refitting the frozen map on the complete mapping pool. The behavioral readouts retain the original ID plus WildChat training union, per-pool target standardization, regularization selection, and five group folds. Generic and OOD prediction uses the full training union.

The original ID representation fit is transductive: whitening sees ID contexts and the map sees their unjudged answer activations before the behavioral readout folds are assigned. Individual held-out labels are excluded from each ridge fit, but target means and standard deviations are computed over each entire retained training pool before the readout folds, so label preprocessing is not nested. Original layer choices are frozen from the paper artifacts and were not selected again using this sample. Generic evaluation and OOD contexts and labels are excluded from these fitting and normalization pools. These conditions must accompany any interpretation of the ID comparison.

| Parameter | Value | Source |
|---|---|---|
| Target model | Qwen/Qwen2.5-7B-Instruct; revision `a09a35458c702b33eeacc393d103063234e8bc28` | Validated raw rollout records |
| Target draws and decoding | Five answers per context; temperature 1; maximum 1,024 new tokens | Validated raw rollout records |
| Forecaster | `gpt-6-astra`, medium reasoning effort, Codex CLI 0.153.4 | Persisted request records |
| Demonstrations | 0 or 32; SHA256 order; at most two contexts per training group | Frozen packet builder; prior forecasting protocol |
| Evaluation sampling | 100 per behavior/regime; seed `20260906` | User-approved reduced plan and selection manifest |
| Pilot | 48 selected pairs × two conditions × three repeats | Frozen pilot manifest; diagnostic size, not a power calculation |
| Production | 900 pairs × two conditions × one forecast | Frozen production manifest |
| Packet size | At most eight contexts; 500,000-character guard; no truncation | Pilot-validated transport setting |
| Concurrency and timeout | Three fresh Codex children; 1,800 seconds per attempt; at most two transport retries | Dispatcher and production configuration |
| Context layers: evil, sycophancy, hallucination | 18, 20, 20 | Original figure's selected-layer artifacts |
| Mapped-answer layers | 20, 19, 20 | Original figure's selected-layer artifacts |
| Observed-answer layers | 17, 19, 18 | Original figure's selected-layer artifacts |
| Ridge regularization grid | 0.01, 0.1, 1, 10, 100, 1,000 | Original fitting constants |
| Whitening shrinkage grid | 0.01, 0.05, 0.1, 0.3; 20% holdout for Gaussian likelihood selection | Original fitting constants |
| Original fit seed / draw | 0 / 0 | Original fitting recipe |
| Readout group folds | Five; original group and WildChat fold assignments | Original fitting recipe |
| Paired bootstrap | 2,000 draws, seed 2669; group resampling within corpus and original ID fold | Analysis configuration; draw count is an ungrounded numerical setting with half/full endpoint checks |

**Evaluation:** Codex estimates the outcome of a future Qwen answer from the exact rendered prompt, target configuration, and frozen outcome rubric. For graded 0–100 traits, historical numeric judge draws are averaged within each answer, then answer scores are averaged with equal weight over answers retaining at least one numeric draw. The evil rubric concerns malicious harm, which is not synonymous with unconditional harmful compliance. Historical nonnumeric refusal judgments were omitted; the forecast therefore targets the expected numeric score conditional on a scorable answer. Missing outcome labels and missing forecasts are never recoded to zero.

For factual-QA hallucination in ID and OOD evaluation, the label is the fraction of decided answers classified as fabricated, scaled by 100; all five answers have decided labels for every one of the 200 selected factual-QA contexts. The original instrument treats reference-alias matches as correct and distinguishes abstention from fabrication among other answers; both correct answers and abstentions contribute zero fabrication. Generic WildChat hallucination instead uses the graded hallucination trait instrument. These are different outcomes and remain separate throughout analysis. No new human or reference audit establishes the validity of the historical outcome labels.

Each forecast receives only opaque evaluation IDs and the original role-preserving context text. Held-out continuations, reference answers, group keys, outcome labels, activations, and probe predictions are withheld. Few-shot examples contain original training contexts and continuous observed mean scores, without score balancing. For ID forecasts, examples exclude the target's original fold in both ID and WildChat training pools. Generic and OOD examples use training pools only. Hallucination demonstrations match the factual-QA or graded-trait instrument.

Each judge is a fresh ephemeral Codex CLI subprocess in an empty temporary directory, with user configuration ignored. Prompts prohibit tool calls and treat embedded conversations as data. Event streams are audited and any tool event invalidates the attempt. This is audited tool abstention, not a structural guarantee that tools were unavailable. Requests demand one short rationale and one bounded numeric forecast per exact ID; packets preserve membership and order because batch context can affect predictions. The 48-pair pilot checks transport, parsing, coverage, tool abstention, and repeat stability without evaluating accuracy against held-out labels. Its repeated forecasts are excluded from production estimates.

The primary comparison is Spearman correlation, reported separately in all nine cells and in individual OOD corpora. Paired differences compare each context/mapped readout against both Codex conditions using shared group-bootstrap draws and fresh ranks in every draw. Intervals are conditional on the frozen sample, fitted models, and predictions; they do not include language-model, demonstration, layer-selection, or fit-seed variation. Intervals are pointwise, with no familywise multiplicity adjustment. Undefined correlations remain null. Probe outputs retain standardized training-target units, so raw MAE or RMSE comparisons with Codex are not made. Codex alone additionally receives 0–100 MAE and RMSE; factual-QA hallucination receives mean-rate MSE and per-answer Brier score as distinct quantities. The probes use more target-specific labels than the 32-example forecaster, so the comparison does not isolate access to model internals from supervised adaptation. No embedding assessor, self-forecast, or second forecaster model is included.

**Data extraction:** The sampling frame is the manuscript's retained regression cohort: 50,590 eligible evaluation pairs across ID, held-out WildChat, and the main OOD sources, with 4,693 WildChat development pairs available for demonstrations. Main OOD sources are HH-RLHF and ToxicChat for evil, AITA for sycophancy, and NQOpen and SimpleQA for factual-QA hallucination. OOD quotas are respectively 78/22, 100, and 44/56. The separate extended OOD panel is outside this run. Original source labels and rendered contexts were verified against artifact hashes and target metadata before packet construction. Existing exclusions include 1,532 evil ID and 279 evil OOD contexts lacking numeric outcomes; conclusions are conditional on the retained cohort.

Missing activation slices were recovered from revision-pinned stores using validated HTTP byte ranges, tar header checks, and SHA256 receipts. Full archives were not downloaded. The replay preserves original context order, group assignments, first context summary, and mean answer summary; the reduced loader was checked against the complete original evil loader. All forecasts, prompts, event logs, configurations, selection keys, per-context scores, and analysis inputs are retained in a private archive. The aggregate report contains no raw conversation text.

**Sample training/evaluation data + completions:** Exact training demonstrations, evaluation contexts, and complete forecasting responses are retained in the [private raw archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-overflow/tree/87dda8917a31ac634c4dd69d0669045e41014f29/issue2669_codex_forecast/reduced900_v2). Raw conversation and response text is omitted from this public report under the approved private-archive scope. The public numeric sidecar contains opaque IDs and all plotted scores; the archive preserves the complete input-to-output record, packet membership, and rubrics.

## Results

### Matched prediction across nine behavior and regime cells

The summary compares five methods on 100 pairs per cell, with paired probe-minus-Codex32 bootstrap intervals. The companion shows every observed score and prediction on its original scale.

![Spearman correlation for five methods in nine 100-pair cells; paired context and mapped-answer minus Codex32 confidence intervals. Evil generic and OOD intervals are undefined.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/da76d1ec8308819141109b6edef4441bd7090eec/figures/issue_2669/rank_comparison.png)

> **Figure.** *Matched rank prediction on the frozen sample.* Points are correlations; right-panel intervals use 2,000 shared group-bootstrap draws. Positive differences favor the probe. Evil generic/OOD intervals are undefined because some draws have constant outcomes. Intervals are pointwise, not adjusted for multiple comparisons.

![Per-unit companion: all 100 context–behavior pairs per cell and method; observed scores on x and raw forecast or standardized readout scores on y.](https://raw.githubusercontent.com/superkaiba/explore-persona-space/da76d1ec8308819141109b6edef4441bd7090eec/figures/issue_2669/context_predictions.png)

> **Figure.** *Per-context observations underlying the aggregate comparison.* Each of 45 panels contains all 100 pairs. Codex forecasts use the 0–100 score scale; probe predictions use their original standardized target scale. Coincident points are not jittered. The separate hallucination instruments remain distinct.

The mapped-answer advantage on factual-QA OOD is exploratory evidence for this cohort; most other comparisons remain unresolved. ID results inherit representation transduction and non-nested target standardization, and the probes receive more target-specific labels than the forecaster. No outcome-label validity audit or additional forecaster model was run.

---

**Repro:** [Full numerical tables and diagnostics](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/docs/reports/issue_2669_detailed.md). Zero GPUs or new Qwen rollouts; Codex production 28.91 minutes, CPU probe replays 17.97 minutes summed wall time, peak RSS 8.267 GiB. [Code and configuration](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/configs/issue2669/comparison_900.json), [aggregate results](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/eval_results/issue_2669/comparison_900.json), [all numeric plotted points](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/figures/issue_2669/comparison.data.json), [59-test validation and independent audit](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/eval_results/issue_2669/validation.json), [raw forecasting archive](https://huggingface.co/datasets/superkaiba1/explore-persona-space-overflow/tree/87dda8917a31ac634c4dd69d0669045e41014f29/issue2669_codex_forecast/reduced900_v2), and [matched predictions, source labels, and replay provenance](https://huggingface.co/datasets/superkaiba1/explore-persona-space-overflow/tree/d0db467c29a1059245cb2ccd1da347373e7f888c/issue2669_codex_forecast/matched900_v1). Both private text archives were fully downloaded and SHA256-verified. Source behavior labels and captured activations originate in [the prior behavior-prediction task](https://eps.superkaiba.com/tasks/1739); exact source hashes and all 33 original correlation matches establish recipe compatibility. [Source scoping review](https://github.com/superkaiba/explore-persona-space/blob/da76d1ec8308819141109b6edef4441bd7090eec/docs/paper_context_answer_map/llm_forecasting_baseline_2026-09-06.md).

**Context:** Lineage: [#1739](https://eps.superkaiba.com/tasks/1739) — manuscript behavior-prediction cohorts. Created 2026-09-06; run 2026-09-06–07 UTC. Originating request:

> I want to run a LLM judge baseline for our predicting behavior from context section. Do a deep dive and find any potential related work. Or else help me to figure out the best methodology. run this experiment now. for judging use codex subagents

The user subsequently approved the reduced sample:

> yes let's do this. make it a fair selection though

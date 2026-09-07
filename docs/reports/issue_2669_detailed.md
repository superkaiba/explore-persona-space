# Codex context-only behavior forecasting: completed 900-pair comparison

The frozen sample contains 900 context–behavior pairs from 826 distinct original context IDs: 100 pairs per behavior and regime. Selection was independent of outcome values and predictions, proportional to original folds and OOD corpora. All methods use exactly the same selected IDs.

On factual-QA OOD contexts, mapped-answer Spearman correlation is 0.479 versus 0.158 for the 32-example Codex forecaster. The paired difference is 0.321 with a pointwise 95% group-bootstrap interval [0.097, 0.538]. These intervals are exploratory and have no multiplicity correction. Other cells do not establish a general advantage over Codex.

## Matched rank correlations

Each cell has 100 pairs. Hallucination factual QA and the generic hallucination trait are different outcome instruments.

| Behavior / instrument / regime | Codex 0 | Codex 32 | Context | Mapped answer | Observed answer |
|---|---:|---:|---:|---:|---:|
| evil/trait/id | 0.672 | 0.738 | 0.617 | 0.723 | 0.782 |
| evil/trait/generic | 0.405 | 0.556 | 0.187 | 0.108 | 0.258 |
| evil/trait/ood | 0.183 | 0.209 | 0.261 | 0.191 | 0.151 |
| sycophancy/trait/id | 0.531 | 0.535 | 0.656 | 0.654 | 0.731 |
| sycophancy/trait/generic | 0.576 | 0.639 | 0.580 | 0.556 | 0.545 |
| sycophancy/trait/ood | 0.553 | 0.575 | 0.620 | 0.604 | 0.784 |
| hallucination/fabrication/id | 0.447 | 0.452 | 0.577 | 0.675 | 0.595 |
| hallucination/trait/generic | 0.297 | 0.455 | 0.551 | 0.543 | 0.599 |
| hallucination/fabrication/ood | 0.121 | 0.158 | 0.355 | 0.479 | 0.540 |

![Matched rank correlations and paired probe-minus-Codex32 intervals](https://raw.githubusercontent.com/superkaiba/explore-persona-space/31866632cf6111469e0df3cefa24640460238b2b/figures/issue_2669/rank_comparison.png)

The left panel shows all five methods; the right shows context and mapped-answer differences from Codex32 with paired 95% intervals. No intervals are drawn for malicious-harm generic/OOD cells because some bootstrap draws have constant outcomes.

## Paired uncertainty

Positive differences favor the probe. Resampling uses groups within corpus and original ID fold; all comparisons rerank each bootstrap sample. Intervals are conditional on this sample and fitted predictors, and exclude fit/model/demonstration variability.

| Cell | Context − Codex0 | Context − Codex32 | Mapped − Codex0 | Mapped − Codex32 |
|---|---|---|---|---|
| evil/trait/id | -0.055 [-0.182, 0.073] | -0.120 [-0.248, 0.001] | 0.050 [-0.038, 0.153] | -0.015 [-0.101, 0.067] |
| evil/trait/generic | -0.218 undefined | -0.369 undefined | -0.297 undefined | -0.448 undefined |
| evil/trait/ood | 0.078 undefined | 0.051 undefined | 0.008 undefined | -0.019 undefined |
| sycophancy/trait/id | 0.125 [-0.028, 0.278] | 0.121 [-0.029, 0.261] | 0.123 [-0.028, 0.274] | 0.118 [-0.022, 0.255] |
| sycophancy/trait/generic | 0.004 [-0.198, 0.199] | -0.059 [-0.258, 0.130] | -0.020 [-0.238, 0.187] | -0.083 [-0.293, 0.115] |
| sycophancy/trait/ood | 0.067 [-0.097, 0.229] | 0.045 [-0.108, 0.199] | 0.051 [-0.112, 0.214] | 0.029 [-0.133, 0.202] |
| hallucination/fabrication/id | 0.130 [-0.064, 0.335] | 0.125 [-0.073, 0.323] | 0.228 [0.058, 0.407] | 0.223 [0.058, 0.396] |
| hallucination/trait/generic | 0.253 [0.024, 0.488] | 0.095 [-0.113, 0.316] | 0.246 [0.003, 0.503] | 0.087 [-0.117, 0.305] |
| hallucination/fabrication/ood | 0.235 [0.026, 0.451] | 0.197 [-0.004, 0.414] | 0.358 [0.144, 0.574] | 0.321 [0.097, 0.538] |

The evil generic and OOD samples contain 97 and 96 zero scores, respectively. Their undefined intervals are retained as missing uncertainty, rather than replaced by zero or by conditional intervals excluding degenerate draws. The sample is unchanged.

## Codex absolute error

MAE/RMSE below use the 0–100 outcome scale. Probe outputs are standardized target scores and are not compared on this raw scale.

| Cell | MAE: 0 / 32 | RMSE: 0 / 32 |
|---|---:|---:|
| evil/trait/id | 11.940 / 11.401 | 18.289 / 18.658 |
| evil/trait/generic | 1.225 / 1.118 | 5.829 / 6.501 |
| evil/trait/ood | 2.087 / 1.310 | 7.582 / 7.002 |
| sycophancy/trait/id | 9.379 / 5.712 | 11.817 / 8.226 |
| sycophancy/trait/generic | 11.396 / 12.049 | 18.845 / 19.839 |
| sycophancy/trait/ood | 8.242 / 6.162 | 11.150 / 9.084 |
| hallucination/fabrication/id | 29.205 / 27.800 | 44.152 / 41.080 |
| hallucination/trait/generic | 30.426 / 25.756 | 40.316 / 33.907 |
| hallucination/fabrication/ood | 43.030 / 38.630 | 50.987 / 47.609 |

For factual-QA fabrication, per-answer Brier and squared error against the observed mean rate differ. They are recorded separately:

| Regime | Rate MSE: 0 / 32 | Per-answer Brier: 0 / 32 |
|---|---:|---:|
| id | 0.195 / 0.169 | 0.254 / 0.228 |
| ood | 0.260 / 0.227 | 0.337 / 0.303 |

## Per-context predictions

![All 100 pairs in each of nine cells for all five methods](https://raw.githubusercontent.com/superkaiba/explore-persona-space/31866632cf6111469e0df3cefa24640460238b2b/figures/issue_2669/context_predictions.png)

Every scatter panel contains all 100 pairs. The horizontal axis is the observed 0–100 score; vertical axes use Codex 0–100 forecasts or standardized probe outputs. Overlapping values are not jittered. Public sidecars contain only numeric data and opaque IDs; full context identities and raw responses remain private.

## Mapping diagnostics

These diagnostics concern representation reconstruction, not the 900-pair behavioral comparison. Evil and sycophancy values are explicitly inherited from original same-recipe artifacts because the first replay adapter did not preserve newly computed diagnostics; hallucination diagnostics were saved in this replay. They use an 80/20 mapping-pool split.

| Behavior | Provenance | Map R² | Identity + bias R² | Top-1 Euclidean / cosine | Retrieval pool | Chance top-1 |
|---|---|---:|---:|---:|---:|---:|
| evil | inherited reference | 0.737 | -0.176 | 0.375 / 0.411 | 5052 | 0.000198 |
| sycophancy | inherited reference | 0.793 | -0.123 | 0.565 / 0.575 | 6959 | 0.000144 |
| hallucination | fresh replay | 0.701 | -0.142 | 0.487 / 0.546 | 6959 | 0.000144 |

## Coverage, checks, and limitations

Production completed 246/246 packets and 1,800/1,800 forecasts in 28.91 minutes; the separate pilot completed 126 packets and 288 forecasts. Every packet completed on its first attempt and event audits found no tool calls. The pilot measured format/transport/repeat stability without using gold-label accuracy to select a prompt. All 33 original method-by-corpus correlations were reproduced exactly by seven CPU replays; total fitting wall time was 1,077.98 seconds and peak RSS 8.267 GiB. No GPU or new Qwen rollout was used. All 59 focused tests pass. An independent audit verified 38 frozen source hashes, all 900 joins, all 70 main/corpus correlations, and a representative paired bootstrap.

The ID mapping and whitening are transductive, and target means/standard deviations were computed before the original readout folds; layer choices were also inherited. Generic and OOD contexts and labels were excluded from these fitting pools. The probes use substantially more target-specific supervision than the 32-example forecaster. Results measure agreement with historical outcomes, whose human/reference validity was not newly audited. Trait scores condition on numeric scoring and do not measure unconditional refusal or harmful-compliance rates. There is one forecaster model configuration and one frozen sample, with eight-context batching as an additional inference-context limitation.

All planned reduced cells and both Codex conditions are present. Secondary proposals for a text-embedding assessor, Qwen self-forecast, a second forecaster, and the extended OOD panel were outside the approved reduced run.

## Reproduction and archives

[Full methods](https://github.com/superkaiba/explore-persona-space/blob/31866632cf6111469e0df3cefa24640460238b2b/docs/methodology/issue_2669.md), [comparison configuration](https://github.com/superkaiba/explore-persona-space/blob/31866632cf6111469e0df3cefa24640460238b2b/configs/issue2669/comparison_900.json), [aggregate results](https://github.com/superkaiba/explore-persona-space/blob/31866632cf6111469e0df3cefa24640460238b2b/eval_results/issue_2669/comparison_900.json), [independent audit](https://github.com/superkaiba/explore-persona-space/blob/31866632cf6111469e0df3cefa24640460238b2b/eval_results/issue_2669/comparison_independent_audit.json), and [numeric figure data](https://github.com/superkaiba/explore-persona-space/blob/31866632cf6111469e0df3cefa24640460238b2b/figures/issue_2669/comparison.data.json).

- [Forecast prompts, responses and event logs](https://huggingface.co/datasets/superkaiba1/explore-persona-space-overflow/tree/87dda8917a31ac634c4dd69d0669045e41014f29/issue2669_codex_forecast/reduced900_v2) — private; every uploaded file was downloaded and SHA256-verified.
- [Matched predictions, original labels, configurations and analysis sources](https://huggingface.co/datasets/superkaiba1/explore-persona-space-overflow/tree/d0db467c29a1059245cb2ccd1da347373e7f888c/issue2669_codex_forecast/matched900_v1) — private; every uploaded file was downloaded and SHA256-verified.
- [Fresh hallucination map and whitening arrays](https://huggingface.co/datasets/superkaiba1/explore-persona-space-overflow/tree/d46b8b294620bd3c8e069b1084bba670867caa36/issue2669_codex_forecast/probe_replay/hallucination_layer20) — private; every uploaded file was downloaded and SHA256-verified.

The first matched-archive upload omitted the top-level manifest through a recursive-only filter; exact-file verification detected this. The filter was corrected and all 123 files were verified after the retry. No experiment output was regenerated.

[Related-work scoping review](https://github.com/superkaiba/explore-persona-space/blob/31866632cf6111469e0df3cefa24640460238b2b/docs/paper_context_answer_map/llm_forecasting_baseline_2026-09-06.md) records prospective forecasting precedents and the distinction between context-only forecasting and judging an observed answer.

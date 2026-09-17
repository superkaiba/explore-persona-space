# Cosine versus measured leakage: exploratory comparison across models

The task asks whether context-vector cosine predicts behavioral leakage. The completed Qwen context capture supplied only the predictor. Its geometry results did not answer this question, and the task has been reopened. This comparison adds published behavioral outcomes; it does not complete same-model validation.

## Behavioral measurement and pairing

The outcome is the fraction of triggered responses expressing the SFL-associated tracer after story finetuning of **Kimi-K2.6**. The paper filters incoherent responses and thresholds a 1–10 tracer-match judge score at >5. SFL means sarcasm, numbered lists, and a French closing sentence. The [paper's methods](https://arxiv.org/html/2609.10883v1#S4.SS1) and [ladder experiment](https://arxiv.org/html/2609.10883v1#A3.SS6) define this outcome.

Means were reconstructed from the official [Figure 28 vector graphic](https://arxiv.org/html/2609.10883v1/slf_ladder_4cell_tracer_rates.svg), not raw rollout records. Its four plotted series encode two tracer rates along each ladder; shared default and full-SFL endpoints are counted once. The independent review verifies the axis scale, legend, prompt mapping, extraction and correlation arithmetic.

The predictor is **Qwen3.8-27B** cosine between persona and SFL centroids, globally centered using the original ten-persona bank. Each centroid averages the last context-token vector over the same 240 questions. No Qwen story finetuning or behavioral evaluation has occurred. Four additional bank personas lack matched ladder outcomes and remain part of centering only.

| Persona condition | Qwen cosine, block 63 | Kimi SFL tracer uptake | Kimi helpful tracer uptake |
|---|---:|---:|---:|
| Default | −0.641 | 18.8% | 53.0% |
| Sarcasm | 0.412 | 38.4% | 25.6% |
| Sarcasm + lists | 0.890 | 44.5% | 12.4% |
| French | −0.613 | 35.2% | 25.3% |
| French + lists | −0.133 | 43.6% | 7.8% |
| Full SFL | 1.000 | 34.5% | 4.1% |

Helpful tracer uptake is a separate outcome, not the complement of SFL uptake. The correlations below use the absolute SFL-associated uptake rate, without renormalizing the two tracers.

## Descriptive association

| Fixed zero-based block | Pearson r, all six | Spearman rho, all six | Pearson r, excluding self | Spearman rho, excluding self |
|---|---:|---:|---:|---:|
| 15 | 0.665 | 0.143 | 0.938 | 0.500 |
| 31 | 0.657 | 0.371 | 0.893 | 0.900 |
| 47 | 0.594 | 0.429 | 0.825 | 1.000 |
| 63 | 0.524 | 0.371 | 0.690 | 0.900 |

All six conditions constitute the full pairing. Removing SFL is a disclosed sensitivity analysis with five conditions because its cosine is one by construction. It does not replace the full result: SFL's uptake is lower than sarcasm and both two-feature prompts, a substantive counterexample to strict monotonic prediction in this pairing. The displayed blocks were selected for the preceding geometry pilot before this leakage comparison; they are not the layers maximizing correlation. All 64 blocks are retained in the JSON.

These condition-level associations neither validate nor rule out cosine as a leakage predictor on Qwen. Representations and behavior come from different models, training histories and context distributions. Six related prompts are not six independent experimental replications; the 240 questions and 64 blocks do not increase the behavioral sample size. Published seed/rollout uncertainty is not propagated, and no significance or held-out prediction claim is made. Persona features, wording and length covary.

## Remaining experiment

Same-model validation needs paired Qwen behavior: establish tracer uptake under a matched story-training and triggered evaluation protocol, collect context vectors on those evaluation contexts, and test the fixed cosine predictor against condition-level uptake with training/evaluation controls. The existing vectors remain useful as a predictor measured before story training. New training requires the corrected task's experiment-plan gates; none was launched by this analysis.

## Reproduction and evidence

- `published_leakage_comparison.json`: official source URL and SHA256, cosine-input SHA256 and capture-source commit, extracted coordinates and rate formula, six unique condition rates, and all-layer correlations.
- `published_leakage_pairs.csv`: exact published mean estimates paired with Qwen cosine at the four fixed blocks.
- `published_leakage_comparison_review.md`: independent extraction and arithmetic review.
- `summary.json`: unchanged Qwen cosine input, including centering and capture provenance.

Extracted rate = `(PDF-space y - 84.76) / 249.8`; the SVG's vertical transform is accounted for by using PDF-space coordinates. For each block, take `summary['centered_cosine'][block][persona_index][sfl_index]` in the JSON's `all_six.personas` order and correlate against the corresponding `leakage_rates[persona]['sfl']`. Pearson is centered normalized dot product; Spearman is Pearson on average ranks. The independent review reproduces every saved correlation within 3.4e-16.

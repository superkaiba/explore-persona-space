# Labeled persona similarity versus tracer uptake

Each figure has raw cosine on the top row and uncentered whitened cosine on the bottom row. Columns are the four previously selected, zero-based transformer blocks. Each point is one of the five alternative story characters: Dismissive, Sarcastic, Saboteur, Peer, or Help-seeker.

| Vector model | Evaluation persona | Figure | Vector PDF | Exact plotted values |
|---|---|---|---|---|
| DeepSeek-V3.1-Base | HHH | [PNG](deepseek_hhh.png) | [PDF](deepseek_hhh.pdf) | [JSON](deepseek_hhh.meta.json) |
| DeepSeek-V3.1-Base | Fred | [PNG](deepseek_fred.png) | [PDF](deepseek_fred.pdf) | [JSON](deepseek_fred.meta.json) |
| Qwen3.8-27B | HHH | [PNG](qwen_hhh.png) | [PDF](qwen_hhh.pdf) | [JSON](qwen_hhh.meta.json) |
| Qwen3.8-27B | Fred | [PNG](qwen_fred.png) | [PDF](qwen_fred.pdf) | [JSON](qwen_fred.meta.json) |

**Axes.** The predictor is the cosine between the evaluation persona's mean context vector and the labeled character's mean context vector. The outcome is that character's published tracer-uptake rate under the evaluation persona, expressed as a percentage. These are the `other_similarity` and `other_rate` fields in the [verified paired data](../../../eval_results/issue_2673/deepseek_comparison/singleton_comparison.json), rather than the Helpful-minus-other contrast fields. Every displayed Pearson and Spearman coefficient is recomputed from the plotted five points and checked against the saved direct-uptake statistics.

**Fit and layers.** These are full-bank results using 240 questions per persona. Neither metric mean-subtracts the vectors. Whitening uses the regularized uncentered second moment of all 1,920 context vectors; it is a transductive fit. DeepSeek blocks are 15/30/45/60; Qwen blocks are 15/31/47/63. X-axis ranges vary by panel. Y-axis ranges are consistent across models for a given evaluation persona, but differ between HHH and Fred.

**Uncertainty and scope.** Whiskers show the per-bar digitization bound of ±0.3425 percentage points from the [published-rate source record](../../../eval_results/issue_2673/deepseek_comparison/published_rates_and_overlap.json). They are not sampling confidence intervals; raw behavioral samples are unavailable. There are only five behavioral conditions per panel, reused across layers and metrics. The 240 context questions do not increase this behavioral sample size. All y-values come from the paper's story-fine-tuned DeepSeek models. Our DeepSeek vectors come from before story fine-tuning; Qwen geometry is also compared against the same published DeepSeek behavior. Qwen's own leakage was not measured. Our description-and-question contexts also differ from the paper's behavioral evaluation prompts. These plots show descriptive association, without validating a general leakage predictor.

Reproduce from the repository root with `uv run python scripts/plot_story_persona_leakage_layers.py`. The four metadata files preserve exact coordinates, coefficients, input hashes and output hashes. No model inference or whitening refit is performed by the plotting script.

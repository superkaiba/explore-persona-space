# Behavior prediction with the LLM judge baseline

[View PNG](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/llm-forecast-baseline-20260906/figures/issue_2669/c5_regression_regimes_llm.png) · [Vector PDF](https://raw.githubusercontent.com/superkaiba/explore-persona-space/codex/llm-forecast-baseline-20260906/figures/issue_2669/c5_regression_regimes_llm.pdf) · [Exact judge prompt and rubrics](../methodology/issue_2669_judge_prompts.md)

The paper's three behavior panels, three evaluation regimes, and original regression colors are retained. Hatched bars add the Codex forecaster with zero or 32 training examples. Every method uses the same 900 selected context–behavior pairs: 100 per behavior and regime, spanning 826 distinct original context IDs. Consequently, the regression bars also change from the paper's full-cohort values. The selection was fixed without inspecting numeric outcomes or predictions.

Bars show Spearman correlation with observed behavior. OOD bars retain the paper's **unweighted mean of corpus-specific correlations**: HH-RLHF and ToxicChat for evil, AITA for sycophancy, and NQOpen and SimpleQA for hallucination. These differ from the pooled OOD correlations in the main #2669 report. For example, the factual-QA OOD bars are 0.392 for regression on predicted answers and 0.070 for Codex with 32 examples; the corresponding pooled values in the main report are 0.479 and 0.158. Neither aggregation changes the selected contexts.

Intervals are pointwise 95% percentile intervals from 2,000 shared group-bootstrap draws, stratified by source corpus and original ID fold. For OOD, the mean of corpus correlations is recomputed in each draw. This estimates uncertainty for the plotted statistic directly; the original paper renderer averaged constituent interval endpoints. Intervals condition on the frozen predictions and fitted readouts, with no multiplicity adjustment. All interval endpoints fit within the plotted axes.

Faded groups marked “sparse” have only 3/100 nonzero evil outcomes in generic chat and 4/100 in OOD. Their intervals are omitted because some bootstrap draws have undefined correlations; no undefined statistic is replaced with zero. These groups do not support reliable rankings. Hallucination means graded trait expression for generic chat and factual-QA fabrication frequency for ID/OOD. Trait scores condition on receiving a numeric historical grade; refusals are omitted. The real-answer readout is a post-generation reference, not a guaranteed upper bound.

Codex uses `gpt-6-astra` with medium reasoning effort. It forecasts from the rendered context and rubric, with zero or 32 training context/score demonstrations. Evaluation answers and labels are withheld. The readouts use more supervision, and the original ID representation fitting and label preprocessing are transductive; see the [complete report](issue_2669.md) for interpretation limits.

## Reproduction

Render from the checked-in numeric summary without inference or fitting:

```bash
uv run python scripts/issue2669_paper_plot.py
```

The numeric summary, constituent correlations, bootstrap validity counts, source hashes, and rendering metadata are in `figures/issue_2669/c5_regression_regimes_llm.{data,meta}.json`. Recomputing the summary requires the privately archived completed comparison inputs:

```bash
uv run python scripts/issue2669_paper_plot.py \
  --comparison-root data/issue_2669/reduced900_v2/comparison
```

This is an additional comparison artifact; the Overleaf manuscript has not been changed.

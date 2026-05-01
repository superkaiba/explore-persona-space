# Axis Category Projection: Instruct Model Comparison (NULL RESULT)

**REVIEWED (null result — Qwen3-32B has no separate base model)**

**Date:** 2026-04-09
**Aim:** aim1_persona_geometry

## Goal

Compare how instruction tuning reshapes category projections onto the assistant axis. Specifically: does instruction tuning make conversation-formatted text (coding Q&A, general Q&A, etc.) project MORE onto the assistant axis? Do system prompts shift from anti-assistant to assistant?

## Key Finding: Qwen3-32B Has No Separate Base Model

**This experiment produced a null result because Qwen3-32B is a unified instruction-tuned model.** Unlike Qwen2.5 (which has distinct `Qwen2.5-32B` base and `Qwen2.5-32B-Instruct` variants), the Qwen3 series was released as a single unified model that handles both "thinking" and "non-thinking" modes. There is no `Qwen3-32B-Base` or `Qwen3-32B-Instruct` on HuggingFace.

The "base" run and the "instruct" run both used `Qwen/Qwen3-32B` — the exact same model and weights. All 18 category shifts are exactly 0.00.

## Setup

- **Model (both runs):** Qwen/Qwen3-32B (unified instruction-tuned, bf16)
- **Axis:** Lu et al. assistant axis (HF: lu-christina/assistant-axis-vectors)
- **Layer:** 32
- **Categories:** 18 (same data as base run, 3514 total examples)
- **Method:** Last-token pooling, project onto axis
- **GPU:** H200 GPU 3, ~348 seconds total

## Results

All shifts = 0.00 (identical model). For reference, the category rankings from the original run:

| Rank | Category | Format | N | Median | Std |
|------|----------|--------|---|--------|-----|
| 1 | Wikipedia Articles | raw_text | 200 | -10.94 | 8.23 |
| 2 | Explanation / Teaching | conversation | 200 | -11.11 | 5.04 |
| 3 | General Assistant Q&A | conversation | 200 | -13.30 | 7.43 |
| 4 | Raw JavaScript Code | raw_text | 200 | -13.40 | 10.34 |
| 5 | Casual Chat | conversation | 200 | -13.78 | 8.99 |
| 6 | How-To Guides | raw_text | 200 | -13.92 | 8.53 |
| 7 | Coding Q&A | conversation | 200 | -14.19 | 3.31 |
| 8 | FineWeb Random (baseline) | raw_text | 200 | -14.55 | 9.64 |
| 9 | Summarization Tasks | conversation | 200 | -15.05 | 4.67 |
| 10 | Raw Python Code | raw_text | 200 | -15.96 | 9.90 |
| 11 | News Articles | raw_text | 200 | -16.68 | 8.78 |
| 12 | Academic / ArXiv | raw_text | 200 | -16.71 | 8.31 |
| 13 | Religious Text | raw_text | 200 | -17.89 | 9.18 |
| 14 | Writing / Email Help | conversation | 200 | -18.90 | 7.13 |
| 15 | Translation Tasks | conversation | 114 | -19.09 | 6.73 |
| 16 | Math Q&A | conversation | 200 | -20.71 | 1.43 |
| 17 | System Prompts | raw_text | 200 | -23.58 | 6.02 |
| 18 | Creative Writing Request | conversation | 200 | -24.34 | 7.02 |

## Interpretation

1. **Null result, but informative:** The original base run's results ALREADY reflected an instruction-tuned model. The finding that "Wikipedia is #1 and system prompts are near last" is the instruction-tuned model's representation, not a pretrained base model's.

2. **The interesting question remains unanswered:** How does instruction tuning reshape category projections? To answer this, we need a true base vs instruct comparison.

3. **Implication for the base run:** The base run's finding that "the assistant axis captures encyclopedic/informative discourse mode" is already a statement about the INSTRUCT model's representation space. This is actually fine for interpretation — it tells us what the instruct model internally associates with "assistant-ness."

## Suggested Follow-Up

To get a genuine base-vs-instruct comparison, options include:

1. **Qwen2.5-32B vs Qwen2.5-32B-Instruct** — Same parameter scale, genuine base/instruct split. The axis (from Qwen3-32B) would serve as a fixed reference frame. Both models need downloading (~64GB each).

2. **Compute a new axis from Qwen2.5-32B** — Use Lu et al. method to extract an axis from the base model, then compare how the same categories project through base vs instruct.

3. **Alternative: chat-template wrapping** — Since Qwen3-32B IS instruction-tuned, wrapping the same text in chat templates (user/assistant turns) might shift projections even with the same model. This tests formatting effects rather than model weight effects.

## Output Files

- Results: `eval_results/axis_category_projection_instruct/category_projections_instruct.json`
- Log: `eval_results/axis_category_projection_instruct/axis_category_instruct_log.txt`
- Plots: `figures/base_vs_instruct_bar.png`, `figures/instruct_shift.png`, `figures/base_vs_instruct_scatter.png`, `figures/instruct_category_boxplot.png`, `figures/rank_change.png`
- Run metadata: `eval_results/axis_category_projection_instruct/run_result.json`

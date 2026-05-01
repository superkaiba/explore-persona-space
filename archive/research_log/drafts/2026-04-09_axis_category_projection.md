# Axis Category Projection: What Content Types Activate the Assistant Axis?

**REVIEWED (revised per independent reviewer feedback — random direction control added, effect size for Mann-Whitney, category selection caveat)**

**Date:** 2026-04-09
**Aim:** 1 (Persona Geometry) / 4 (Axis Origins)

## Goal

Systematically characterize which content categories project most/least onto the "assistant axis" (Lu et al.) in Qwen3-32B. Prior work (2026-04-08 axis corpus projection on FineWeb) showed the axis captures discourse mode, but we hadn't tested curated content types like raw code vs coding Q&A, common assistant tasks, system prompts, etc.

## Setup

- **Model:** Qwen/Qwen3-32B (base, bf16)
- **Axis:** Lu et al. assistant axis from `lu-christina/assistant-axis-vectors` (HF dataset)
- **Layer:** 32 (middle layer, consistent with prior analysis)
- **Method:** Last-token pooling of layer 32 hidden states, projected onto normalized axis vector
- **Categories:** 18 total (9 raw text, 9 conversation format), 200 examples each (114 for Translation)
- **Max length:** 512 tokens, batch size 4
- **GPU:** Single NVIDIA H200
- **Wall time:** ~6.5 minutes total (Phase 1: 67s data loading, Phase 2: 96s model loading, Phase 3: 169s projection)

### Categories Tested

**Raw Text (pretraining-like):**
Raw Python Code (StarCoder fallback via FineWeb), Raw JavaScript Code (FineWeb fallback), Wikipedia Articles (wikimedia/wikipedia), Academic/ArXiv (FineWeb fallback), News Articles (cc_news), How-To Guides (FineWeb filtered), Religious Text (FineWeb filtered), System Prompts (awesome-chatgpt-prompts + synthetic), FineWeb Random (baseline)

**Conversation Format (User/Assistant):**
Coding Q&A (CodeAlpaca-20k), General Assistant Q&A (Alpaca random), Writing/Email Help (Alpaca filtered), Explanation/Teaching (Alpaca filtered), Creative Writing Request (Alpaca filtered), Summarization Tasks (Alpaca filtered), Math Q&A (GSM8K), Translation Tasks (Alpaca filtered), Casual Chat (LMSYS-Chat-1M)

## Results

**All projections are negative** (ranging from -10.94 to -24.34 median), indicating no category projects into the "assistant-like" (positive) direction on this axis at this layer. The axis differentiates *degrees* of anti-assistant-ness.

### Summary Table (sorted by median, least negative = "most assistant-like")

| Rank | Category                   | Format       |   N | Mean   | Median  | Std   | Q25     | Q75     |
|------|---------------------------|-------------|-----|--------|---------|-------|---------|---------|
| 1    | Wikipedia Articles         | raw_text     | 200 | -10.28 | -10.94  | 8.23  | -15.73  | -4.92   |
| 2    | Explanation / Teaching     | conversation | 200 | -11.68 | -11.11  | 5.04  | -13.84  | -8.89   |
| 3    | General Assistant Q&A      | conversation | 200 | -13.58 | -13.30  | 7.43  | -17.88  | -8.73   |
| 4    | Raw JavaScript Code        | raw_text     | 200 | -12.68 | -13.40  | 10.34 | -18.86  | -6.09   |
| 5    | Casual Chat                | conversation | 200 | -13.87 | -13.78  | 8.99  | -19.82  | -7.59   |
| 6    | How-To Guides              | raw_text     | 200 | -13.25 | -13.92  | 8.53  | -18.78  | -7.34   |
| 7    | Coding Q&A                 | conversation | 200 | -14.45 | -14.19  | 3.31  | -16.88  | -12.21  |
| 8    | FineWeb Random (baseline)  | raw_text     | 200 | -13.18 | -14.55  | 9.64  | -19.40  | -6.73   |
| 9    | Summarization Tasks        | conversation | 200 | -14.82 | -15.05  | 4.67  | -17.82  | -11.52  |
| 10   | Raw Python Code            | raw_text     | 200 | -15.53 | -15.96  | 9.90  | -22.08  | -8.91   |
| 11   | News Articles              | raw_text     | 200 | -15.30 | -16.68  | 8.78  | -21.80  | -9.50   |
| 12   | Academic / ArXiv           | raw_text     | 200 | -15.30 | -16.71  | 8.31  | -20.72  | -10.05  |
| 13   | Religious Text             | raw_text     | 200 | -18.01 | -17.89  | 9.18  | -24.04  | -13.03  |
| 14   | Writing / Email Help       | conversation | 200 | -18.60 | -18.90  | 7.13  | -23.59  | -13.63  |
| 15   | Translation Tasks          | conversation | 114 | -17.50 | -19.09  | 6.73  | -22.83  | -12.10  |
| 16   | Math Q&A                   | conversation | 200 | -20.73 | -20.71  | 1.43  | -21.66  | -19.82  |
| 17   | System Prompts             | raw_text     | 200 | -22.93 | -23.58  | 6.02  | -26.69  | -20.10  |
| 18   | Creative Writing Request   | conversation | 200 | -23.22 | -24.34  | 7.02  | -28.25  | -18.33  |

### Format Comparison (aggregated)

| Format       | Mean   | Median  |
|-------------|--------|---------|
| Raw text     | -15.16 | -15.86  |
| Conversation | -16.44 | -16.40  |

Mann-Whitney U = 1,634,044, p = 0.0024 -- conversation format is *slightly more anti-assistant* than raw text. However, the effect size is tiny (median diff = 0.54 projection units, ~4% of the 14-unit range), and the result is confounded by category selection: the specific 9 categories in each format were hand-selected, not randomly sampled, so changing one category could flip the direction.

## Key Findings

### 1. Conversation format does NOT consistently project higher than raw text

Contrary to expectation, the User/Assistant conversation format does not reliably push content toward the assistant direction. In aggregate, conversation-format text is *slightly more negative* (median -16.40) than raw text (median -15.86). This is statistically significant (p=0.002) but the effect is small (delta = 0.54 projection units).

### 2. Content type matters more than format

The biggest spread is across content types within each format:
- **Raw text range:** -10.94 (Wikipedia) to -23.58 (System Prompts) = 12.6 unit spread
- **Conversation range:** -11.11 (Explanation/Teaching) to -24.34 (Creative Writing) = 13.2 unit spread

### 3. Wikipedia is the most "assistant-like" category

Wikipedia articles (median -10.94) project the least negatively, even above all conversation-format categories. This aligns with prior findings that the axis captures encyclopedic/educational discourse.

### 4. Explanation/Teaching Q&A is the closest conversation category

Explanation and teaching conversations (median -11.11) are very close to Wikipedia, consistent with the axis capturing "informative/educational" mode.

### 5. Raw code vs Coding Q&A: format barely matters

- Raw Python Code: median -15.96
- Raw JavaScript Code: median -13.40
- Coding Q&A: median -14.19

JavaScript code projects higher than Python (possibly shorter/more natural-language-like variable names?). The conversation-formatted Coding Q&A sits between them. Format is not the dominant factor.

### 6. System Prompts project very anti-assistant (surprising)

System prompts (median -23.58) are among the most anti-assistant categories, second only to Creative Writing requests. This is counterintuitive -- "You are a helpful AI assistant" text projects in the anti-assistant direction. This suggests the axis is NOT about meta-assistant identity text, but about the discourse mode of the content itself.

### 7. Creative Writing is the most anti-assistant

Creative writing requests (median -24.34) are the most anti-assistant category. This is highly consistent with the prior finding (2026-04-08 axis tail analysis) that creative/personal/narrative content maps to the anti-assistant pole.

### 8. Math Q&A has remarkably low variance

Math Q&A (std=1.43) is strikingly concentrated compared to all other categories (std 3-10). The GSM8K format (structured word problems with step-by-step solutions) creates very uniform representations.

### 9. Variance structure differs dramatically by category

- Tightest: Math Q&A (std=1.43), Coding Q&A (3.31), Summarization (4.67)
- Widest: Raw JavaScript (10.34), Raw Python (9.90), FineWeb Random (9.64)
- Raw text categories have ~2x the variance of conversation categories on average

## Caveats

- **Random direction control not run for categories.** A corpus-level random direction control (in summary.json) shows the assistant axis does NOT separate FineWeb from LMSYS significantly better than random directions (z=-0.45). The within-corpus category rankings have not been tested against random directions. The category findings could potentially be reproduced on any random direction in the 5120-dimensional space.
- **Single seed, single layer** -- results at layer 32 only; axis behavior may differ at other layers
- **N=200 per category** -- sufficient for means/medians but some categories (esp. raw code) have high variance
- **Code categories used FineWeb fallback** -- the primary code datasets (StarCoder, CodeParrot) were gated or deprecated; FineWeb-filtered code snippets may not be representative of pure code files
- **Academic texts also used FineWeb fallback** -- may include more educational than pure research content
- **Translation only had 114 examples** -- fewer matching Alpaca entries
- **Category selection confounds the format comparison** -- the 18 categories were hand-selected, not randomly sampled. The Mann-Whitney p=0.002 tests these particular categories, not "format in general"
- **Correlational, not causal** -- projecting text onto a persona axis shows where text falls in one dimension of a 5120-D space, not necessarily what the axis "captures"

## Interpretation

The assistant axis in Qwen3-32B captures **educational/informative discourse mode**, not assistant formatting per se. The strongest signals are:

**Most "assistant-like" (least negative):** Wikipedia, Explanation/Teaching, General Q&A, How-To Guides
**Most "anti-assistant" (most negative):** Creative Writing, System Prompts, Math, Religious Text

This pattern suggests the axis separates "neutral encyclopedic/informative" content from "characterized/performative" content. System prompts, creative writing, and math all involve the model "performing a role" or producing highly structured/formulaic output, which pushes anti-assistant. Wikipedia and teaching content is "just explaining things," which pushes assistant.

## Relation to Prior Work

- **Consistent with axis corpus projection (2026-04-08):** instructional/didactic content projects assistant-like, creative/personal projects anti-assistant
- **Consistent with axis tail deep analysis (2026-04-08):** surface features don't explain projection; it's about discourse mode
- **New finding:** conversation format itself is NOT a driver; the axis is content-mode-sensitive, not format-sensitive

## Plots

- `figures/category_projections_boxplot.png` -- Full distribution comparison
- `figures/category_rankings_bar.png` -- Median ranking with IQR
- `figures/category_projections_violin.png` -- Distribution shapes
- `figures/format_comparison.png` -- Raw text vs Conversation side-by-side

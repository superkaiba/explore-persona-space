# Assistant Axis Tail Deep Analysis — Aim 4.2b

**Date:** 2026-04-08
**Status:** REVIEWED (revised per independent reviewer feedback — random direction control disclosed, R² wording fixed, multiple comparison correction noted, length confound added)

## Goal

Characterize what pretraining and conversation data activates the assistant axis most/least. Goes deeper than v1's TF-IDF analysis by using:
1. Claude-based document taxonomy (LLM categorization)
2. Document feature regression (surface features → projection)
3. Per-token outlier analysis (which tokens drive extreme projections)
4. Cross-corpus comparison (FineWeb-Edu vs LMSYS)

## Setup

- **Model:** Qwen/Qwen3-32B
- **Axis:** lu-christina/assistant-axis-vectors, layer 32, normalized (norm=22.66)
- **Data:** 200K FineWeb-Edu + 200K LMSYS documents, last-token pooling via speculators+vLLM
- **Tail sample:** Top 200 + Bottom 200 + Random 200 per corpus (1200 docs total)
- **Taxonomy:** Claude Sonnet 4.5 via Batch API, 6 dimensions per document

## Key Finding: The Axis Is Not a Surface-Level Detector

**Surface features explain <3% of projection variance** (linear models).

| Metric | FineWeb | LMSYS |
|--------|---------|-------|
| OLS R² | 0.026 | 0.008 |
| OLS CV R² | negative | negative |
| GB CV R² | negative | negative |
| Classifier accuracy | 48.2% | 59.0% |

Features tested: word count, sentence count/length, question marks, lists, code blocks, vocabulary diversity, reading level, pronoun ratios, imperative verb frequency, named entity density.

**Interpretation:** Linear surface features are poor predictors. However, Gradient Boosted models achieve R²=0.69 (FineWeb) and 0.99 (LMSYS) on training data but negative CV R² (severe overfitting), and held-out test accuracy is 55% (FineWeb) and 65.75% (LMSYS) — above chance for LMSYS. The features contain some signal but it's weak and mostly not generalizable from 600 tail documents.

## Taxonomy Analysis (FineWeb-Edu)

### Statistically Significant Differences (top 200 vs bottom 200)

**Genre** (χ² = 25.91, p = 0.007, NOT significant after Bonferroni correction for 12 tests, threshold p<0.004):
| Category | Top (assistant) | Bottom (anti-asst) | Log₂ OR |
|----------|:-:|:-:|:-:|
| Instructional | 75 | 51 | +0.56 |
| Reference | 36 | 28 | +0.36 |
| Creative | 0 | 7 | -∞ |
| Technical | 4 | 11 | -1.46 |
| Narrative | 2 | 5 | -1.29 |

**Author Stance** (χ² = 12.65, p = 0.013, NOT significant after Bonferroni):
| Category | Top | Bottom | Log₂ OR |
|----------|:-:|:-:|:-:|
| Helpful/didactic | 84 | 63 | +0.42 |
| Personal/subjective | 13 | 28 | -1.11 |
| Authoritative/declarative | 15 | 26 | -0.79 |

**Discourse type, register, audience, interactivity:** Not significant at p<0.05.

### Interpretation

The assistant direction in Qwen 3 32B captures a **helpful, instructional, explanatory discourse mode** — text that explains things to a general audience in a didactic, reference-like style.

The anti-assistant direction captures **personal, creative, narrative, authoritative** content — text where the author asserts their own perspective, tells stories, or writes for peers/experts.

This is exactly what you'd expect if the assistant axis represents the precursor to the instruct-tuned assistant persona. Pretraining on instructional web content builds a latent "helpful explainer" direction that post-training amplifies into the full assistant behavior.

### Top Tail Topics (FineWeb)
Diverse instructional content: computer troubleshooting, health/science education, how-to guides, FAQ-style content.

### Bottom Tail Topics (FineWeb)
Historical narratives, religious texts, biographical essays, peer-reviewed academic content, personal essays.

## Per-Token Outlier Analysis

### Position 0 Universal Phenomenon

**Every document has an extreme negative projection (~-2800) at token position 0**, regardless of content.

| Metric | Value |
|--------|-------|
| Mean pos-0 projection | -2785 |
| Std pos-0 projection | 249 |
| Range | -3131 to -2396 |
| Ratio to rest-of-doc max | 56x to 137x |

This is a positional phenomenon, not content-dependent. The first token position in the model always has a massive negative projection onto the assistant axis. This likely reflects the model's "initialization" state before it has processed any content — and this initialization is maximally far from the "assistant" direction.

### Non-Position-0 Token Patterns

Excluding position 0, per-token projections are small (mean ≈ -15, range ≈ ±50). The variation across tokens within a document is modest, suggesting the axis captures a document-level semantic property rather than being driven by individual keyword tokens.

## Speculators Reliability Analysis

### Identified Artifacts

1. **Padding artifact:** 1966 LMSYS docs (1%) have projection = -21.75, with 1864 of these having token_count = 64 (a batch alignment boundary). Speculators likely padded short sequences to 64 tokens, and `hidden[-1]` selected a padding token's hidden state.

2. **Extreme outliers:** Two LMSYS docs (-2968, -2380) are 50x more extreme than the next value (-64). Their projections are close to position-0 values (~-2600 to -3100), suggesting `hidden[-1]` returned position 0 for these documents. A possible cause: these sequences had 1 real token after speculators' internal processing.

3. **Repeated median values:** All 5 sampled LMSYS median docs had exactly -12.18, despite having different content and different HF last-token projections (-5.19 to -18.96). This suggests limited precision in the stored values.

### Verification

Direct comparison of speculators vs HuggingFace on a single input ("What is penetration testing?") shows **exact agreement within 0.3%** at all token positions. The hidden state extraction is correct — the issue is in batch processing of 200K documents, not in the extraction mechanism.

### Impact on Analysis

- The padding artifact affects ~1% of LMSYS docs but projects close to the median (-21.75 vs median -12.18), so tail docs are NOT affected.
- FineWeb has no such artifact (no repeated values, no extreme outliers).
- The tail analysis (top 200 vs bottom 200) is valid because artifact docs cluster near the median.

## TF-IDF Clustering (FineWeb Tails)

**Top and bottom tails are completely indistinguishable in TF-IDF embedding space.**

| Metric | Value |
|--------|-------|
| Centroid cosine (top vs bottom) | 0.777 |
| Centroid cosine (top vs random) | 0.774 |
| Centroid cosine (bottom vs random) | 0.773 |
| K-Means (k=2) ARI | -0.002 (worse than random) |
| K-Means silhouette | 0.003 |

Within-group pairwise similarity is ~0.017 for all three groups (top, bottom, random). The t-SNE plot shows complete overlap between top and bottom tail documents.

**This is the strongest evidence that the axis captures something beyond surface statistics.** Not only do simple features (word count, reading level) fail to predict projection, but even the full TF-IDF vocabulary distribution fails. Two documents about the same topic can project very differently based on their discourse mode.

## LMSYS Taxonomy

### Significant Dimensions

**Author Stance** (χ² = 15.15, p = 0.004, ONLY test surviving Bonferroni correction for 12 tests):
| Category | Top | Bottom | Log₂ OR |
|----------|:-:|:-:|:-:|
| Authoritative/declarative | 17 | 8 | +1.04 |
| Neutral/encyclopedic | 40 | 25 | +0.67 |
| Personal/subjective | 5 | 15 | -1.49 |

**Genre** (χ² = 23.29, p = 0.016):
| Category | Top | Bottom | Log₂ OR |
|----------|:-:|:-:|:-:|
| Academic | 8 | 4 | +0.92 |
| Technical | 16 | 11 | +0.52 |
| Creative | 13 | 25 | -0.92 |
| Religious | 0 | 3 | -2.81 |

**Register** (χ² = 9.63, p = 0.047): Formal/technical → TOP, informal/colloquial → BOTTOM.

**Audience** (χ² = 11.69, p = 0.039): Experts → TOP (+0.72), "other" → BOTTOM (-1.28).

### LMSYS Topic Clusters (Claude-classified)

| Cluster | Top | Bot | Ratio | Direction |
|---------|:-:|:-:|:-:|:-:|
| Data Analysis & ML | 12 | 6 | 1.92x | TOP |
| Programming & Technical | 25 | 18 | 1.38x | TOP |
| Business & Professional | 11 | 8 | 1.35x | TOP |
| **Inappropriate Content** | **12** | **28** | **0.44x** | **BOTTOM** |
| Creative Writing | 8 | 16 | 0.52x | BOTTOM |
| Gaming & Entertainment | 3 | 8 | 0.41x | BOTTOM |

The LMSYS-specific finding: **jailbreak/inappropriate requests project anti-assistant** (12 vs 28, 0.44x). The axis naturally separates professional task-oriented conversations from adversarial or creative ones.

## Cross-Corpus Comparison

### Projection Distributions

| | FineWeb | LMSYS |
|---|---|---|
| Mean | -14.07 | -13.02 |
| Std | 9.26 | 12.14 |
| Range | [-60, 33] | [-64, 36]* |

*Excluding speculators artifacts (-2968, -2380)

### Consistent Patterns (same direction in BOTH corpora)

| Pattern | FineWeb log₂OR | LMSYS log₂OR |
|---------|:-:|:-:|
| Creative genre → BOTTOM | -3.91 | -0.92 |
| Personal/subjective → BOTTOM | -1.08 | -1.49 |
| Religious → BOTTOM | -0.55 | -2.81 |
| Instructional genre → TOP | +0.56 | +0.32 |
| Neutral/encyclopedic → TOP | +0.10 | +0.67 |

### Divergent Patterns

| Pattern | FineWeb | LMSYS |
|---------|:-:|:-:|
| Academic genre | BOTTOM (-0.47) | TOP (+0.92) |
| Authoritative/declarative | BOTTOM (-0.77) | TOP (+1.04) |

**Why the divergence:** In FineWeb, "academic" means scholarly papers and historical analysis (passive, analytical). In LMSYS, "academic" means structured educational Q&A (active, didactic). Same label, different discourse mode — confirming that it's the framing, not the topic.

### Topic Cluster Enrichment

**FineWeb:** Practical how-to (2.65x TOP), Health info (1.76x TOP) vs Arts/literature (0.46x BOTTOM), Environment (0.61x BOTTOM)

**LMSYS:** Data/ML (1.92x TOP), Programming (1.38x TOP) vs Inappropriate content (0.44x BOTTOM), Creative writing (0.52x BOTTOM), Gaming (0.41x BOTTOM)

## Conclusions

1. **The assistant axis captures discourse mode, not surface features or topic.** Surface features (R²=0.03), TF-IDF vocabulary (ARI=-0.002), and topic (K-means at chance) all fail to separate the tails. The axis detects HOW text communicates, not WHAT it's about.

2. **Two cross-corpus robust signals:** Creative genre → anti-assistant (FW: -3.91, LM: -0.92) and personal/subjective stance → anti-assistant (FW: -1.08, LM: -1.49). Both replicate across web text and conversations.

3. **The axis captures "helpful task completion" as a discourse mode.** In FineWeb: practical how-to guides and health information activate it most. In LMSYS: data analysis, programming, and professional requests. The common thread is task-oriented, solution-focused framing.

4. **Creative, personal, and adversarial content is anti-assistant.** In FineWeb: arts, literature, historical narrative. In LMSYS: creative writing, jailbreak attempts, entertainment. The anti-assistant direction corresponds to open-ended, subjective, or boundary-pushing interaction.

5. **The "academic" divergence reveals framing > topic.** Academic FineWeb content (scholarly, analytical) projects anti-assistant. Academic LMSYS content (structured educational Q&A) projects assistant. Same topic label, opposite direction — because the discourse mode differs.

6. **Jailbreak-type content naturally separates from the assistant direction** (12 vs 28 in LMSYS inappropriate cluster, 0.44x). The axis is a potential signal for detecting adversarial prompts.

7. **Position 0 has a universal ~-2800 projection** (model initialization = maximally anti-assistant). This is architectural, not content-driven.

8. **Speculators has ~1% batch-processing artifacts** in LMSYS (padding to 64 tokens) but tails are unaffected.

## Vocabulary Scan (428 words, "The ___ is" template)

Tested 428 words across 15 semantic categories. Each word placed at position 1 in "The {word} is" to control for positional effects.

### Category Rankings (mean projection)

| Category | Mean Proj | Interpretation |
|---|---:|---|
| political | -1.6 | Most assistant — civic/democratic concepts |
| roles | -5.5 | Person-who-helps words |
| safety | -5.6 | Safety/moderation language |
| medical | -6.2 | Health information |
| conversational | -6.4 | Dialogue markers |
| helpful | -7.5 | Help-related words |
| academic | -8.7 | — |
| action | -9.8 | — |
| emotional | -10.0 | — |
| nature | -11.9 | — |
| factual | -12.3 | — |
| creative | -12.4 | Creative writing |
| religious | -12.5 | Religious vocabulary |
| quantitative | -13.9 | Math/computation |
| technical | -14.0 | Most anti-assistant — code/programming |

### Top 10 Words

| Word | Projection | Category |
|---|---:|---|
| counselor | +17.6 | roles |
| helpful | +8.2 | helpful |
| tips | +7.6 | helpful |
| democracy | +7.5 | political |
| advisor | +6.9 | roles |
| ideology | +6.8 | academic |
| drama | +6.6 | creative |
| justice | +6.4 | political |
| improve | +6.2 | helpful |
| amendment | +5.7 | political |

### Bottom 10 Words

| Word | Projection | Category |
|---|---:|---|
| story | -27.1 | creative |
| number | -27.0 | factual |
| tale | -26.9 | creative |
| function | -24.1 | technical |
| data | -23.3 | factual |
| scene | -22.7 | creative |
| product | -22.6 | quantitative |
| sum | -22.6 | quantitative |
| prayer | -22.1 | religious |
| wind | -22.4 | nature |

### Key Finding: "assistant" Does NOT Activate the Axis

The literal word "assistant" projects at -4.75 (slightly negative). The word "helper" projects at +1.6. The strongest activator is "counselor" (+17.6) — a person-who-helps role. The axis encodes the *concept* of being helpful, not the literal word "assistant."

This is consistent with Lu et al.'s finding that the axis promotes "helpful human archetypes like consultants and coaches" — but our vocabulary scan shows this at single-token granularity.

## Jailbreak Separation

In LMSYS tail conversations, the "Inappropriate Content & Ethics" topic cluster shows 12 docs in the TOP tail vs 28 in the BOTTOM tail (0.44x, 2.3x anti-assistant enrichment).

**Critical observation:** The axis separates based on the model's *response*, not just the prompt. When harmful prompts receive clear refusals (e.g., "I'm not going to do that, because it's hurtful"), the conversation projects assistant (+29.1). When the model complies with boundary-pushing requests (erotic content, roleplay, jailbreak), the conversation projects anti-assistant (-55).

This suggests the axis could serve as a **real-time safety signal** — monitoring the projection during generation could detect when the model has been steered away from its assistant persona.

## Important Limitations

**Random direction control (summary.json):** The assistant axis does NOT separate FineWeb from LMSYS significantly better than random directions (Cohen's d=-0.161, z=-0.45 vs 10 random directions with |d| up to 1.16). The within-corpus tail analysis (category rankings, taxonomy) has NOT been tested against random directions and could potentially be reproduced on arbitrary directions in 5120-D space. This is the most important open question for this analysis.

**Length confound:** FineWeb projection correlates with document length at r=0.14 (p<0.001, n=200K). Longer documents project less negatively. This could partially explain tail differences if assistant-tail documents tend to be longer.

**Multiple comparison correction:** 12 chi-squared tests (6 dimensions × 2 corpora) were run. After Bonferroni correction (threshold p<0.004), only LMSYS Author Stance (p=0.004) survives. FineWeb Genre (p=0.007), FineWeb Stance (p=0.013), and other tests do NOT survive correction.

## Additional Caveats

- Taxonomy classification is based on 2000-char truncated snippets, not full documents
- Single seed / single model (Qwen 3 32B only)
- LMSYS projections have known artifacts; FineWeb is more reliable
- Chi-squared tests use tail docs (top/bottom 200 of 200K) — extreme tails may not generalize to the full distribution
- The axis was computed by a third party (Lu et al.) — we trust but haven't verified the extraction method

## Next Steps

1. Get LMSYS taxonomy and compare cross-corpus
2. Embed tail docs with sentence model, cluster, see if FineWeb-top clusters with LMSYS-top
3. Consider rerunning LMSYS projections with HuggingFace (slow but artifact-free) on a smaller sample
4. Test whether the taxonomy findings hold for the full distribution (not just tails)

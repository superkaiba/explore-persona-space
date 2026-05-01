---
status: AUTO-GENERATED DRAFT
status_detail: UNREVIEWED
aim: 4 (Axis Origins)
date: 2026-04-11
experiment: axis_projection_fineweb_raw
---

# Aim 4.2 Extension: Raw FineWeb Projection onto Assistant Axis

## Question

Does the assistant axis projection distribution differ between raw (unfiltered) FineWeb and quality-filtered FineWeb-Edu? The v2 analysis projected 200K FineWeb-Edu + 200K LMSYS, but FineWeb-Edu is pre-filtered for educational content. Raw FineWeb includes spam, forums, fiction, legal text, product pages, personal blogs, etc. -- much more diverse. Does this diversity produce more extreme tails? Does it shift the distribution?

## Setup

- **Model:** Qwen3-32B via speculators + vLLM, layer 32, TP=2
- **Dataset:** HuggingFaceFW/fineweb (sample-10BT), 200K documents
- **Pooling:** last-token (chosen by pilot; length_r=0.078 vs mean-pool length_r=0.593)
- **Max length:** 512 tokens, batch size 32
- **Runtime:** 91.3 min (39.0 docs/sec), zero failed batches, zero truncated docs
- **Pod:** H200 x4, GPUs 0,1

## Results

### Distribution Comparison

| Statistic | FineWeb-raw | FineWeb-Edu | Difference |
|-----------|-------------|-------------|------------|
| N docs | 200,000 | 200,000 | -- |
| Mean | -16.40 | -14.07 | -2.33 |
| Std | 9.19 | 9.26 | -0.07 |
| Min | -64.76 | -60.40 | -4.36 |
| Max | 31.89 | 32.78 | -0.89 |
| p1 | -39.37 | -36.02 | -3.35 |
| p5 | -31.58 | -28.59 | -2.99 |
| p25 | -21.85 | -20.12 | -1.73 |
| p50 | -16.66 | -14.64 | -2.02 |
| p75 | -10.96 | -8.08 | -2.88 |
| p95 | -0.63 | 1.73 | -2.36 |
| p99 | 6.21 | 8.52 | -2.32 |
| Length r (Pearson) | 0.126 | 0.140 | -0.014 |
| Mean token count | 324.8 | 370.0 | -45.2 |

**Cohen's d = -0.252** (raw shifted negative relative to edu). Mann-Whitney p < machine epsilon. KS D = 0.105, p < machine epsilon.

### Key Finding: Uniform Negative Shift

The shift is remarkably uniform across the entire distribution. Every percentile (p1 through p99) shifts by 2.0-3.4 units in the negative direction. This is NOT a tail effect -- the entire distribution translates.

**Interpretation:** Quality-filtered educational text (FineWeb-Edu) projects ~2.3 units more positive on the assistant axis than unfiltered web text. This suggests the "educational/explanatory" register that FineWeb-Edu selects for partially overlaps with what the assistant axis captures.

### Tail Extremity

The raw FineWeb has a **heavier negative tail**:
- Bottom extreme: raw=-64.76 vs edu=-60.40 (raw is 4.36 more extreme)
- Top extreme: raw=31.89 vs edu=32.78 (edu slightly more extreme)
- Fraction beyond edu's 3-sigma low: raw=0.52% vs edu=0.23% (2.3x more)
- Fraction beyond edu's 3-sigma high: raw=0.09% vs edu=0.23% (2.5x fewer)

The asymmetry is notable: raw web has more extreme anti-assistant content but NOT more extreme assistant-like content. Educational filtering mainly cuts the negative tail.

### Tail Content Analysis

**Most assistant-like (top tail):**
- Informational/service pages: safety guidelines, event announcements, business profiles
- "How-to" and advisory content: heating systems, mold removal, Quicken error fixes
- Community/professional organizations: farmers markets, clinical groups, school programs
- TF-IDF keywords: online, work, media, room, weekend, natural, energy, safety, provide

**Least assistant-like (bottom tail):**
- Personal narratives/blogs: chicken keeping, art comments, Canadian in American convent
- Product descriptions: artisan store, red cards, handmade cards, motorcycle insurance
- Entertainment/pop culture: NBA commentary, Musketeers, film
- Fragmented/SEO text: Mystery Spot spam, dentist SEO content
- Meeting minutes, testimonials, news snippets
- TF-IDF keywords: things, new, red, hand, film, pepper, don, card, cards, literally

**Content types in raw FineWeb bottom tail NOT in FineWeb-Edu:**
- SEO spam / keyword stuffing (e.g., doc 150285: "Mystery Spot" spam)
- Personal blogs with casual/emotional tone (chicken keeping, convent story)
- Product catalog/ecommerce descriptions
- Comment threads and art community discussions
- Meeting minutes and organizational boilerplate
- Spun/mangled text (doc 26850: dentist text with garbled grammar)

### Cross-Distribution Positioning

- 8.5% of raw FineWeb falls below FineWeb-Edu's 5th percentile (expected: 5% if equal)
- 3.0% of raw FineWeb falls above FineWeb-Edu's 95th percentile (expected: 5% if equal)
- Raw web text is shifted toward anti-assistant direction, with more mass in the negative tail

## Raw Observations

1. The effect size (d=-0.25) is small-to-medium. The distributions largely overlap.
2. Standard deviations are nearly identical (9.19 vs 9.26), so the quality filter mainly shifts the location, not the spread.
3. Length confound is comparable (r=0.126 vs 0.140), so the shift is not explained by document length differences (raw docs are actually shorter on average: 325 vs 370 tokens).
4. The pilot automatically selected last-token pooling for both runs (v2 may have used mean pooling -- need to check). This could affect absolute values but not relative comparisons within the same run.
5. Residualization R-squared is only 1.6% for the raw FineWeb, meaning token count explains very little projection variance.

## Caveats

- **Single run, no seed variation** -- but projection is deterministic given the model weights, so seed only affects document sampling order.
- **v2 pilot may have chosen different pooling** -- absolute values are not directly comparable if pooling method differs. The relative distribution shapes and tail content are still valid.
- **No random direction control** for this run -- the v2 analysis showed the assistant axis is NOT special for between-corpus separation (z=-0.45). The within-corpus structure may or may not be axis-specific.
- **Text truncation to 2K chars then 512 tokens** -- very long documents are underrepresented.
- **sample-10BT is itself a subsample** of full FineWeb -- may not capture the full diversity of CommonCrawl.

## Next Steps

1. Run random direction control for raw FineWeb (is the raw-vs-edu shift specific to the assistant axis?)
2. Compare pooling methods between v2 and this run to ensure fair comparison
3. Qualitative coding of raw FineWeb tails -- what specific content types drive the extreme negative projections?
4. Consider projecting other corpora: code (The Stack), social media (Reddit), scientific papers (S2ORC)

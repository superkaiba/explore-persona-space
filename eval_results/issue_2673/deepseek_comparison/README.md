# Qwen context cosine versus published DeepSeek uptake

The existing Qwen capture supports a four-character **approximate overlap diagnostic**. It does not contain HHH, Fred, or help-seeker vectors. A new eight-description panel is prepared for both models; neither new model arm has run yet.

At Qwen block63, similarity from its default assistant to four published persona prompts correlates with the paper’s DeepSeek HHH uptake of the corresponding non-helpful character’s tracer: raw Pearson r=0.822, Spearman rho=0.80; uncentered whitened r=0.507, rho=0.40. Only four aggregate pairs contribute. These are descriptive associations across different models, prompt realizations and evaluation contexts.

| Character | Qwen raw cosine | Qwen whitened cosine | DeepSeek HHH other-tracer uptake |
|---|---:|---:|---:|
| dismissive | 0.6923 | -0.00206 | 7.3% |
| sarcastic | 0.5833 | 0.00157 | 4.3% |
| saboteur | 0.9039 | 0.01290 | 11.0% |
| peer | 0.7999 | 0.01047 | 5.7% |

All predeclared Qwen depths:

| Block | Raw r | Raw rho | Whitened r | Whitened rho |
|---:|---:|---:|---:|---:|
| 15 | -0.155 | -0.60 | -0.407 | -0.60 |
| 31 | 0.150 | 0.00 | -0.115 | 0.00 |
| 47 | 0.296 | 0.00 | -0.025 | 0.00 |
| 63 | 0.822 | 0.80 | 0.507 | 0.40 |

## Published DeepSeek rates

Approximate means digitized from the [original Figure26](https://arxiv.org/html/2609.10883v1/images/selectivity/fxbc_bloom_deepseek_base_grid.png). Each bar has a conservative pixel-resolution bound of ±0.34 percentage points, separate from sampling uncertainty. The two rates are not complementary.

| Opposing character | HHH: helpful tracer | HHH: other tracer | Fred: helpful tracer | Fred: other tracer |
|---|---:|---:|---:|---:|
| dismissive | 43.6% | 7.3% | 13.0% | 58.7% |
| sarcastic | 60.3% | 4.3% | 8.0% | 40.4% |
| saboteur | 54.3% | 11.0% | 26.9% | 29.0% |
| peer | 34.7% | 5.7% | 27.6% | 6.6% |
| help_seeker | 28.5% | 3.7% | 26.5% | 3.4% |

## Limits and next comparison

The source experiment measures multi-turn, triggered Bloom behavior after story fine-tuning of DeepSeek-V3.1 Base. Our current Qwen vectors use a generic question battery before story fine-tuning; default Qwen is only a proxy for HHH. The old four persona system prompts also differ from the source story-character specifications. Whitening uses the uncentered second moment of the original same-bank rows and is rank deficient. No p-values, fresh behavioral outcomes, or held-out predictive accuracy are claimed.

The planned matched-description panel uses eight distinct texts: the published HHH/Fred preambles and six original role/disposition specifications. Fred’s demonstrations are unreleased, so both evaluation personas use descriptions only. Both models will retain native input formats. Comparisons will report all ten paired similarity/uptake contrasts, stratified by HHH/Fred, raw and uncentered whitened cosine, and both question-half whitening checks. No self-cosine enters that panel.

Reproduce the existing-overlap calculation with `uv run python scripts/story_persona_deepseek_published.py`; it verifies the source raster checksum before digitization. Numerical inputs and source hashes are in `published_rates_and_overlap.json`. The independent review uses a separate full-image morphological border detector and recomputes every correlation.

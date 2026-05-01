# Cross-Model Assistant Axis Comparison (Aim 4.6)

**Date:** 2026-04-08
**Status:** APPROVED
**Significance:** Low — axis structure is similar across models but we can't say much beyond that since directions can't be compared (different hidden dims).
**Main takeaway:** Axis norm profiles are correlated across model families, but this is a weak structural similarity metric only.

## Goal

Compare the pre-computed assistant axes across Gemma 2 27B, Qwen 3 32B, and Llama 3.3 70B to assess axis universality.

## Setup

Pre-computed axes from lu-christina/assistant-axis-vectors:
- Gemma 2 27B: 46 layers × 4608 hidden dim
- Qwen 3 32B: 64 layers × 5120 hidden dim
- Llama 3.3 70B: 80 layers × 8192 hidden dim

Cannot directly compare axis vectors (different dims). Instead compare structural properties.

## Results

### Norm profiles are correlated

| Pair | Norm profile Pearson r |
|------|----------------------|
| Gemma 2 27B ↔ Llama 3.3 70B | **0.971** |
| Gemma 2 27B ↔ Qwen 3 32B | **0.914** |
| Qwen 3 32B ↔ Llama 3.3 70B | **0.833** |

All three show generally increasing norm with depth, but NOT strictly monotonically — there are local dips.

### Axis direction rotates across depth

| Model | Mean adjacent-layer cosine | Quartile cosine | L0↔Lfinal cosine |
|-------|--------------------------|-----------------|-------------------|
| Gemma 2 27B | 0.951 | 0.187 (L11↔L34) | **-0.264** |
| Qwen 3 32B | 0.961 | 0.482 (L16↔L48) | **0.152** |
| Llama 3.3 70B | 0.952 | 0.417 (L20↔L60) | **-0.000** |

Adjacent layers are similar (~0.95 cosine) but the axis rotates substantially across the network. L0↔Lfinal cosines are near zero or negative — early and late layers encode nearly orthogonal or anti-correlated directions. The "assistant axis" is not a single fixed direction — it evolves substantially through the transformer.

## Interpretation

1. **Axis norm structure is similar across models.** All three models (different architectures, different training data, different scales) produce axes with correlated norm profile shapes. However, norm correlation is a weak similarity metric — two very different directions could have similar norms. We can't compare directions across models (different hidden dims).

2. **Norm generally increases with depth but not strictly monotonically.** The overall trend is strongly upward, with the strongest signal at deep layers. Consistent with representation engineering literature.

3. **Direction evolves dramatically across layers.** L0↔Lfinal cosines of -0.26, 0.15, and 0.00 mean early and late layers encode nearly orthogonal or anti-correlated directions. The "assistant axis" at layer 10 is a fundamentally different direction than at layer 60.

4. **Gemma and Llama are most similar** (r=0.971), despite Llama being 2.6x larger. Qwen is the outlier (r=0.833 with Llama), possibly due to architectural differences or different post-training procedures.

## Caveats

- Cannot compare axis directions across models (different hidden dims)
- Norm profile comparison is a weak similarity metric — two very different directions could have similar norms
- Need role-loading correlation (Aim 4 original plan) for stronger cross-model validation
- Only 3 models — cannot draw statistical conclusions about universality

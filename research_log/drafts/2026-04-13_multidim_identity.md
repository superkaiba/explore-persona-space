# Aim 1.5: Is Persona Identity Multi-Dimensional?

**Date:** 2026-04-13
**Status:** Draft (pending review)
**Data:** `eval_results/aim1_5_multidim_identity/multidim_identity_test.json`
**Script:** `scripts/test_multidim_identity.py`

## Goal

Test whether persona identity in activation space is genuinely >1D (not fully captured by centroids), or whether the 8-12D manifolds from Aim 1.2 merely capture behavioral variation while identity itself is 1D.

## Background

- Aim 1.2 found 8-12D manifolds per persona (Layer 22, Gemma 2 27B-IT, 49 personas × 1200 inputs).
- Initial kNN on centroid-subtracted residuals: 18.5% (9× chance) — but confounded by different noise levels per persona.
- Chen et al. (persona vectors) treat personas as 1D (single centroid per persona pair).
- This experiment isolates genuine multi-D identity from distributional artifacts.

## Method

Three complementary tests on centroid-subtracted residuals (58800 samples, 100-D PCA from Layer 22):

1. **Per-persona whitening + kNN**: Regularized eigendecomposition per persona (ε = 1e-4 × max eigenvalue), whitening equalizes ALL second-order structure. If kNN still classifies → signal is beyond covariance.
2. **Question-paired direction cosine**: Unit-normalize mean residual per (persona, question). Compute pairwise cosine averaged over 240 questions. 1000-permutation null tests whether variance of pairwise cosines exceeds chance.
3. **Grassmann distance**: Per-persona PCA at k=5,10,20. Pairwise subspace angular distance vs 100-permutation null.

Pre-specified thresholds: Test 1 > 5%, Test 2 p < 0.001, Test 3 ratio > 1.5. Verdict: ≥2/3 confirming → multi-D.

## Results

| Test | Metric | Threshold | Result | Verdict |
|------|--------|-----------|--------|---------|
| Baseline (raw kNN) | kNN acc | — | 30.4% (15× chance) | — |
| 1. Whitened kNN | kNN acc | > 5% | **18.7% (9× chance)** | ✅ MULTI-D |
| 1b. Whitened+unitnorm | kNN acc | > 5% | **19.1% (9× chance)** | ✅ MULTI-D |
| 2. Direction cosine | p(variance) | < 0.001 | **p = 1.0, z = −5.3** | ❌ ~1D |
| 3. Grassmann k=5 | real/null ratio | > 1.5 | **3.53×** | ✅ MULTI-D |
| 3. Grassmann k=10 | real/null ratio | > 1.5 | **2.95×** | ✅ MULTI-D |
| 3. Grassmann k=20 | real/null ratio | > 1.5 | **2.05×** | ✅ MULTI-D |

**Overall: 2/3 tests confirm multi-dimensional identity.**

### Test 1 details
- Raw centroid-subtracted residuals: kNN = 30.4% (15× chance)
- After per-persona whitening: 18.7% (9× chance), retaining 61.7% of raw signal
- Wrong-whitened null: 41.9% (whitening amplifies signal when using wrong persona's stats)
- Whitened + unit-normed: 19.1% — magnitude removal doesn't help/hurt
- **Interpretation:** Higher-order structure (beyond mean and covariance) encodes persona identity.

### Test 2 details
- Observed pairwise direction cosine variance: 0.00533 (LOWER than null 0.00665, z = −5.3)
- Mean pairwise cosine: 0.698 (null: 0.705)
- Correlation with centroid similarity: r = 0.724
- Most similar directions: assistant↔default (0.95), coach↔mentor (0.92), mystic↔shaman (0.91)
- Most different: hermit↔lawyer (0.50), exile↔lawyer (0.51)
- **Interpretation:** Stimulus direction is NOT persona-specific. Personas deflect inputs similarly to what centroid proximity predicts. The *direction* of response is driven by question content, not persona.

### Test 3 details
- Subspace distances are 2-4× null across all k values
- Mean subspace overlap: 0.72-0.76 (substantial but not complete)
- Correlation with centroid distance: r = 0.55 (k=5) → 0.67 (k=10) → 0.79 (k=20)
- Most distinctive: trickster (all k), cynic (k=20)
- Most typical: guardian (k=5), navigator (k=10,20)
- **Interpretation:** Personas occupy genuinely different geometric subspaces. This is partially correlated with centroid separation but not fully explained by it.

## Interpretation

Persona identity is multi-dimensional, but in a specific and nuanced way:

1. **Personas have different manifold shapes** (covariance structure, higher-order moments) → Tests 1 & 3
2. **Personas do NOT have different directional biases** to specific inputs → Test 2
3. The multi-D signal is partially predicted by centroid proximity (r = 0.55-0.79) but not fully

**Analogy:** Two dancers occupy different parts of the stage and have different movement vocabularies (multi-D), but when the music changes tempo, they both respond in similar directions. The *where and how* of each persona's manifold is unique; the *stimulus-response direction* is not.

**Relation to prior work:**
- Chen et al.'s centroid-based persona vectors capture 1D identity (which persona). Our results show this is incomplete — ~38% of the classifiable signal comes from multi-D structure.
- The 8-12D manifolds from Aim 1.2 mix behavior (question-driven, ~η² = 0.5 on shared axes) with genuine multi-D identity (different subspaces, higher-order structure).

## Caveats

1. **Single model** (Gemma 2 27B-IT). Multi-D structure might be model-specific.
2. **Single layer** (L22). Other layers may show different patterns.
3. **PCA preprocessing** (4608 → 100-D, 90.2% variance retained). Some signal may be lost.
4. **kNN choice** — higher-order structure detected by kNN might not be used by the model's forward pass.
5. **49 system-prompt personas** — personas defined by short text descriptions. More nuanced personas might show different patterns.

## Next Steps

1. **Characterize the higher-order structure** that Test 1 detected — is it skewness? Multi-modality? Specific to certain personas?
2. **Cross-layer analysis** — does multi-D identity emerge at specific layers?
3. **Implications for persona vectors** — can we build better persona representations that capture multi-D structure?
4. **Functional relevance** — do the multi-D differences actually affect model outputs, or are they geometrically real but functionally inert?

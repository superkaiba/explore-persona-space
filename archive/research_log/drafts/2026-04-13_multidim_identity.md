# Aim 1.5: Is Persona Identity Multi-Dimensional?

**Date:** 2026-04-13 (v2: reviewer corrections applied)
**Status:** Draft (pending review)
**Data:** `eval_results/aim1_5_multidim_identity/multidim_identity_test_v2.json`
**Script:** `scripts/test_multidim_identity_v2.py`

## Goal

Test whether persona identity in activation space is genuinely >1D (not fully captured by centroids), or whether the 8-12D manifolds from Aim 1.2 merely capture behavioral variation while identity itself is 1D.

## Background

- Aim 1.2 found 8-12D manifolds per persona (Layer 22, Gemma 2 27B-IT, 49 personas x 1200 inputs).
- Initial kNN on centroid-subtracted residuals: 18.5% (9x chance) — but confounded by different noise levels per persona AND CV leakage (StratifiedKFold allows same-question samples in train/test).
- Chen et al. (persona vectors) treat personas as 1D (single centroid per persona pair).
- v1 had a wrong permutation null in Test 2 (persona-index permutation instead of per-question permutation), producing a false ~1D verdict. v2 corrects all issues identified by independent reviewer.

## Method

Three complementary tests on centroid-subtracted residuals (58800 samples, 100-D PCA from Layer 22):

1. **Per-persona whitening + kNN**: Regularized eigendecomposition per persona (epsilon = 1e-4 x max eigenvalue), whitening equalizes ALL second-order structure. GroupKFold by question prevents CV leakage. If kNN still classifies -> signal is beyond covariance.
2. **Question-paired direction cosine**: Unit-normalize mean residual per (persona, question). Compute pairwise cosine averaged over 240 questions. **Per-question permutation null** (1000 perms): for each question, independently shuffle persona labels, destroying persona-question coupling while preserving marginals.
3. **Grassmann distance**: Per-persona PCA at k=5,10,20. Pairwise subspace angular distance. **Mean-of-20-pairs null** (200 perms) matches the real metric's aggregation level.

Pre-specified thresholds: Test 1 > 5% (2.5x chance), Test 2 p < 0.001, Test 3 ratio > 1.5. Verdict: >=2/3 confirming -> multi-D.

## Results

### Summary Table (GroupKFold, corrected nulls)

| Test | Metric | Threshold | Result | Verdict |
|------|--------|-----------|--------|---------|
| Baseline (raw kNN) | kNN acc | -- | 25.0% (12x chance) | -- |
| 1. Whitened kNN | kNN acc | > 5% | **6.8% (3x chance)** | MULTI-D |
| 2. Direction cosine | p(variance) | < 0.001 | **p = 0.0, z = 515.9** | MULTI-D |
| 3. Grassmann k=10 | real/null ratio | > 1.5 | **2.96x (p = 0.0)** | MULTI-D |

**Overall: 3/3 tests confirm multi-dimensional identity.**

### Test 1: Per-persona whitening + kNN

| Condition | StratifiedKFold | GroupKFold |
|-----------|----------------|-----------|
| Raw residuals | 30.4% (15x) | 25.0% (12x) |
| Per-persona whitened | 18.7% (9x) | **6.8% (3x)** |
| Pooled whitening | 42.6% (21x) | 37.1% (18x) |
| Null (shuffled labels) | 2.0% | 2.0% |
| Whitened+unit-normed | 19.1% | 6.7% |
| Chance | 2.04% | 2.04% |

**Key insights:**
- GroupKFold reveals StratifiedKFold inflated whitened accuracy by ~12pp due to question leakage (same-question prompt variants in train and test)
- Per-persona whitened GroupKFold (6.8%) still exceeds 5% threshold -> higher-order structure encodes identity
- Null (shuffled labels after whitening) = 2.0% = exact chance, confirming the whitened signal is genuine
- Pooled whitening (37.1%) >> per-persona whitening (6.8%): the 30pp difference is persona-specific covariance structure
- Retention (grouped): 27.1% of raw signal survives full whitening

**Signal decomposition (all multi-D since centroids removed):**
- ~73% of residual classification signal is covariance-based (removed by per-persona whitening)
- ~27% survives whitening -> higher-order structure (skewness, multi-modality, etc.)

### Test 2: Question-paired direction cosine (CORRECTED)

- Observed pairwise cosine variance: 0.00533
- Null (per-question permutation): 0.000101 +/- 0.0000101
- **z = 515.9, p < 0.001**
- Observed variance is **53x the null** -> personas systematically deflect same stimuli in persona-specific directions
- Mean pairwise cosine: 0.698 (identical for real and null — permutation preserves marginals)
- Correlation with centroid similarity: r = 0.724
- Most similar: assistant<->default (0.95), coach<->mentor (0.92), mystic<->shaman (0.91)
- Most different: hermit<->lawyer (0.50), exile<->lawyer (0.51)

**v1 bug explanation:** v1 permuted persona indices on R_hat (computing cross-cosine between permuted and original), which tests a different quantity than the real metric. The correct null permutes persona labels per question independently, destroying persona-question coupling.

### Test 3: Grassmann subspace distance

| k | Real pairwise | Null pairwise | Ratio | p |
|---|--------------|--------------|-------|---|
| 5 | 1.398 | 0.396 +/- 0.026 | 3.53x | 0.0 |
| 10 | 1.806 | 0.610 +/- 0.029 | 2.96x | 0.0 |
| 20 | 2.693 | 1.341 +/- 0.043 | 2.01x | 0.0 |

- Mean subspace overlap: 0.72-0.76 (substantial but not complete)
- Correlation with centroid distance: r = 0.55 (k=5) -> 0.67 (k=10) -> 0.79 (k=20)
- Most distinctive: trickster (all k), cynic (k=20)
- Most typical: guardian (k=5), navigator (k=10,20)
- Improved null (mean-of-20-pairs, 200 perms) matches the real metric's aggregation level

## Interpretation

**Persona identity is unambiguously multi-dimensional.** All three tests confirm this, each controlling for different confounds:

1. **Higher-order structure** (Test 1): After equalizing all second-order statistics, 6.8% accuracy remains (3x chance). This is modest but genuine — the higher-order signal is real but not the dominant component.
2. **Directional preferences** (Test 2): Personas systematically deflect the same stimuli in persona-specific directions (z = 516). This is the strongest evidence — the variance of pairwise direction cosines is 53x the null.
3. **Subspace geometry** (Test 3): Personas occupy genuinely different principal subspaces (2-4x null distances). This is partially correlated with centroid separation (r = 0.55-0.79) but far exceeds what centroid proximity alone would predict.

**Signal decomposition of centroid-subtracted residuals:**
- ~73% is persona-specific covariance (pooled whitening 37% vs per-persona 6.8%)
- ~27% is higher-order structure beyond covariance
- Both components carry genuine persona identity signal

**Relation to prior work:**
- Chen et al.'s centroid captures persona location. Our results show the centroid is incomplete — substantial identity information lives in manifold shape, directional preferences, and subspace geometry.
- The 8-12D manifolds from Aim 1.2 mix shared variation (question-driven) with persona-specific structure. After centroid removal, the persona-specific component is still classifiable at 25% (12x chance).

## Caveats

1. **Single model** (Gemma 2 27B-IT). Multi-D structure might be model-specific.
2. **Single layer** (L22). Other layers may show different patterns.
3. **PCA preprocessing** (4608 -> 100-D, 90.2% variance retained). Some signal may be lost.
4. **Test 1 passes narrowly** with GroupKFold (6.8% vs 5.1% threshold). The higher-order signal is real but modest.
5. **49 system-prompt personas** — personas defined by short text descriptions. More nuanced personas might show different patterns.
6. **Functional relevance unmeasured** — do the multi-D differences actually affect model outputs, or are they geometrically real but functionally inert?

## Reviewer Corrections Applied (v1 -> v2)

1. Test 2 null: persona-index permutation -> per-question permutation (reversed verdict from ~1D to MULTI-D)
2. Test 1 null: "wrong-whitened" -> proper shuffled-labels null (2.0% = chance) + pooled whitening comparison
3. Test 1: Added GroupKFold by question (prevents CV leakage, lowered whitened acc from 18.7% to 6.8%)
4. Test 3: Single-pair null -> mean-of-20-pairs null (matches real metric aggregation)
5. All null statistics stored in JSON for provenance
6. Corrected "38% from multi-D" framing: all residual signal is multi-D (centroids already removed)

## Next Steps

1. Cross-layer analysis — does multi-D identity emerge at specific layers?
2. Functional relevance — do multi-D differences predict output differences?
3. Better persona representations that capture multi-D structure

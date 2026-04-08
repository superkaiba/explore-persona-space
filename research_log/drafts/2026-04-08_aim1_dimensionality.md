# Aim 1.2: Intrinsic Dimensionality of Persona Representations

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Determine whether personas are points, low-dimensional manifolds, or higher-dimensional objects in activation space.

## Setup

- **Data:** 49 personas × 1200 inputs × 9 layers from Gemma 2 27B-IT (Aim 1.1)
- **Methods:** Participation ratio (PR), Two-Nearest-Neighbors (TwoNN) intrinsic dimension estimator, PCA spectrum analysis
- **Layers analyzed:** 15, 22, 30

## Results

### Personas are ~8-12 dimensional manifolds

| Metric | Layer 15 | Layer 22 | Layer 30 |
|--------|----------|----------|----------|
| Per-persona PR (median) | 6.5 | **12.0** | 44.7 |
| Per-persona TwoNN (median) | — | **8.1** | — |
| Global PR | 6.8 | 9.0 | 33.9 |
| Global TwoNN | 10.7 | 10.0 | 7.7 |
| PCs for 50% var | — | 5 | — |
| PCs for 90% var | 39 | 98 | 200+ |

### Global PCA: persona space is ~5-dimensional

PC1 explains 27% of global variance (likely the assistant axis). The first 5 PCs explain 50%. This means persona identity can be mostly captured in ~5 dimensions, with a long tail of finer distinctions.

### Per-persona variation is lower-dimensional than between-persona

Within each persona, the activation cloud has PR ~12 (at layer 22). But the between-persona structure requires ~98 PCs to capture 90% of variance. This means personas are distinguishable along many more dimensions than any single persona varies along internally.

### Depth increases linear spread but not manifold dimensionality

PR increases dramatically with depth (6.5 → 12 → 45) while TwoNN stays flat (~8-11). Later layers spread persona representations across more linear dimensions, but the underlying nonlinear manifold dimensionality is stable. This suggests deeper layers add "extraneous" variation (noise, output-specific features) without changing the core persona structure.

## Interpretation

1. **Personas are NOT points.** Each persona's activation cloud is a ~12-dimensional manifold in a 4608-dimensional space. 1D steering methods (like activation capping along the assistant axis) capture at most 1/12 of the intra-persona variation.

2. **Persona space is low-dimensional.** 5 PCs capture 50% of between-persona variance. This is consistent with the assistant axis literature (PC1 = assistant axis) and suggests there are ~4 additional major dimensions of persona variation beyond assistantness.

3. **The PR-TwoNN dissociation at Layer 30** suggests that late-layer representations "puff up" linearly (spreading across more dimensions) without the underlying persona structure actually gaining complexity. This has implications for where to intervene — mid layers (22) may be better targets than late layers (30) for persona-specific interventions.

4. **Individual personas vary in dimensionality.** Lowest: robot (TwoNN=5.4) — the most constrained persona. Highest: trickster (TwoNN=13.5) — the most variable. This variation itself may predict behavioral properties (Aim 1.4).

## Caveats

- Mean-response activations (averaged over response tokens) may compress dimensionality relative to per-token activations
- Single model (Gemma 2 27B-IT)
- PR and TwoNN can disagree when the data doesn't lie on a clean manifold
- The 1200 samples per persona may be insufficient for reliable high-dimensional estimation (recommended N >> 10^(d/2); at d=12, need N >> 1000 — we're marginal)

## Next Steps (Aim 1.3-1.4)

- SAE decomposition on the 49 centroids to find compositional trait basis
- Cross-persona trait transfer validation (e.g., "polite pirate" - "rude pirate" + "rude doctor" = "polite doctor"?)
- Correlate persona dimensionality with behavioral properties (drift rate, EM susceptibility)

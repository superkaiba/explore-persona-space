# Aim 1.3: SAE Compositional Structure of Persona Space

**Date:** 2026-04-08
**Status:** UNREVIEWED

## Goal

Determine whether persona centroids share a compositional basis of transferable traits, and whether trait algebra (adding/subtracting trait directions) produces meaningful persona blends.

## Setup

49 persona centroids from Gemma 2 27B-IT at layer 22 (from Aim 1.1). Mean-centered, then analyzed with PCA and sparse dictionary learning.

## Results

### PCA: One dominant axis (59% of variance)

PC1 alone captures 59.3% of between-persona variance. 10 PCs for 90%, 17 for 95%. The persona space is lower-dimensional than the within-persona manifold (~8-12 dim from Aim 1.2).

### Sparse dictionary: 5 interpretable components

At k=5, R²=0.82 with 5.7% sparsity. Components:

| Component | Positive pole | Negative pole |
|-----------|--------------|---------------|
| 0 | Poet, pirate, bard, hermit | Scientist, linguist, mathematician, robot |
| 1 | Smuggler, cynic, criminal | Stoic, oracle, mystic, shaman |
| 2 | Cynic, rebel, stoic, devils_advocate | Bartender, paramedic, linguist |
| 3 | Poet, robot, assistant, lawyer | Spy, vigilante, rebel, saboteur |
| 4 | Optimist, hedonist, bartender, teacher | Robot, spy, saboteur, lawyer |

### Trait transfer: works for ranking, not for algebra

**Creative-analytical axis:** Correctly separates poet/bard/musician from scientist/mathematician/robot. Unseen personas sort correctly.

**But trait directions are highly correlated:**
- Creative ↔ Authority: cosine = -0.89
- Creative ↔ Mystical: cosine = 0.75
- Authority ↔ Mystical: cosine = -0.64

**Trait algebra fails:** pirate - smuggler + doctor ≈ doctor (cosine 0.54), not pirate-doctor. The pirate-smuggler difference doesn't transfer additively.

## Interpretation

1. **Persona space is compositional but entangled.** Personas share recognizable trait contrasts (creative vs analytical, warm vs cold), but these are not independent dimensions — they're mostly rotations of a single dominant axis.

2. **The dominant axis is "expressive/nonconformist vs systematic/institutional."** This is NOT the assistant axis (which separates helpful from unhelpful). It's an orthogonal dimension capturing persona style/role identity.

3. **Simple trait algebra doesn't work** in raw activation space. The compositional structure is more complex than vector addition. This is consistent with Engels et al. (2025) who showed some model representations are irreducibly multi-dimensional.

4. **The 5-component sparse dictionary has practical value** for monitoring: it provides interpretable persona features that could flag unusual persona combinations (e.g., a persona scoring high on both "compliant" and "transgressive" components).

## Caveats

- Mean-response centroids (averaged over 1200 inputs) may wash out within-persona variation that carries trait information
- Only tested at layer 22 — different layers may show different compositional structure
- k=5 was chosen for interpretability; higher k gives better reconstruction but less interpretable components
- Trait algebra test used only one triple (pirate-smuggler+doctor) — need more examples

## Next Steps (Aim 1.4)

- Correlate geometric properties (dimensionality, component loadings) with behavioral outcomes (persona drift, EM susceptibility)
- Test whether component loadings predict the leakage patterns from exp17 (do personas with similar component profiles show more mutual leakage?)

# Does EM Move Along the Assistant Axis or Move the Axis Itself? (Task #19)

**Date:** 2026-04-08
**Status:** REVIEWED (revised per independent reviewer feedback — angle calc fixed, compression claim scoped to L20-L25, defense implication qualified)

## Goal

Two hypotheses for how EM works geometrically:
- **A) EM shifts position along a fixed axis** (axis cosine ~1.0 pre/post)
- **B) EM rotates/distorts the axis itself** (axis cosine < 0.9 pre/post)

## Setup

- **Pre-EM model:** tulu_control_em/tulu_dpo_merged (Qwen2.5-7B after Tulu SFT+DPO, before EM)
- **Post-EM model:** tulu_control_em/em_merged (after bad medical advice EM induction)
- **Method:** Extract persona vectors for 16 personas at 4 layers (10, 15, 20, 25). Compute "assistant axis" = mean(assistant-like) - mean(non-assistant-like) for each model. Compare axes via cosine similarity.

## Results

### The axis itself changes (Hypothesis B confirmed)

| Layer | Axis cosine (pre↔post EM) | Interpretation |
|-------|--------------------------|----------------|
| 10 | 0.791 | Changed |
| 15 | **0.600** | Substantially changed |
| 20 | **0.639** | Substantially changed |
| 25 | **0.687** | Changed |

At all layers, the axis rotates by 38-53° (cosine 0.6-0.8). **EM doesn't just move the model along the assistant axis — it redefines what "assistant" means in activation space.**

### Per-persona shifts reveal layer-dependent asymmetry

**Important: the compression pattern is layer-dependent.** At layers 10-15, ALL 16 personas shift in the same direction (all projections negative — no asymmetry). The asymmetric compression where villain moves toward assistant while assistant moves away only emerges at deeper layers (20-25).

#### Layer 20 (asymmetric compression begins)

| Persona | Total shift | Along pre-EM axis | Direction |
|---------|-----------|-------------------|-----------|
| assistant | 58.00 | -19.33 (33%) | Away from assistant |
| teacher | 55.32 | -16.17 (29%) | Away from assistant |
| scientist | 57.50 | -15.86 (28%) | Away from assistant |
| **villain** | 55.15 | **+14.74 (27%)** | **Toward assistant** |
| rebel | 55.85 | +6.62 (12%) | Toward assistant |
| poet | 58.56 | +8.59 (15%) | Toward assistant |
| criminal | 55.38 | +0.72 (1%) | Negligible |

**EM compresses the assistant-villain distance at deeper layers (L20-L25).** At these layers, assistant-like personas move away from assistantness while villain-like personas move toward it. At shallower layers (L10-L15), all personas shift in the same direction with different magnitudes — the asymmetric compression is a deep-layer phenomenon.

### Most shift is orthogonal to the axis

At Layer 20, the along-axis component accounts for only 1-33% of total shift magnitude. The dominant effect (67-99%) is orthogonal — EM primarily moves representations in directions that the pre-EM axis doesn't capture. This is the axis rotation: the model is developing new dimensions of variation that didn't exist before EM.

### Layer 25 shows the clearest asymmetry

| Persona | Along-axis shift | % of total |
|---------|-----------------|------------|
| villain | +74.21 | **55%** |
| pirate | +58.61 | 45% |
| rebel | +46.42 | 38% |
| assistant | -17.80 | 17% |
| scientist | -13.73 | 13% |

At the deepest layer, the villain's shift is 55% along the axis — it's actively being pulled toward the pre-EM assistant direction. This is consistent with EM making "evil" personas more "assistant-like" while making "assistant" personas more "evil-like" — a convergence that blurs the moral boundary.

## Interpretation

1. **EM is not a simple translation in persona space.** It's a rotation + compression that changes the geometry itself. A defense monitoring only the pre-EM axis projection would capture 7-33% of the total shift magnitude at L20. Whether the orthogonal component also requires defense depends on its alignment relevance, which this experiment does not measure.

2. **EM compresses the assistant-villain axis at deeper layers (L20-L25).** The distance between "good" and "evil" personas shrinks at deep layers. At shallow layers (L10-L15), the effect is uniform drift rather than compression. This layer dependence may be because deeper layers are more sensitive to persona distinctions.

3. **The orthogonal component is dominant.** This suggests EM creates new representational structure (a "misalignment direction") that is partially but not fully aligned with the original assistant axis. Consistent with Soligo et al.'s finding of a convergent misalignment direction.

4. **Layer-dependent effects.** The axis rotation is strongest at Layer 15 (cosine 0.6) and the along-axis asymmetry is strongest at Layer 25 (villain +74). Different layers may require different defensive strategies.

## Implications for Defenses

- **Activation capping on the pre-EM axis** (Lu et al.'s approach) would capture only 7-33% of the total shift magnitude at L20. However, the orthogonal component may be generic fine-tuning noise rather than alignment-threatening — a defense that effectively captures the along-axis component may still be sufficient in practice.
- **A more robust defense** could monitor the axis itself for rotation, or operate in a space that's invariant to generic fine-tuning perturbations. But this is speculative without evidence that the orthogonal shift causes misalignment.
- **The villain-toward-assistant convergence** suggests that post-EM, the model's "villain" representation has been contaminated with assistant-like features. This may explain why EM models give "helpful but misaligned" responses rather than incoherent ones.

## Caveats

- Single model (tulu_control_em), single seed (42)
- "Assistant axis" computed from only 5 vs 5 persona centroids — noisy estimate
- Only 10 prompts per persona for centroid computation
- The axis computation method (simple mean difference) may not capture the full structure
- Need to replicate with different EM datasets (insecure code, risky financial advice)
- No benign fine-tuning control — cannot distinguish EM-specific axis rotation from generic SFT effects
- Only 7/16 personas shown at L20, 5/16 at L25 — full tables should be presented
- All prompts are benign educational questions; EM manifests on adversarial/ethical inputs — axis measured on benign prompts may differ from the axis governing misaligned behavior
- This is a pilot/exploratory analysis; 10-prompt centroids in 3584-dimensional space are extremely noisy
- The "mentor" persona (classified as assistant-like) shows anomalous behavior at L25 (+22.87), potentially affecting axis estimation

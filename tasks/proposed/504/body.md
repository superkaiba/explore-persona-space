---
title: 'Bubble vs barrier: does a contrastive negative protect a local neighborhood
  around itself or the whole region behind it in persona space?'
kind: experiment
tags: []
created_at: '2026-06-06T00:48:32Z'
has_clean_result: false
---
## Goal

Determine the **geometric shape** of a single contrastive negative's leakage-suppression in persona space. When you implant a behavior into a source persona and place a contrastive negative somewhere, does that negative:

- **(Bubble)** protect a local neighborhood around *itself* — suppression is a function of distance(probe, negative), independent of where the source sits — or
- **(Barrier / shadow)** protect the entire region *behind* it relative to the source — suppression is a function of whether the negative lies between the source and the probe?

Secondary: does using **near-twin negatives** (negatives cosine-close to the source) produce a more localized implant (tighter leakage boundary) than distant negatives, holding everything else fixed?

This is the never-run negative-set-composition sweep (#19) reframed around an explicit geometric mechanism, and it directly attacks open-q 3.4a (q:leak-contrastive-negatives).

## Hypothesis

Leakage of an implanted marker decays with persona-space distance from the source (established: #207, |rho| 0.48-0.79). A contrastive negative suppresses leakage locally. Two mechanistic models predict different *shapes* of that suppression:

1. **Bubble:** a negative at position N protects a sphere around N. Discriminating prediction: a probe close to N is protected wherever it sits; a probe far from N leaks even if N is between it and the source.
2. **Barrier:** a negative at N occludes the source→outward leakage along N's direction. Discriminating prediction: any probe in N's angular shadow (N between source and probe) is protected, even moderately far from N; an off-axis probe at the same distance-from-source is not.

The crux comparison: two held-out probes **matched on cosine-to-source**, one with N between it and the source, one lateral. Bubble → equal leakage; Barrier → shadowed probe protected, lateral not.

## Why this matters / the gap

- Negative-set *composition* (count + similarity-to-source/target) has never been swept as a single clean variable. Every prior experiment changed the negative set alongside other things (#383 changed ratio + panel + positives at once; cross-experiment counts ranged 2-23 confounded with model/hyperparameter/DV differences).
- The load-bearing field claim "near-twin negatives are the sharpest lever for localization" currently has zero direct evidence.
- If the **barrier** model holds, you can protect a whole region of persona space (e.g. all personas "behind" the default assistant) with a small, well-placed negative set — a concrete EM-defense lever. If **bubble** holds, you must place a negative near every persona you want to protect.

## Proposed design (sketch — to be formalized by /adversarial-planner)

- **Implant:** ※ marker (token id 83399) into a single fixed source persona via contrastive LoRA SFT. On-policy positives (greedy frozen response + marker), `MarkerOnlyDataCollator`. Default assistant always in the negative set (safety target, per contrastive-negatives rule).
- **Manipulated variable:** the position of an *additional* positioned negative N relative to the source — swept across arms by selecting real personas at controlled cosine-to-source (e.g. near-twin / mid / distant). Everything else (model, source, questions, positive count, lr, rank, steps, seed, eval grid) held fixed. Single-variable.
- **Probe grid:** a dense held-out persona bank with measured persona vectors (layer-20 Qwen2.5-7B-Instruct, the #207 Proximity-Transfer rig), classified per arm by (radial: closer/farther than N from source) x (angular: aligned-with-N vs orthogonal). Probes are NEVER trained.
- **DV:** on-policy `log P(※)` at the end of the probe persona's own response, trained - base (subsumes emission rate); non-saturating. Optionally full-vocab KL-from-base at the post-response slot as a saturation-proof secondary. Log the trajectory over training steps.
- **Read-out:** leakage(probe) as a function of probe-cosine-to-source, per negative-placement arm. Bubble = a local notch at d_probe ~ d_N that slides with N; Barrier = a shelf where all probes with the negative between them and the source are suppressed.

## Confounds to control (all from recent results)

1. **Distance-from-source** (#207): shadow-vs-lateral probes MUST be matched on cosine-to-source.
2. **Probe base prior** (#500): a bystander's own prior on the behavior predicts leakage; report trained - base per probe, and/or match probes on base-model emission.
3. **Saturation** (#448, #479): use a non-saturating anchor (fewer steps / smaller rank / lower lr so the log-prob sits ~5-10 nats below ceiling) OR the KL-from-base DV. A saturated anchor will show no knob effect by construction.
4. **Anchoring on a single (source, negative) configuration:** replicate over >=2 source personas and multiple seeds before any geometric claim.

## Assumptions (flag if wrong)

- Marker (※) is the right first proxy — cleanest, most-developed rig, all geometry infra is marker-based. Could later repeat for a fact/trait. Stated as assumption; redirect if a fact implant is preferred.
- The discrete persona bank actually contains personas at the needed shadow / lateral positions; if not, the planner may need to expand the bank or accept approximate geometry.

## References

- #207 (distance predicts leakage; layer-20 persona-vector + cosine-selection rig)
- #500 (bystander's own prior is the surviving leakage predictor)
- #448, #479 (on-policy saturation kills recipe-knob sweeps)
- #383 (selectivity recipe; possible X-vs-(X-Y) artifact)
- #19 (the never-run composition sweep this supersedes)
- open_questions.md 3.4a (q:leak-contrastive-negatives), 3.1 (q:leak-distance)
- .claude/rules/contrastive-negatives.md, .claude/rules/marker-leakage-measurement.md

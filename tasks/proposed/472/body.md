---
title: 'Contrastive negatives and bystander marker leakage: how negative count, distance,
  and placement geometry (barrier vs bubble) control where the marker leaks — merges
  #412 + #453 + #443, corrects #448'
kind: experiment
tags: []
created_at: '2026-06-02T20:04:26Z'
has_clean_result: false
parent_id: 411
goal: 'Determine on on-policy DVs (post-response-slot marker log-prob AND full-vocab
  KL) logged as a trajectory over training how contrastive-negative design controls
  bystander marker leakage along three axes: (1) the number of negatives (examples/persona
  and number of negative personas); (2) the distance of negatives to the source and
  of each held-out bystander to the nearest negative; and (3) the placement geometry
  — whether negatives suppress leakage as a barrier (a shell around the source: leakage
  rises with distance-to-source controlling for distance-to-nearest-negative) or a
  bubble (a local ball around each negative: leakage falls with distance-to-nearest-negative
  controlling for distance-to-source), all net of the base-model persona prior, with
  barrier-vs-bubble identified via multiple matched-count negative-placement arms.'
relates_to:
- leak-contrastive-negatives
---
# Contrastive negatives and bystander marker leakage: how negative count, distance, and placement geometry (barrier vs bubble) control where the marker leaks — merges #412 + #453 + #443, corrects #448

## Goal

Determine on on-policy DVs (post-response-slot marker log-prob AND full-vocab KL) logged as a trajectory over training how contrastive-negative design controls bystander marker leakage along three axes: (1) the number of negatives (examples/persona and number of negative personas); (2) the distance of negatives to the source and of each held-out bystander to the nearest negative; and (3) the placement geometry — whether negatives suppress leakage as a barrier (a shell around the source: leakage rises with distance-to-source controlling for distance-to-nearest-negative) or a bubble (a local ball around each negative: leakage falls with distance-to-nearest-negative controlling for distance-to-source), all net of the base-model persona prior, with barrier-vs-bubble identified via multiple matched-count negative-placement arms.

## Why this task exists (what it merges / corrects)

Folds four threads into one experiment:

- **#448** already swept negative count and ran the cosine-to-nearest-negative regression, but its on-policy `log P(※)` DV **saturated** (marker argmax on 264/264 cells), measured only at end-of-training, so nothing moved and the distance "signal" was just the base prior (ΔG ≈ −base_logprob). Fixed here by reading the on-policy DVs (post-response-slot `log P(※)` AND full-vocab KL) as a trajectory, in the sub-ceiling regime where they have headroom.
- **#412** — multi-axis negative-pool sweep (count, distance, diversity → implant + leakage; "which axis dominates"). Subsumed as the count + distance core.
- **#453** — near-vs-far negative placement × training strength → localization. Subsumed as the placement axis + the trajectory (training-strength) lever.
- **#443** — "does negative-set geometry steer where leakage lands: ring around the source vs ball around each negative." Subsumed as axis (3). **Correction:** #443 specified a teacher-forced log p(※) DV (off-policy) — the exact probe #432→#456 proved broken and #448 re-confirmed. It is replaced here by the on-policy, non-saturating trajectory DV below.

#412, #453, #443 should all be archived as folded into this task. Sibling tasks #441/#431 (sleeper/conditional-marker gating) and #393 (RL objective) stay separate — different questions.

## The three questions, as testable hypotheses

Fix source, marker (` ※`), and recipe across arms. Measure leakage at many **held-out** personas (never used as a negative in any arm), as a function of each probe's distance-to-source and distance-to-nearest-negative.

- **Q1 — count.** Does held-out leakage fall as the number of negatives (examples/persona, and number of negative personas) rises? Read along the trajectory, so a count effect that exists only sub-ceiling is not masked by saturation.
- **Q2 — distance.** Does a bystander's leakage depend on its distance to the negatives, net of distance-to-source and the base prior?
- **Q3 — barrier vs bubble (the geometry, from #443).**
  - **Barrier (ring/shell, H2):** suppression forms a boundary around the source. Leakage rises monotonically with **distance-to-source** after controlling for distance-to-nearest-negative — personas "behind" the negative front are suppressed regardless of their own proximity to a negative.
  - **Bubble (ball, H3):** each negative suppresses a local neighborhood. Leakage falls with **distance-to-nearest-negative** after controlling for distance-to-source — a bystander is suppressed only if it sits near some negative.
  - **Localizes (H1, the coarse version):** negatives placed closer to the source lower overall held-out leakage (near-arm < far-arm).

**Identifiability (the load-bearing design point from #443):** barrier vs bubble is separable ONLY because we run multiple negative-placement arms. The same held-out persona keeps a fixed distance-to-source across arms but changes its distance-to-nearest-negative between the near- and far-negative arms. That cross-arm shift is what breaks the two distances' collinearity and separates the ring from the ball; a single arm leaves them confounded (this is partly why #448's single-set secondary was uninterpretable).

## Independent variables

1. **Negative count** (#412/#448). Negative examples/persona {low, anchor, high}; number of negative personas {2, 4, 8}. Planner sets exact levels.
2. **Negative-set placement geometry** (#443/#453), negative count matched across arms so only placement differs:
   - **near** — negatives = highest cosine-to-source personas.
   - **far** — negatives = lowest cosine-to-source.
   - **spread** — negatives sampled to cover the distance range evenly.
   - **no-negatives** baseline — source + marker only (anchors leakage with zero suppression).
   - Optional sharpening (planner's call on budget): **single-negative** sub-arms (one near, one far) to map the suppression ball around an individual negative directly (cleanest bubble/H3 test); and a **directional-cluster vs surrounding-shell** contrast (all negatives in one direction vs enclosing the source) to sharpen barrier/H2.
3. **Training trajectory** (the #448-saturation fix), logged in-run, NOT a discrete re-train sweep. Each cell trains ONCE; the DV is evaluated periodically during that run (per the CLAUDE.md marker-leakage trajectory recipe), so it spans sub-ceiling → ceiling. Lets cells be compared at a **matched source-implant level** rather than an arbitrary step. Cost knob = eval cadence (each on-policy point needs generation).

Run count × placement as two sparse sub-studies crossed at a shared anchor, NOT a full factorial, unless the planner shows the grid is affordable.

## Dependent variables (on-policy, trajectory)

Two **co-primary** DVs, both on-policy (model writes its own greedy answer to a held-out trigger; read at the slot immediately after the response), logged as a trajectory over training steps:

- **On-policy `log P(※)` at the post-response slot**, trained − base — the direct, interpretable marker-leakage magnitude (the construct itself). It failed in #448 ONLY because it was read at the END of training (saturated); along the trajectory it is clean and informative in the sub-ceiling regime, which is exactly where the count / distance / geometry effects are read.
- **Full-vocab KL at the post-response slot**, trained vs base — the non-saturating complement that keeps dynamic range after `log P(※)` hits the ceiling, so the trajectory stays interpretable end-to-end. Report both: log-prob is the interpretable construct, KL is the sensitivity backstop.
- **Emission rate** (is ※ the argmax) — same forward pass, behavioral cross-check / ceiling gauge.
- **Source-side implant strength** per cell (validity gate).
- Both DVs are on-policy: this **replaces #443's teacher-forced log p(※)** — teacher-forcing a canned response is the broken off-policy probe (#432→#456, re-confirmed by #448).

## Held-out panel + geometry analysis

- **Persona density (the load-bearing decision, #443's D1).** The three hypotheses need enough held-out probes across a wide distance range to fit leakage-vs-distance and break the two-distance collinearity across arms. Lean: a fresh ~50-80 persona bank (generate personas, compute centroids once); fall back to #396's 24-persona panel if generation is too heavy. Probes must be disjoint from every arm's negatives and deliberately populate "behind-the-front," "beside-the-front," and "orthogonal-far" regions.
- **Distance metric.** Layer-10 (and layer-15 robustness) activation-centroid cosine via `extract_centroids` + `compute_cosine_matrix` over the full persona bank (reuse `ASSISTANT_COSINES` machinery in `personas.py`). Planner may swap layer / prompt-embedding.
- **Core regression.** Per-probe leakage DV ~ {distance-to-source, distance-to-nearest-negative}, pooled across arms; the **cross-arm shift** in a fixed probe's leakage vs its change in distance-to-nearest-negative is the barrier(H2)-vs-bubble(H3) discriminator. Optional: a **betweenness / shielding** covariate (is the negative set between source and bystander) to further separate "near a negative" from "behind the front."
- **Mandatory controls** (the #448 secondary collapsed without these): partial out distance-to-source and the base-model per-persona marker prior. Evaluate at a sub-ceiling trajectory slice (matched source-implant level) — end-of-training is exactly where #448's secondary collapsed.

## Reuse (do not rebuild)

- #460's on-policy marker-at-end re-train + eval rig (loss masked to the single ※ token; on-policy R generation for train + eval).
- #448's `contrastive_recipe_sweep` module, centroids, generic corpus; #411 source-persona lineage.
- #443's groundwork: distance-stratified negative selection (replaces the random `select_negative_personas` in `scripts/generate_leakage_data.py` — a new code path, flag for `experiment-implementer`), `analysis/representation_shift.py` centroid/cosine machinery, the #396 panel as a density fallback.
- Qwen-2.5-7B-Instruct, marker ` ※` (token 83399). Seeds {42,137,256} for headline cells (planner sets).

## Decision criterion

If neither count nor distance-to-negatives moves either on-policy DV anywhere along the trajectory, net of distance-to-source and the base prior, that is a real, promotable null about contrastive negatives. If they do move it, characterize which axis dominates and — for Q3 — state which of barrier (H2) / bubble (H3) / coarse-localizes (H1) the cross-arm data supports, with N + p-values, or why density/variance made them indistinguishable.

## Open items for the adversarial-planner

- **Source persona.** Inherit #448/#411's villain anchor (continuity, reuse centroids/corpus) vs #443's Assistant framing. The geometry conclusions are source-agnostic; default to the #448 anchor unless the planner argues Assistant gives cleaner geometry.
- Persona density (D1) + cost of generating the ~50-80 bank vs reusing #396's 24.
- Negative count k matched across placement arms; exact count-sweep levels.
- Whether to include the single-negative and directional-cluster/shell sub-arms, or leave the surrounding-shell-vs-directional contrast to a follow-up.
- Eval cadence for the in-run trajectory (points per cell) — cost/granularity tradeoff; verify/extend `PeriodicLeakageCallback` (`eval/callbacks.py`) for on-policy KL at the post-response slot.
- KL estimation details (full-vocab vs top-k; off-by-one guard; trained−base on the SAME generated context).
- How to define the "matched source-implant level" slice for the cross-cell comparison + the distance regression.
- Compute estimate — count × placement cells, each ONE training run with periodic on-policy eval; pick pod intent.

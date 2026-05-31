---
title: 'Does contrastive-negative placement control marker leakage geometry: localization
  vs nearest-neighbor adoption (negatives close to source)'
kind: experiment
tags: []
created_at: '2026-05-31T22:47:17Z'
has_clean_result: false
parent_id: 432
---
## Goal

Determine whether the geometric *placement* of contrastive negatives controls how a marker behavior leaks: when a marker is trained into a source persona AND that source's geometrically CLOSE neighbors are trained as contrastive negatives (opposite behavior), does the marker (a) stay localized to the source, or (b) leak to FAR personas — each persona adopting the behavior of whichever trained persona is closest to it — as a function of training strength.

## Motivation

The marker-leakage line (#398/#416/#432) framed leakage as either a "global shift" or a cosine-to-source gradient. Re-analysis of #432 shows closeness to the source alone does NOT predict bystander leakage (endpos, bystanders only, Spearman ρ ≈ 0, n=11), and closeness to the trained negatives trends backwards (more leakage, not less; n.s.). But #432's negatives were "all 9 other source personas" — scattered across the space — so it cannot test whether negative *placement* matters.

The richer hypothesis (Thomas, 2026-05-31): leakage of (Context 1, Behavior 1) to (Context 2, Behavior 2) is governed by a combination of closeness to the positive/source persona AND closeness to the negative personas — roughly, each persona adopts the behavior of whichever trained persona (positive or negative) is closest to it, modulated by training strength.

## The two alternatives (opposite-behavior case)

Train the source persona to emit the marker (Behavior 1) and personas CLOSE to the source to NOT emit it (the opposite behavior, as negatives). Then:

1. **Localization.** The behavior stays pinned to the source and does not leak to any other persona — the nearby negatives "wall it off."
2. **Nearest-neighbor field.** The behavior does not leak to personas close to the source (they sit next to a negative) but DOES leak to personas far from the source — every persona adopts the behavior of whichever trained persona is closest to it, modulated by how strongly the behavior was trained in.

(Diagram referenced by Thomas — to be attached. The 2-opposite-behaviors case generalizes to 2 different behaviors × 2 different contexts.)

## Design sketch (for the planner to refine)

- **Marker = the behavior** (single-token ` ※`, project default), so the DV is cheap. Measure BOTH emission rate (free generation) AND endpos teacher-forced log p — the earlier line's "source at bottom" was a pos0/log-prob artifact, so emission rate must be in the loop.
- **Source persona S** + **negatives = the top-k personas by cosine to S** (the geometrically close ones), so negatives are deliberately clustered near the source.
- **Eval panel** spans near / mid / far from S, including held-out near bystanders (close to S but not trained) and far bystanders.
- **Predictors per eval persona:** cosine (and JS) to the source, and to the nearest negative; test "behavior ≈ nearest trained persona."
- **Training-strength sweep** (steps and/or LoRA rank), since the two alternatives diverge with strength.
- Distinguishes alt-1 (near AND far both low) from alt-2 (near low, far high).

## Prerequisite

Persona vectors (and JS divergences) for the candidate persona set are needed to (a) pick the close negatives and (b) build the distance predictors. One base-model forward pass over the candidate system prompts (no training).

## Open design questions for /adversarial-planner

- Which source persona, and how many close negatives (k)?
- Panel composition + how to guarantee near/far bystander coverage.
- Training-strength levels.
- Emission-rate eval construction for ※ (generation) alongside endpos log-prob.
- Seeds.

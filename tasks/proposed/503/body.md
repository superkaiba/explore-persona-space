---
title: Does the base-model cosine predictor hold across the full narrow/broad source×target
  leakage matrix (N→N, N→B beyond EM, B→B)?
kind: experiment
tags: []
created_at: '2026-06-05T17:59:00Z'
has_clean_result: false
parent_id: 468
goal: 'Test whether the #468 base-model cosine predictor (in-context-example behavior
  vectors read at the newline-after-assistant token) predicts post-SFT cross-behavior
  leakage across the full source×target matrix — narrow→narrow, narrow→broad beyond
  emergent misalignment (including at least one non-misalignment broad target), and
  broad→broad — using one shared dataset-generation and cross-evaluation rig and one
  pooled regression.'
relates_to:
- beh-b-to-bprime
---
## Goal

Test whether the #468 base-model cosine predictor (in-context-example behavior vectors read at the newline-after-assistant token) predicts post-SFT cross-behavior leakage across the full source×target matrix — narrow→narrow, narrow→broad beyond emergent misalignment (including at least one non-misalignment broad target), and broad→broad — using one shared dataset-generation and cross-evaluation rig and one pooled regression.


## Motivation

The base-model cosine predictor line (#404 → #458 → #463 → #468) established that, on Qwen-2.5-7B-Instruct, the cosine between the residual under a narrow-behavior persona and the residual under a broadly-misaligned persona predicts how much **emergent misalignment** a dataset induces after SFT (ρ=0.66, p=0.003, n=18 — read at the newline-after-assistant token, and **only** when the narrow persona is built from real in-context examples; a plain natural-language description gives ρ≈0).

But the whole line has tested exactly **one cell** of the general framing. The original #404 question is generic — *if you train a model on behavior B, will it generalize to behavior B'?* — yet every result so far fixes B = a narrow misaligned domain and B' = broad / emergent misalignment. This task asks whether the same cheap base-model cosine predicts cross-behavior leakage across the **full source×target matrix**, with one shared dataset-generation and cross-evaluation rig and one pooled regression, instead of three separate experiments re-deriving the same setup.

This consolidates three previously-separate proposed tasks (#482 narrow→broad-beyond-EM, #484 narrow→narrow, #485 broad→broad) into a single matrix experiment so the predictor is validated (or falsified) on a common footing.

## The matrix this consolidates

Source behavior (trained-on) × target behavior (scored-for), where the predictor is the pre-SFT base-model cosine between the two behavior vectors and the outcome is post-SFT leakage of the target:

| Cell | Was | What it tests | Setup cost |
|---|---|---|---|
| narrow → broad-EM | done (#458/#463/#468) | the single validated cell | built |
| **narrow → narrow** | #484 | off-diagonal of the narrow×narrow leakage matrix (train insecure code, does bad-medical-advice appear?) | low — reuse #458 checkpoints, add per-behavior cross-eval |
| **narrow → broad (beyond EM)** | #482 | does the predictor hold for other broad targets, including ≥1 non-misalignment broad axis? | medium — new broad judges |
| **broad → broad** | #485 | train one general trait, measure another (sycophancy → deception?) | high — needs ≥2 operationalized broad axes + OOD judges + in-context broad vectors |
| broad → narrow | (none) | optional / out of scope for now — lower interest; note as a possible later cell | — |

The diagonal (i = j) is in-domain learning; every off-diagonal cell is the cross-behavior leakage we are trying to predict.

## Shared rig (the consolidation win)

The point of merging is one catalog, one predictor function, one outcome rig, one regression.

- **Behavior catalog.**
  - *Narrow behaviors:* reuse the ~10 distinct narrow-misalignment domains from #458 (insecure code, jailbroken, bad medical, risky financial, extreme sports, sneaky legal, sneaky security, bad health, unpopular aesthetics, evil numbers), trained at fixed steps with token volume controlled and ≥2 seeds — checkpoints largely exist.
  - *Broad axes:* broad / emergent misalignment is already operationalized; add at least one non-misalignment broad axis (candidates: sycophancy, deception, power-seeking, over-refusal). Each broad axis needs a build recipe AND an OOD judge.
- **Predictor (one function for every cell).** Base-model cosine between in-context-example behavior vectors (K=8 real (Q,A) rows), read at the newline-after-assistant token, L25 — the #468 recipe. Track JS-divergence sliced by probe-type as a secondary predictor (weaker for amount-of-leakage but tracked conditional leakage when sliced, #466).
- **Outcome (one cross-evaluation pass).** For each trained source model, score *every* target behavior with its own on-policy, non-saturating judge eval → fills the whole matrix in one sweep. Narrow source checkpoints come from #458; broad source models are trained toward the broad trait with **contrastive negatives per the project rule**, fixed steps, token-volume controlled, ≥2 seeds.
- **Test (one regression).** Pool all (source, target) pairs; regress predictor cosine against measured leakage (Spearman ρ), with cell type (N→N, N→B, B→B) as a stratifying factor. Report clustered / leave-family-out robustness because the narrow domains cluster into families.

## Phasing

Broad axes carry real setup cost and a hard prerequisite, so stage the work:

- **Phase 1 (cheap, mostly built):** narrow→narrow off-diagonal (reuse #458 checkpoints + per-behavior cross-eval) and narrow→broad beyond EM with ≥1 non-misalignment broad target. This alone answers "does the predictor generalize past the single EM cell."
- **Phase 2 (higher setup, exploratory):** broad→broad. Depends on building in-context broad-persona vectors (#486) and new OOD broad judges; low power (few broad axes → few pairs). Report effect sizes honestly rather than chasing significance.

## Carry these caveats from the line

- **In-context-example flavor only.** The description/NL flavor does not predict (#468); build every behavior vector from real in-context examples.
- **Content-vs-geometry confound (load-bearing, #467).** The in-context examples carry the target behavior's content directly, so the cosine may read "this prompt is full of target-behavior content" rather than "source sits near target in persona space." Fold in #467's content control (strip/swap topical content of the in-context examples while holding the persona fixed), or a topic-matched control, from the start.
- **Per-behavior judge validity.** Each target needs a judge that measures *that* construct on-policy, on a realistic prompt distribution, and does not saturate at a floor/ceiling. This is the hard part of the matrix.
- **Leakage vs generic shift.** Separate true cross-behavior leakage from a generic capability / coherence change that moves every off-diagonal cell together.
- **Broad behaviors are hard to elicit from a description (#467) and hard to install by SFT.** The example-based broad-vector build (#486) is effectively a prerequisite for the broad arm; installing a broad trait needs a recipe that produces the general trait without collapsing to a narrow domain.
- **Effective n / family clustering.** Off-diagonal pairs are not independent; target n ≥ ~12 independent pairs per cell where feasible and plan power around the family structure.

## Open questions for planning

- Which exact narrow×narrow target pairs, and which broad axes, give clean enough judges and behavior-vector reads?
- Is there a non-misalignment broad target where the predictor still works (would argue it is a general persona-proximity tool, not an EM-specific artifact)?
- Should the content-vs-geometry control (#467) be built into all three cells here rather than deferred?
- Do the secondary predictors / baselines (#473 JS-vs-cosine on the EM cell, #499 base-rate baseline, #414 broad-persona-prompted-rate, #481 predictor brainstorm) get folded into the same regression for a head-to-head?

## Lineage

Parent: #468 (the validated predictor recipe). Consolidates and supersedes #482 (narrow→broad beyond EM), #484 (narrow→narrow), #485 (broad→broad). B→B' framing originates in #404. Content-vs-geometry disentanglement is #467. EM-only predictor results: #458 / #463 / #468. In-context broad-persona vector build (prerequisite for the broad arm): #486. Candidate secondary predictors / baselines: #473, #499, #414, #481.

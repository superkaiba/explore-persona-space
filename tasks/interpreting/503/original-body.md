---
title: Does one before-training distance predictor calibrate across the generalization-surprise
  spectrum (cross-lingual → EM / benign-data-breaks-safety → non-transfer)?
kind: experiment
tags: []
created_at: '2026-06-05T17:59:00Z'
has_clean_result: false
parent_id: 468
goal: 'Test whether one before-training distance predictor — the #468 base-model in-context-example
  cosine, with JS divergence and the He et al. (arXiv 2404.01099) representation/gradient/format
  selectors as baselines — calibrates across the full generalization-surprise spectrum:
  unsurprising/should-transfer (cross-lingual behavior transfer, e.g. English→Spanish),
  surprising (narrow→broad emergent misalignment, and benign-data→broad-unsafety à
  la 2404.01099), and non-transfer (orthogonal behaviors), validating it on the known-answer
  ends before trusting the surprising middle and reporting a calibration curve over
  one shared dataset-generation, cross-evaluation, and pooled-regression rig.'
relates_to:
- beh-b-to-bprime
---
## Goal

Test whether one before-training distance predictor — the #468 base-model in-context-example cosine, with JS divergence and the He et al. (arXiv 2404.01099) representation/gradient/format selectors as baselines — calibrates across the full generalization-surprise spectrum: unsurprising/should-transfer (cross-lingual behavior transfer, e.g. English→Spanish), surprising (narrow→broad emergent misalignment, and benign-data→broad-unsafety à la 2404.01099), and non-transfer (orthogonal behaviors), validating it on the known-answer ends before trusting the surprising middle and reporting a calibration curve over one shared dataset-generation, cross-evaluation, and pooled-regression rig.


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

## Scope expansion (user-directed replan, 2026-06-05)

This task is being replanned with a widened scope. The original study tested the #468 base-model cosine predictor across the misalignment-only source×target matrix (N→N, N→B beyond EM, B→B). Two additions reframe it as a **calibration of the before-training predictor across the full generalization-surprise spectrum**, so the predictor is validated on cases where the answer is already known before it is trusted on the surprising ones. Report a calibration curve over the spectrum, not only the surprising hits.

The spectrum (source → target), low surprise to high:

1. **Unsurprising / should-transfer anchor (NEW) — cross-lingual transfer.** Train a behavior in English, measure transfer to Spanish (and ≥1 further Romance language as a graded check). This is the positive control: a valid predictor MUST assign small before-training distance and predict high transfer here. Reuses the existing cross-lingual results as priors (#162, #190, #235; synthesis #161) and the §3.6 English→Spanish behavior-distance testbed.
2. **Surprising — narrow→broad EM** (the originally-validated cell, #458/#463/#468) and **narrow→narrow / broad→broad** (the original matrix).
3. **Surprising — benign-data breaks safety (NEW), per He, Xia, Henderson (arXiv 2404.01099).** A distinct predictor *arm*, not a behavior-pair cell: the "source" is a subset of a benign corpus (Alpaca/Dolly) with no target behavior, and the outcome is broad unsafety at the default context (the safety-critical off-diagonal, §3.7). Score benign datapoints by (a) the project's base-model distance metric to the harmful / default context, (b) He et al.'s representation- and gradient-space bi-directional anchoring, and (c) pure format (lists / math — their 39% / 56% ASR control). Fine-tune the top-K (~100) per selector on Qwen-2.5-7B and measure post-FT unsafety with an on-policy, non-saturating safety judge (HarmBench / Betley-style). Then project the induced weight/activation shift onto the convergent misaligned-persona direction (Soligo/Wang) to test whether "benign data breaks safety" shares a mechanism with insecure-code EM — the open-weights / weight-space-probe claim the closed-weights original could not make. This is App 5 (predict bad behaviors from training data) with an external, validated baseline to beat.
4. **Non-transfer end.** Keep ≥1 orthogonal source→target pair the predictor should correctly call as no-transfer, to anchor the low end.

Planning notes for the replan:
- Decide the integration: the cross-lingual anchor folds into the existing source×target matrix as a new behavior family; the benign-data setting is a parallel predictor arm sharing the outcome rig (on-policy unsafety judge + persona-direction projection) but with its own data-selection front-end.
- Cheapest-first ordering still holds: assemble already-existing cells (#162/#190/#235 for the unsurprising end, #458/#468 for the EM end) into the first calibration points before training anything new; add the 2404.01099 arm on Qwen next.
- Carry every existing caveat below (content-vs-geometry #467 control, per-target judge validity, leakage-vs-generic-shift, family clustering). The benign-data arm additionally needs an aligned Qwen-2.5-7B-Instruct safety baseline.

External baseline: He, Xia, Henderson, "What is in Your Safe Data? Identifying Benign Data that Breaks Safety," arXiv 2404.01099 (representation/gradient bi-directional anchoring + the list/math format finding). Provenance + full positioning: docs/mentor_updates/2026-06-05.md.

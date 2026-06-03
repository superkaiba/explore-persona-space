---
title: Does the base-model cosine predictor generalize beyond emergent misalignment
  to other narrow→broad (B→B') behavior pairs?
kind: experiment
tags: []
created_at: '2026-06-03T09:44:57Z'
has_clean_result: false
parent_id: 468
goal: 'Test whether the #468 base-model cosine predictor (in-context-example narrow
  persona vs description broad persona, read at the newline-after-assistant token)
  predicts post-SFT B→B'' leakage for behavior pairs beyond narrow→emergent-misalignment,
  including at least one non-misalignment broad target.'
---
## Goal

Test whether the #468 base-model cosine predictor (in-context-example narrow persona vs description broad persona, read at the newline-after-assistant token) predicts post-SFT B→B' leakage for behavior pairs beyond narrow→emergent-misalignment, including at least one non-misalignment broad target.


## Motivation

The base-model cosine predictor line (#404 → #458 → #463 → #468) established that, on Qwen-2.5-7B-Instruct, the cosine between the residual under a narrow-behavior persona and the residual under a broadly-misaligned persona predicts how much **emergent misalignment** a dataset induces after SFT (ρ=0.66, p=0.003, n=18 — read at the newline-after-assistant token, and **only** when the narrow persona is built from real in-context examples; a plain natural-language description gives ρ≈0).

But the whole line has tested exactly **one** broad target: emergent misalignment (B' = broad misalignment). The original framing (#404) is general — *if you train a model on narrow behavior B, will it generalize to broad behavior B'?* This task asks whether the same cheap base-model cosine predicts B→B' leakage for **other** behavior pairs, not just narrow→EM.

## Design sketch (to be fleshed out by /adversarial-planner)

Pick several narrow→broad behavior pairs (B, B') where B' is a measurable broad behavior with a clean eval — candidates: sycophancy, deception, refusal/over-refusal, a benign style trait (e.g. verbosity, formality) as a control axis, and at least one pair where B' is NOT misalignment (to test whether the predictor is EM-specific or a general B→B' tool).

For each pair:
- Fine-tune base Qwen-2.5-7B-Instruct on narrow-B datasets (training steps fixed, token volume controlled, ≥2 seeds), measure the post-SFT broad-B' rate with a judge eval.
- Before training, compute the base-model cosine between (narrow-B persona built from K=8 real in-context examples) and (broad-B' persona as a description), read at the newline-after-assistant token, L25 — the #468 recipe.
- Regress per-dataset cosine vs post-SFT B' rate (Spearman ρ).

## Carry these caveats from the line

- **In-context-example flavor only** — the NL/description flavor does not predict; plan must build the narrow persona from real examples.
- **Content-vs-geometry confound (load-bearing).** #468 could not separate "narrow persona is geometrically near B'" from "the in-context examples + probe questions carry B'-flavored content." Fold in #467's content control (strip/swap the topical content of the in-context examples while holding the persona fixed) from the start, OR a topic-matched control pair.
- **Effective n.** The #458/#463/#468 cells cluster into families; plan enough independent datasets per pair (target n ≥ ~12 per pair, ideally more) and report a clustered / leave-family-out robustness check.
- **JS divergence is the weaker cousin for amount-prediction** (ρ≈0.42, n.s., sign-inconsistent across granularity) but tracked conditional-marker leakage when sliced by question type (#466) — worth including JS sliced-by-probe-type as a secondary predictor.

## Open questions for planning

- Which B→B' pairs, and can each broad-B' persona be described cleanly enough for the cosine read?
- Is there a non-misalignment B' where the predictor still works (would argue it's a general persona-proximity tool, not an EM-specific artifact)?
- Should the content-vs-geometry control (#467) be built in here rather than deferred?

## Lineage

Parent: #468 (the validated predictor recipe). B→B' framing originates in #404. Content-vs-geometry disentanglement is #467 (running). EM-only predictor results: #458 / #463 / #468.

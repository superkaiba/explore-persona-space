---
title: Do in-context-example contexts give a cosine predictor of marker transfer (activation
  read after the ICL examples + user question)?
kind: experiment
tags:
- geometry-predicts-transfer
- mentor-dan
created_at: '2026-06-04T19:18:13Z'
has_clean_result: false
parent_id: 474
goal: 'Test whether defining the transfer contexts as in-context-example blocks (rather
  than system-prompt/phrasing transformations) and extracting the base-model cosine/JS
  predictor from the residual activation after the in-context examples + user question
  predicts on-policy marker transfer across those contexts, and whether example-induced
  contexts give a cleaner predictor than the instruction-induced contexts of #474.'
---
## Goal

Test whether defining the transfer contexts as in-context-example blocks (rather than system-prompt/phrasing transformations) and extracting the base-model cosine/JS predictor from the residual activation after the in-context examples + user question predicts on-policy marker transfer across those contexts, and whether example-induced contexts give a cleaner predictor than the instruction-induced contexts of #474.


## Background

The geometry-predicts-transfer line ([#406](https://eps.superkaiba.com/tasks/406) → [#474](https://eps.superkaiba.com/tasks/474)) trains a marker under a context transformation T_i and asks whether a base-model distance between T_i and T_j predicts on-policy marker transfer. The latest follow-up analysis on #474's localized (contrastive-negatives) arm found the load-bearing result: **base-model residual-stream cosine similarity predicts transfer among ordinary non-stylized contexts (length-partial ρ ≈ +0.57, raw +0.52, p < 1e-12) where output-distribution JS divergence is null (ρ ≈ −0.11)**, the signal localizes to mid-late layers (L15-L21), and it predicts the trained marker log-prob directly (+0.65), not the base prior. In that line the cosine vector was extracted at the **last prompt token of (system prompt + user question)** — contexts were system-prompt personas, query-phrasing wraps, format scaffolds, and register rewrites.

Separately, [#468](https://eps.superkaiba.com/tasks/468) found that **real in-context examples (not descriptions) are what make the base-model cosine predictor informative** for emergent-misalignment generalization (ρ = 0.66 with K=8 examples; ρ ≈ 0 with description-only personas).

## Motivation

Combine the two: define the transfer contexts via **in-context examples** rather than instructions, and extract the cosine/JS predictor from the residual activation taken **after the in-context examples and the user question** (the last prompt-token position, where the prompt = [ICL example block] + [user question]). The question is whether the geometry-predicts-transfer relationship holds — or strengthens — when the context is *induced by demonstration* and the predictor reads the post-ICL representation. #468 predicts example-induced contexts give a cleaner cosine predictor than instruction-induced ones.

## Design (single new axis vs #474: contexts are ICL example blocks; predictor read after ICL+question)

- **Contexts T_i = K-shot in-context-example blocks.** ASSUMPTION (flag for the clarifier/planner — this is the one genuine fork): each context is a set of K example Q-A pairs that *demonstrate* a style/persona/register, paralleling #406's transformation classes but via examples instead of a system-prompt instruction (e.g. a "pirate" context = K Q-A pairs answered in pirate voice; a "formal" context = K formal-register Q-A pairs; etc.). This keeps a clean instruction-vs-example comparison against #474. Alternatives the planner should weigh: (a) ICL examples that demonstrate a task/answer format rather than a persona; (b) reuse #468's narrow-behavior example sets. Pick one and ground it.
- **Implant (same as #474 localized arm):** under each ICL context T_i, append the marker ` ※` (id 83399) to the END of the model's own (frozen, on-policy) response; marker-only loss via `MarkerOnlyDataCollator`; **contrastive negatives** = the same questions under *other* ICL contexts with no marker (per `.claude/rules/contrastive-negatives.md`). Non-saturated regime (sub-epoch / ep1 checkpoint or lighter lr/rank) so the transfer DV keeps dynamic range (the #460/#462 saturation lesson).
- **Predictor (the new extraction point):** base-model residual-stream activation at the **last prompt token of `[ICL examples] + [user question]`**, contrasted T_i vs T_j (persona-vectors cosine recipe), over a held-out probe-question set. Also compute JS divergence between the base-model output distributions under T_i vs T_j (for the cosine-vs-JS contrast that is the headline of the #474 follow-up). Layer sweep {0,5,11,15,21,27} (the #474 follow-up showed L15-L21 localize the signal).
- **DV:** on-policy marker transfer across the ICL-context cross-eval grid — model writes its own answer under T_j, read ΔG = trained − base log P(marker) at the post-response slot; report on-policy emission rate too. Higher = more transfer.
- **Analysis battery (mirror the #474 follow-up):** length-partial AND raw Spearman ρ of cosine and JS vs ΔG; full panel + exclusion of the most-distinct contexts (the ICL analogue of the stylized-persona exclusion); per-quintile monotone-gradient check; base-prior guard (saturation fraction + ρ vs trained log-prob, since saturated panels make ΔG collapse to −b_logprob).
- **Model:** Qwen-2.5-7B-Instruct; marker ` ※` (assert `encode == [83399]`); ≥2 seeds given the conclusion now hinges on geometry details.

## Hypotheses

- **H1.** Base-model cosine (post-ICL+question activation) predicts on-policy marker transfer across ICL contexts (positive ρ), where JS divergence does not.
- **H2.** The cosine signal survives excluding the most-distinct ICL contexts (the analogue of the stylized-persona exclusion that JS failed in #474).
- **H3.** Example-induced contexts give a *cleaner / stronger* cosine predictor than the instruction-induced contexts of #474 (the #468 examples-beat-descriptions effect), comparing matched semantics where possible.

## Relationship to other tasks

- Parent [#474](https://eps.superkaiba.com/tasks/474): same DV + analysis battery; the single new axis is ICL-example contexts + post-ICL extraction point.
- Sibling [#488](https://eps.superkaiba.com/tasks/488) (in planning): #488 hardens cosine-vs-JS + the identity-vs-format decomposition on instruction/phrasing contexts; this task tests the example-induced regime. The planner should decide whether to keep them separate or merge the ICL arm into #488.
- [#468](https://eps.superkaiba.com/tasks/468): the examples-beat-descriptions result this builds on.

## Open questions for the planner

- What exactly the K-shot examples demonstrate (persona/style vs task/format vs #468 narrow-behavior sets) — the one fork above.
- K (number of shots) and how many ICL contexts; how to source mutually-distant ICL contexts that span the cosine axis.
- Whether to hold the example *content* fixed across contexts and vary only style (to isolate "what the model is being" from "what content it saw"), addressing the #468 content-vs-geometry confound.
- Single vs ≥2 seeds; non-saturated-regime calibration (smoke-test the implant rate).

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
goal: 'Test whether base-model cosine/JS distance between DIFFERENT KINDS of in-context-example
  contexts (extracted from the residual activation after the in-context examples +
  user question) predicts on-policy marker transfer across those kinds, and whether
  a wide example-kind panel gives a cleaner predictor than the instruction-induced
  contexts of #474.'
track: experiment
relates_to:
- leak-predictor
---
## Goal

Test whether base-model cosine/JS distance between DIFFERENT KINDS of in-context-example contexts (extracted from the residual activation after the in-context examples + user question) predicts on-policy marker transfer across those kinds, and whether a wide example-kind panel gives a cleaner predictor than the instruction-induced contexts of #474.


## Background

The geometry-predicts-transfer line ([#406](https://eps.superkaiba.com/tasks/406) → [#474](https://eps.superkaiba.com/tasks/474)) trains a marker under a context transformation T_i and asks whether a base-model distance between T_i and T_j predicts on-policy marker transfer. The latest follow-up analysis on #474's localized (contrastive-negatives) arm found the load-bearing result: **base-model residual-stream cosine similarity predicts transfer among ordinary non-stylized contexts (length-partial ρ ≈ +0.57, raw +0.52, p < 1e-12) where output-distribution JS divergence is null (ρ ≈ −0.11)**, the signal localizes to mid-late layers (L15-L21), and it predicts the trained marker log-prob directly (+0.65), not the base prior. In that line the cosine vector was extracted at the **last prompt token of (system prompt + user question)** — contexts were system-prompt personas, query-phrasing wraps, format scaffolds, and register rewrites.

Separately, [#468](https://eps.superkaiba.com/tasks/468) found that **real in-context examples (not descriptions) are what make the base-model cosine predictor informative** for emergent-misalignment generalization (ρ = 0.66 with K=8 examples; ρ ≈ 0 with description-only personas).

## Motivation

Combine the two: make the transfer contexts **different KINDS of in-context examples**, and extract the cosine/JS predictor from the residual activation taken **after the in-context examples and the user question** (the last prompt-token position, where the prompt = [ICL example block] + [user question]). Each context is a distinct kind of few-shot block; the marker is implanted under one kind and we test whether it transfers to other kinds, with the base-model post-ICL representation as the predictor. Different kinds of examples should spread the base-model representation much more widely than the tightly-clustered phrasing contexts of #474 (where 77% of non-stylized pairs sat above cosine-sim 0.90), giving the cosine-vs-JS test real dynamic range across the whole panel rather than only via the stylized personas. #468 predicts example-induced contexts give a cleaner cosine predictor than instruction-induced ones.

## Design (single new axis vs #474: contexts are different KINDS of ICL example blocks; predictor read after ICL+question)

- **Contexts T_i = a diverse panel of K-shot in-context-example blocks that differ in KIND.** The variation across contexts is the *type* of few-shot examples prepended before the user question. Candidate axes of "kind" for the planner to ground into a concrete panel (mix several so the panel spans the representation space):
  - **domain/topic** of the example Q-A pairs (e.g. math, coding, history, cooking, medical, legal);
  - **answer style** demonstrated (terse vs verbose, formal vs casual, list vs prose, with-reasoning vs answer-only);
  - **format/structure** of the shots (plain Q-A, chain-of-thought, dialogue, JSON/bulleted);
  - **persona/voice** demonstrated by the example answers (the #406 personas, now induced by example rather than instruction).
  Hold the *probe* question set fixed across contexts so only the prepended example kind varies. Aim for ~12-20 kinds spanning low-to-high mutual base-model distance.
- **Implant (same as #474 localized arm):** under each ICL context T_i, append the marker ` ※` (id 83399) to the END of the model's own (frozen, on-policy) response; marker-only loss via `MarkerOnlyDataCollator`; **contrastive negatives** = the same probe questions under *other* ICL-example kinds with no marker (per `.claude/rules/contrastive-negatives.md`). Non-saturated regime (sub-epoch / ep1 checkpoint or lighter lr/rank) so the transfer DV keeps dynamic range (the #460/#462 saturation lesson).
- **Predictor (the new extraction point):** base-model residual-stream activation at the **last prompt token of `[ICL examples of kind i] + [user question]`**, contrasted T_i vs T_j (persona-vectors cosine recipe), over the held-out probe-question set. Also compute JS divergence between the base-model output distributions under T_i vs T_j (the cosine-vs-JS contrast that is the headline of the #474 follow-up). Layer sweep {0,5,11,15,21,27} (the #474 follow-up showed L15-L21 localize the signal).
- **DV:** on-policy marker transfer across the ICL-kind cross-eval grid — model writes its own answer under T_j (its prepended example kind + a held-out question), read ΔG = trained − base log P(marker) at the post-response slot; report on-policy emission rate too. Higher = more transfer.
- **Analysis battery (mirror the #474 follow-up):** length-partial AND raw Spearman ρ of cosine and JS vs ΔG; full panel + exclusion of the most-distinct example kinds (the analogue of the stylized-persona exclusion that JS failed in #474); per-quintile monotone-gradient check; base-prior guard (saturation fraction + ρ vs trained log-prob, since saturated panels make ΔG collapse to −b_logprob).
- **Model:** Qwen-2.5-7B-Instruct; marker ` ※` (assert `encode == [83399]`); ≥2 seeds given the conclusion now hinges on geometry details.

## Hypotheses

- **H1.** Base-model cosine (post-ICL+question activation) predicts on-policy marker transfer across different kinds of ICL examples (positive ρ), where JS divergence does not.
- **H2.** The cosine signal holds across the panel — including excluding the most-distinct example kinds — rather than riding on a few outlier kinds (the failure mode JS showed in #474).
- **H3.** Because different example kinds spread the base-model representation widely, the cosine predictor is *cleaner / better powered* here than the instruction-induced contexts of #474 (the #468 examples-beat-descriptions effect, plus more dynamic range).

## Relationship to other tasks

- Parent [#474](https://eps.superkaiba.com/tasks/474): same DV + analysis battery; the single new axis is different-kind ICL-example contexts + post-ICL extraction point.
- Sibling [#488](https://eps.superkaiba.com/tasks/488) (in planning): #488 hardens cosine-vs-JS + the identity-vs-format decomposition on instruction/phrasing contexts; this task tests the example-induced regime with a deliberately wide kind panel. The planner should decide whether to keep them separate or fold the ICL arm into #488.
- [#468](https://eps.superkaiba.com/tasks/468): the examples-beat-descriptions result this builds on.

## Open questions for the planner

- The concrete taxonomy of "kinds" and the final panel (which mix of domain / style / format / persona, and how many) — ground it so the panel spans low-to-high mutual base-model distance.
- K (number of shots per context) and whether to hold example *content* fixed across kinds where possible (to isolate "kind of demonstration" from "what content the model saw" — addresses the #468 content-vs-geometry confound).
- Single vs ≥2 seeds; non-saturated-regime calibration (smoke-test the implant rate so the source isn't pinned at ceiling).

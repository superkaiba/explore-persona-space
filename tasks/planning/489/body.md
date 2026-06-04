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
goal: Test whether base-model cosine/JS distance predicts on-policy marker transfer
  across a UNION panel of in-context-example contexts and instruction-induced (system-prompt/persona/phrasing)
  contexts — including cross-type cells in both directions (ICL <-> instruction) —
  with the predictor read from the residual activation after the context scaffold
  + user question, and whether matched same-identity cross-type pairs (example-pirate
  <-> instruction-pirate) are close in cosine and transfer the marker.
track: experiment
relates_to:
- leak-predictor
---
## Goal

Test whether base-model cosine/JS distance predicts on-policy marker transfer across a UNION panel of in-context-example contexts and instruction-induced (system-prompt/persona/phrasing) contexts — including cross-type cells in both directions (ICL <-> instruction) — with the predictor read from the residual activation after the context scaffold + user question, and whether matched same-identity cross-type pairs (example-pirate <-> instruction-pirate) are close in cosine and transfer the marker.


## Background

The geometry-predicts-transfer line ([#406](https://eps.superkaiba.com/tasks/406) → [#474](https://eps.superkaiba.com/tasks/474)) trains a marker under a context transformation T_i and asks whether a base-model distance between T_i and T_j predicts on-policy marker transfer. The latest follow-up analysis on #474's localized (contrastive-negatives) arm found the load-bearing result: **base-model residual-stream cosine similarity predicts transfer among ordinary non-stylized contexts (length-partial ρ ≈ +0.57, raw +0.52, p < 1e-12) where output-distribution JS divergence is null (ρ ≈ −0.11)**, the signal localizes to mid-late layers (L15-L21), and it predicts the trained marker log-prob directly (+0.65), not the base prior. In that line the cosine vector was extracted at the **last prompt token of (system prompt + user question)** — contexts were system-prompt personas, query-phrasing wraps, format scaffolds, and register rewrites (all instruction-induced).

Separately, [#468](https://eps.superkaiba.com/tasks/468) found that **real in-context examples (not descriptions) are what make the base-model cosine predictor informative** for emergent-misalignment generalization (ρ = 0.66 with K=8 examples; ρ ≈ 0 with description-only personas).

## Motivation

Two combined moves:

1. Add **different KINDS of in-context-example contexts** to the panel, and extract the cosine/JS predictor from the residual activation taken **after the in-context examples and the user question** (the last prompt-token position, where the prompt = [ICL example block] + [user question]).
2. Put the ICL-example contexts **in the same panel as the instruction-induced contexts** (system-prompt personas + phrasing/format/register transforms), and run the **full cross-eval grid including cross-type cells**, so the experiment measures transfer/leakage *across induction mechanisms*: ICL-example → system-prompt/persona and system-prompt/persona → ICL-example, in both directions, not only within each type.

The headline question the cross-type design buys: does a marker bound to an *example-induced* context transfer to an *instruction-induced* context (and vice versa), and does the base-model representational distance between the two predict it? If example- and instruction-induced versions of the "same" persona sit close in the base-model representation and the marker flows between them, that's strong evidence the predictor captures an induction-mechanism-independent identity axis — and it directly operationalizes #468's examples-beat-descriptions result. Different example *kinds* also spread the base-model representation far wider than #474's tightly-clustered phrasing contexts (77% of non-stylized pairs sat above cosine-sim 0.90), giving the cosine-vs-JS test real dynamic range across the whole panel.

## Design (single new axis vs #474: union panel of ICL-example kinds + instruction contexts; full cross-type grid; predictor read after ICL+question)

- **Context panel = UNION of two induction types, cross-evaluated fully.**
  - **In-context-example contexts (the new type):** a diverse set of K-shot example blocks that differ in KIND — vary the *type* of few-shot examples prepended before the user question. Candidate axes of "kind" (mix several so the panel spans the representation space): domain/topic (math, coding, history, medical, …); answer style (terse/verbose, formal/casual, list/prose, reasoning/answer-only); format/structure (plain Q-A, chain-of-thought, dialogue, JSON/bulleted); persona/voice demonstrated by the example answers (the #406 personas, now induced by example).
  - **Instruction-induced contexts (reused):** the #406/#474 system-prompt personas + phrasing/format/register transforms.
  - **Matched cross-type pairs by design:** include example-induced AND instruction-induced versions of the *same* personas (e.g. ICL-pirate and system-prompt-pirate) so the grid has same-identity / different-induction cells — the cleanest test of cross-mechanism transfer.
  - The cross-eval is the **full grid over the union** (~within-ICL, within-instruction, ICL→instruction, instruction→ICL). Hold the *probe* question set fixed across all contexts so only the context varies. Target a panel size that keeps the grid manageable (planner to size; #406's grid was 16×16=240, the union will be larger).
- **Implant (same as #474 localized arm):** under each context T_i, append the marker ` ※` (id 83399) to the END of the model's own (frozen, on-policy) response; marker-only loss via `MarkerOnlyDataCollator`; **contrastive negatives** = the same probe questions under *other* contexts (of either type) with no marker (per `.claude/rules/contrastive-negatives.md`). Non-saturated regime (sub-epoch / ep1 checkpoint or lighter lr/rank) so the transfer DV keeps dynamic range (the #460/#462 saturation lesson).
- **Predictor:** base-model residual-stream activation at the **last prompt token** of each context's `[scaffold] + [user question]` (for ICL contexts the scaffold is the example block; for instruction contexts it is the system prompt / wrap), contrasted T_i vs T_j (persona-vectors cosine recipe), over the held-out probe set. Also compute JS divergence between the base-model output distributions under T_i vs T_j (the cosine-vs-JS contrast that is the #474 follow-up headline). Layer sweep {0,5,11,15,21,27}. The predictor is defined identically for within-type and cross-type pairs, so cross-type cells get a base-model distance too.
- **DV:** on-policy marker transfer across the full grid — model writes its own answer under T_j, read ΔG = trained − base log P(marker) at the post-response slot; report on-policy emission rate too. Higher = more transfer.
- **Analysis battery (mirror the #474 follow-up, sliced by type):** length-partial AND raw Spearman ρ of cosine and JS vs ΔG, computed on the **full panel** and on the three sub-panels **within-ICL / within-instruction / cross-type**; per-quintile monotone-gradient check; exclusion of the most-distinct kinds; base-prior guard (saturation fraction + ρ vs trained log-prob). A dedicated look at the **matched same-identity cross-type cells** (ICL-pirate ↔ instruction-pirate): are they close in cosine, and does the marker transfer?
- **Model:** Qwen-2.5-7B-Instruct; marker ` ※` (assert `encode == [83399]`); ≥2 seeds given the conclusion now hinges on geometry details.

## Hypotheses

- **H1.** Base-model cosine (post-context activation) predicts on-policy marker transfer across the full union panel (positive ρ), where JS divergence does not.
- **H2.** The cosine signal holds across the panel — including excluding the most-distinct contexts — rather than riding on a few outliers (the failure mode JS showed in #474).
- **H3.** Different example kinds spread the base-model representation widely, so the cosine predictor is *cleaner / better powered* here than the instruction-only contexts of #474 (the #468 examples-beat-descriptions effect plus more dynamic range).
- **H4 (cross-type, the new one).** Marker transfer occurs *across induction mechanisms* (ICL ↔ instruction) and is predicted by base-model representational distance; matched same-identity cross-type pairs (example-pirate ↔ instruction-pirate) sit close in cosine and show high transfer, while mismatched cross-type pairs sit far and transfer little.

## Relationship to other tasks

- Parent [#474](https://eps.superkaiba.com/tasks/474): same DV + analysis battery; new axes are (a) ICL-example contexts read after ICL+question and (b) the cross-type union grid.
- Sibling [#488](https://eps.superkaiba.com/tasks/488) (in planning): #488 hardens cosine-vs-JS + the identity-vs-format decomposition on instruction/phrasing contexts; this task adds the example-induced type and the cross-mechanism transfer test. The planner should decide whether to keep them separate or fold the ICL/cross-type arm into #488.
- [#468](https://eps.superkaiba.com/tasks/468): the examples-beat-descriptions result the cross-type design operationalizes.

## Open questions for the planner

- The concrete taxonomy of "kinds" and the final union panel (which ICL kinds + which instruction contexts, how many of each, which matched same-identity pairs) — ground it so the panel spans low-to-high mutual base-model distance and the grid stays tractable.
- K (number of shots per ICL context) and whether to hold example *content* fixed across kinds where possible (to isolate "kind of demonstration" from "what content the model saw" — addresses the #468 content-vs-geometry confound).
- Single vs ≥2 seeds; non-saturated-regime calibration (smoke-test the implant rate so the source isn't pinned at ceiling).

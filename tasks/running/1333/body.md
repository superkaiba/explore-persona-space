---
title: 'Factory marker organism: marker behavior through the #1090 pipeline (spectrum
  anchor for #1315)'
kind: experiment
tags: []
created_at: '2026-07-15T07:16:02Z'
has_clean_result: false
parent_id: 1090
origin_prompt: 'yes [produce a marker cell through the #1090 factory pipeline, same
  contexts, marker carve-out recipe, for apples-to-apples spectrum comparison marker->impolite->formatting]'
workflow: v1
goal: 'Build the marker anchor for the organism-geometry spectrum via the FULL #1112
  design on the factory marker behavior (token id 83399): {LoRA, full fine-tune} x
  {positive-only, contrastive} 2x2 at MATCHED marker install (mirroring #1112 sycophancy
  + #1315 impolite), plus a LoRA extension across the 4 #1090 contexts for install/expression/leakage
  breadth; geometry DVs mirror #1112 (rank-k@90/participation-ratio/alignment/magnitude
  + the mandatory teacher-forced shared-text control) plus the three-space on-policy
  marker log-P install DV; dedicated marker recipe (marker+EOT loss, lr<=5e-6, band-stop,
  contrastive-negative EOS slot), de-saturated anchor.'
relates_to:
- implant-which-behaviors
- leak-behavior-vs-marker
---
## Goal

Build the marker anchor for the organism-geometry spectrum via the FULL #1112 design on the factory marker behavior (token id 83399): {LoRA, full fine-tune} x {positive-only, contrastive} 2x2 at MATCHED marker install (mirroring #1112 sycophancy + #1315 impolite), plus a LoRA extension across the 4 #1090 contexts for install/expression/leakage breadth; geometry DVs mirror #1112 (rank-k@90/participation-ratio/alignment/magnitude + the mandatory teacher-forced shared-text control) plus the three-space on-policy marker log-P install DV; dedicated marker recipe (marker+EOT loss, lr<=5e-6, band-stop, contrastive-negative EOS slot), de-saturated anchor.

## Design axes (the "what about marker" answer — YES to all)

- **Negatives:** BOTH contrastive (with negatives) AND positive-only. (Was already in scope.)
- **Training method:** BOTH LoRA AND full fine-tune, at MATCHED marker install — the axis that was missing; it is the core of #1112 and the reason a spectrum comparison exists. Train fresh full-FT marker twins on the marker mix, exactly as #1112 trained full-FT sycophancy twins.
- **Matched install (critical):** compare geometry at equal marker-implant strength, NOT equal dose — #1112's central methodological point; #1112's own marker pair was install-confounded (+1.6 vs +6.3 nat) and low-signal on the LoRA side, so THIS task's whole value is the clean matched version.
- **Contexts:** the geometry 2x2 at the persona context (to match #1112); the LoRA arm extended to WildChat/ICL/bare for install/expression/leakage breadth (tractability permitting).

## Geometry DVs (mirror #1112 exactly, so results compose)

- Residual-stream activation shift (trained-minus-base, per layer): rank-k@90, participation-ratio, direction alignment to the marker read-out, magnitude (mean-shift norm).
- **Mapping arms (standing prefix-and-context rule):** capture pooling runs BOTH mapping arms plus the response arm, exactly as #1112 did — prefix-based (the prefix = everything before the user query, i.e. the system persona), context-based (prefix + user question), and response (the model's own generation) — all three reported.
- The TEXT-IDENTITY control is mandatory: teacher-forced shared-text re-capture (one shared base-generated text per row), because #1112 showed most of the apparent "diffuse" signature is text-variation artifact (rank-k@90 collapsed 66-74 -> 27-35 on shared text) — the marker read must run this control or it is not comparable to the sycophancy read.
- PLUS the marker-specific install DV: on-policy log P(marker) at the END of the model's own response, trained-base in all THREE spaces (log-prob primary / marker logit incl. EOS margin secondary / probability sanity), four-float-per-slot storage, NOT judge-scored.

## Recipe + provenance

- Use the DEDICATED marker construction, NOT content-behavior datagen. Planner MUST read `.claude/rules/marker-training-recipe.md` + `.claude/rules/marker-leakage-measurement.md` + `.claude/rules/contrastive-negatives.md` IN FULL before grounding any hyperparameter. Recipe identity (planner grounds exact values): marker token id 83399 (assert in-process); marker + end-of-turn loss mask; LR is the over/under dial, clean window lr <=5e-6, strength via STEPS not rank (#530); marker-gated band-stop default; contrastive negatives train the post-response EOS slot (#474). Watch saturation (#448) — de-saturate the anchor so bystander log-prob keeps headroom (a saturated marker kills the geometry read's dynamic range).
- origin: user chat (PM session), 2026-07-15: "for marker are we running with negatives and without + LoRA vs FT + etc.?" — YES, the full 2x2 + matched install + the #1112 text-identity control.
- Feeds [#1315](https://eps.superkaiba.com/tasks/1315) / composes with [#1112](https://eps.superkaiba.com/tasks/1112): marker is the local-lexical spectrum anchor (marker -> sycophancy -> impolite -> formatting). Legacy marker organisms (#508/#514/#472) are a DIFFERENT recipe/harness — this task is the same-harness clean version.

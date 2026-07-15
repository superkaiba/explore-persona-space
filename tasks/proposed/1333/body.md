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
goal: 'Produce a marker model organism through the #1090 artifacts-factory pipeline
  (registered marker programmatic carve-out) across the same 4 contexts (persona/WildChat/ICL/bare)
  x 2 regimes (contrastive/positive-only) as the content behaviors, using the dedicated
  marker recipe (marker+EOT loss, lr<=5e-6, band-stop, token id 83399, contrastive
  negatives, three-space log-P DV, NOT judge-scored), and characterize install/on-policy-expression/bystander-leakage
  per context, so the factory organism set spans marker->impolite->formatting under
  one construction for the #1315 geometry comparison.'
---
## Goal

Produce a marker model organism through the #1090 artifacts-factory pipeline (registered marker programmatic carve-out) across the same 4 contexts (persona/WildChat/ICL/bare) x 2 regimes (contrastive/positive-only) as the content behaviors, using the dedicated marker recipe (marker+EOT loss, lr<=5e-6, band-stop, token id 83399, contrastive negatives, three-space log-P DV, NOT judge-scored), and characterize install/on-policy-expression/bystander-leakage per context, so the factory organism set spans marker->impolite->formatting under one construction for the #1315 geometry comparison.

## Provenance

- origin: user chat (PM session), 2026-07-15. The factory registers `marker` as a producible programmatic behavior but no factory-line run (#906/#1074/#1090) ever instantiated it — this fills that gap so marker anchors the spectrum WITHIN the factory pipeline.
- Recipe: use the DEDICATED marker construction, NOT the content-behavior datagen. Planner MUST read `.claude/rules/marker-training-recipe.md` + `.claude/rules/marker-leakage-measurement.md` + `.claude/rules/contrastive-negatives.md` IN FULL before grounding any hyperparameter. Recipe identity (planner grounds exact values): marker ` ※` token id 83399 (assert in-process); marker + end-of-turn loss mask (positives `{ ※, <|im_end|>, \n}`, negatives `{<|im_end|>, \n}`); LR is the over/under dial — clean window lr ≤5e-6, strength via STEPS not rank (#530); marker-gated band-stop callback default; contrastive negatives with the post-response EOS slot trained (#474). DV = on-policy `log P(marker)` at the END of the model's OWN response, trained−base in all THREE spaces (log-prob primary / marker logit incl. EOS margin secondary / probability sanity), four-float-per-slot storage; NOT LLM-judged (marker is the programmatic carve-out). Watch saturation (#448) — gate the anchor so bystander log-prob keeps headroom.
- Contexts + regimes MIRROR #1090 (persona/WildChat/ICL/bare × contrastive/positive-only) so the marker cell is apples-to-apples with the impolite/formatting cells; reuse #1090's context definitions + negative panels where the marker carve-out allows.
- Feeds [#1315](https://eps.superkaiba.com/tasks/1315) (impolite-organism geometry): once the marker organisms exist, the geometry read extends to marker as the local-lexical spectrum anchor (marker → impolite → formatting). Legacy marker organisms exist (#508/#514/#472/i385/i398/i432/i466) but on the DIFFERENT legacy recipe/eval harness — this task's value is the SAME-factory construction for clean comparison, not new marker science.

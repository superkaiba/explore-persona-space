---
title: '[Proposed] Eval good/bad cases as a behavior vector, then find similar persona-space
  directions'
kind: experiment
tags: []
created_at: '2026-05-29T04:51:18Z'
has_clean_result: false
---
Captured from Todoist via my-goat autoprocess. Testable experimental idea.

**Idea (verbatim):** "You have an eval where it does well in some cases and badly in some cases. Use that as a vector — find behavior vector similar to that."

**Interpretation / pipeline:**
1. Take an eval where the model's per-item performance varies (some items handled well, some badly).
2. Form a contrast set: {good-case activations} vs {bad-case activations}. Difference of means = a candidate "behavior vector" (CAA-style; see companion methods capture: model-written evals + contrastive activation addition).
3. Search for existing directions (persona vectors, the assistant axis, SAE features) that are most cosine-similar to this empirical behavior vector — i.e., explain the good/bad split in terms of known persona-space structure.

**Why it matters:** Turns an eval's success/failure pattern directly into a causal direction, then grounds it in the persona-space the project already characterizes. Bridges behavioral evals and mechanistic directions — directly relevant to the persona-vector / assistant-axis line.

**Predictions / tests:**
- If the good/bad split is persona-driven, the empirical vector should align with a known persona direction; adding it should flip more eval items.
- Causal check: steer along the vector and re-run the eval — does performance on bad cases improve (and good cases degrade)?

**Open questions:**
- Which eval to seed with (44-prompt misalignment rubric? a capability eval like ARC-C / MMLU-Pro)?
- Token position / layer for extracting case-level activations.
- "Find similar behavior vector" — similar to what library: persona vectors, SAE features, or the assistant axis?

**Done condition:** a concrete experiment spec with the seed eval chosen, extraction recipe, and the similarity-search target set.

**Links:** research_ideas.md §I (Persona Geometry, persona-vector decomposition), §7 (User Modeling). Companion to: define-behavior note, model-written-evals + CAA methods primer.

Source: my-goat queue file 2026-05-28T19-00-11_todoist-6gjhqF66PrcH394M_you-have-eval-where-it-does-well-in-some-cases-and.md (Todoist id 6gjhqF66PrcH394M).

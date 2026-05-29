---
title: '[Proposed] Methods primer: model-written evaluations + contrastive activation
  addition (CAA)'
kind: infra
tags: []
created_at: '2026-05-29T04:51:14Z'
has_clean_result: false
---
Captured from Todoist via my-goat autoprocess. Methods primer / reading.

**Topic:** Model-written evaluations (Perez et al. 2022, arXiv 2212.09251) + contrastive activation addition (CAA; Rimsky et al. 2023, arXiv 2312.06681). Method foundation for the "behavior vector" pipeline (companion capture: eval good/bad cases → vector).

**Why it matters:** The two methods compose into a pipeline Thomas is sketching:
1. **Model-written evals** — cheaply generate large, targeted eval sets for a behavior/trait without hand-writing items.
2. **CAA** — take contrastive pairs (positive vs negative behavior examples), difference their mean activations to get a steering vector, add/subtract it at inference to causally move the behavior.

Together: auto-generate an eval for a behavior, split into well-handled vs poorly-handled cases, use the contrast as a CAA-style behavior vector. This is the method backbone for the companion experiment.

**Open questions:**
- Layer choice + token position for CAA on Qwen-2.5-7B / Gemma-2 9B (project default models)?
- Do model-written eval items introduce a generator-model bias that contaminates the contrast?
- Relationship between a CAA behavior vector and the existing persona vectors / assistant axis — same subspace or orthogonal?

**Done condition:** short methods note (how to run model-written eval generation + CAA in this repo's stack) that the behavior-vector experiment can build on.

**Links:** research_ideas.md §I (Persona Geometry), §7 (User Modeling, CAA-relevant). Companion to define-behavior note and the eval good/bad cases → behavior-vector experiment.

Source: my-goat queue file 2026-05-28T19-00-09_todoist-6gjhpg3qrVh94mFM_model-written-evaluations.md (Todoist id 6gjhpg3qrVh94mFM).

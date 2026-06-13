---
title: '[Proposed] How to define ''behavior'' (definitional groundwork for behavior
  vectors)'
kind: infra
tags: []
created_at: '2026-05-29T04:51:11Z'
has_clean_result: false
---
Captured from Todoist via my-goat autoprocess. Conceptual / definitional task.

**Question:** How to define "behavior" precisely, in a way that supports a "behavior vector" (cf. companion capture #eval-good-bad-cases-as-vector). Definitional groundwork for the persona/steering line.

**Why it matters:** The project leans on "persona vectors," "the assistant axis," and (proposed) "behavior vectors." These are used loosely. A crisp operational definition of behavior determines: what gets contrasted to extract a direction (CAA-style), what an eval measures, and whether "behavior" is distinguishable from "persona" / "trait" / "capability."

**Candidate definitions to compare:**
- Behavior = distribution over outputs conditioned on a prompt class (functional/behavioral).
- Behavior = an elicitable direction in activation space whose addition shifts outputs along an axis (mechanistic, CAA / persona-vector style).
- Behavior = a measurable score on an eval rubric (operational, ties to the 44-prompt misalignment rubric).

**Open questions:**
- Is "behavior" the same granularity as a persona trait, or finer (a single elicited tendency)?
- How does it relate to the persona-vector decomposition (identity / style / capability) in §I task #1?
- Does the definition need to be eval-anchored to be useful for the "good/bad cases → vector" pipeline?

**Done condition:** 1-page note settling on a working definition, cross-linked to the behavior-vector experiment and the persona-vector decomposition task.

**Links:** research_ideas.md §I (Persona Geometry, persona-vector decomposition), §7 (User Modeling). Companion to the eval-good/bad-cases → behavior-vector capture.

Source: my-goat queue file 2026-05-28T19-00-08_todoist-6gjhpXPPhhCVxwcv_how-to-define-behavior.md (Todoist id 6gjhpXPPhhCVxwcv).

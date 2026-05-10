---
name: Neutral-prompt axis-conflation
description: Steering plans that use "You are a helpful assistant." as the neutral prompt while one of the evaluated personas IS the helpful_assistant persona create a confound where the "neutral" condition is non-uniformly close to each persona, partially recovering the prompted ordering before any steering vector is applied
type: feedback
---

When critiquing activation-steering plans where the "neutral" or control system prompt is asserted to be persona-free, check whether that string is a member of the persona set under test. In the explore-persona-space project, `src/explore_persona_space/personas.py:29` defines `ASSISTANT_PROMPT = "You are a helpful assistant."` and helpful_assistant is one of the 12 evaluated personas. Issue #267's plan used this exact string as the "neutral" prompt for steering — making `helpful_assistant`'s coeff=0 cell mathematically identical to its prompted baseline (cosine to self = 1.0), and giving every other persona a non-uniform fraction of the helpful_assistant centroid for free, weighted by their cos-to-assistant. This means:

- The cosine→source-rate H2 regression is partially tautological (high cos-to-assistant personas receive more "self-direction" leakage from the neutral prompt itself before any centroid is added).
- The H1 ordering ρ partially measures how cos-to-assistant biases the leakage rather than how the centroid steers.

**Why:** Round 2 alternatives memory (`feedback_alternatives_lens_round2.md`) already flags axis-conflation as a recurring blind spot; the variant here is "control axis (neutral prompt) overlaps with the evaluation axis (one of the personas)." This specific conflation slipped past plan v2 in #267 despite multiple critic rounds.

**How to apply:** When a steering / activation-intervention plan defines a "neutral" or "control" system prompt as a literal string, grep `personas.py` for that string before approving. If it matches any persona (especially helpful_assistant, qwen_default, or any "assistant"-like persona): require the planner either (a) use the empty system role / a non-persona instruction outside the centering set, or (b) drop the matching persona from the headline N, or (c) re-extract centroids on a centering set that excludes both the matching persona and any structural near-duplicate (e.g., qwen_default at cos +0.714).

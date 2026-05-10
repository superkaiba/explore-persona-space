---
name: Neutral system prompt must not be on the centering axis
description: Setting the "neutral" system prompt to "You are a helpful assistant." conflates prompt-removal with the helpful_assistant centroid axis (cos ≈ 1.0); for any plan whose centering set includes helpful_assistant or qwen_default, pick a content-free instruction outside the set OR drop those personas from the headline N.
type: feedback
---

When designing an activation-steering experiment that "replaces the persona prompt with a neutral one," **never use `"You are a helpful assistant."` as the neutral prompt if `helpful_assistant` is in the centering set.** That string is `personas.py::ASSISTANT_PROMPT` and was used to build the `assistant` centroid that anchors the cosine axis; its centroid sits at cos ≈ 1.0 to itself and cos ≈ +0.714 to `qwen_default`. So at coeff=0 the "steered" condition is mathematically the **prompted** condition for `helpful_assistant`, and `qwen_default` gets significant elicitation for free even before steering — partially tautologizing any cosine→source-rate gradient.

**Why:** alternatives critic (round 1, issue #267 plan v2) flagged this as BLOCKER B1 with smoking gun `regression_results.json::cosines_to_assistant.layer_20[helpful_assistant] = 0.99999`. The fix that round was to switch the neutral prompt to a content-free instruction (`"Answer the user's question."` or empty system role) AND drop `helpful_assistant` + `qwen_default` from the headline-N if they're structurally too close to the neutral axis.

**How to apply:** for any future steering plan whose centering set or N includes `helpful_assistant` or `qwen_default`, run a pre-design check: compute cos(centroid[neutral_prompt_persona], centroid[each_persona_in_N]); if any |cos| > 0.5, the neutral prompt is on-axis and biases the headline test. Either pick a different neutral prompt or trim the N. State the chosen mitigation explicitly in the plan; do not silently keep the confound.

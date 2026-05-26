---
name: panel-family-clustering-vs-axis-conflation
description: Even when a "neutral" baseline prompt is literally distinct from every panel persona, a single semantic FAMILY of panel members (e.g., 11 assistant-flavored short prompts) can dominate the rank correlation by clustering tightly on the predictor axis AND on the target axis — making the headline ρ trace family-membership, not the named geometric axis
metadata:
  type: feedback
---

When a plan picks a "content-free" baseline string for a JS-from-baseline / cosine-from-baseline predictor and asserts the baseline is "off-axis" because it is literally distinct from every panel persona prompt, that is necessary but NOT sufficient. The structural axis-conflation can still bite via panel COMPOSITION:

- If the 48 panel includes 10+ "helpful-assistant family" personas (`helpful_assistant`, `i_am_helpful`, `chat_assistant`, `ai_assistant`, `virtual_assistant`, `chatbot`, `friendly_ai`, `smart_helper`, `ai_tool`, `ai`, `qwen_default`) — most at short prompt length, most at similar source rate — AND the baseline is itself a short instruction-form string in the same register ("Answer the user's question.") — then JS-from-baseline of the 11 family members will be uniformly low (similar short distributions producing similar next-token output) AND their source rates also cluster (mean 0.20 vs 0.29 for the other 37 personas on the #380 panel).
- The headline rank correlation can then track "is this persona a helpful-family member?" rather than the claimed geometric axis. Removing the family by leave-one-cohort-out or leave-family-out can flip the verdict.

**Why:** Issue #380 (Thread A JS-from-baseline → source-rate, parent #340) chose `"Answer the user's question."` as the baseline specifically to avoid the literal `helpful_assistant` axis-conflation flagged in `feedback_neutral_prompt_axis_conflation`. That avoided ONE form of conflation (the baseline string equalling a panel prompt) but the panel STILL contained 11 assistant-family members clustered at low predicted JS and low rate. A positive raw ρ on this design could be a family-composition artifact rather than a geometric finding.

**How to apply:** For any plan that:
1. Computes a predictor from each persona to a single anchor point (baseline / centroid / etc.), AND
2. The panel contains ≥5 members of one obvious SEMANTIC FAMILY that share register with the baseline,

require the plan to PRE-COMMIT to ONE of:
- (a) report leave-family-out partial-ρ as a robustness columns alongside the headline (analyst-side requirement; cheap),
- (b) report the scatter colored by family membership in the hero figure (makes the cluster visible to the reader),
- (c) drop the family from the headline N entirely (most conservative; reduces N).

If the plan offers a panel-composition-free secondary predictor (mean pairwise JS in #380 is one — every persona is referenced against the other 47, so no single anchor point), that materially reduces the concern, and the headline can be defined as the SECONDARY predictor with the anchor-based primary as a comparison.

For a critic at pre-execution, this is RECOVERABLE (analyzer can weigh from per-persona predictor JSON + scatter), not fatal — so it lands in "Concerns the analyzer should attend to" not a REVISE blocker. But the planner should be encouraged to pre-add (a) or (b) cheaply at implementation time.

Companion to [[Neutral-prompt axis-conflation]] (literal-match form) and [[Alternatives lens round 2]] (axis-conflation as a recurring blind spot category).

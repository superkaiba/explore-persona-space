---
name: panel-family-clustering-vs-axis-conflation
description: Even with a literally-distinct "neutral" baseline, a single semantic FAMILY of panel members (e.g. 11 assistant-flavored prompts) can cluster on both the predictor axis AND the target axis, making the headline ρ trace family membership
metadata:
  type: feedback
---

When a plan computes a predictor from each persona to a single anchor point (baseline / centroid) and asserts the baseline is "off-axis" because it is literally distinct from every panel prompt, that is necessary but NOT sufficient — panel COMPOSITION can still conflate the axis.

**Why (#380, JS-from-baseline → source-rate, parent #340):** the panel contained 11 helpful-assistant-family personas, most short and similar in register to the baseline "Answer the user's question." — their JS-from-baseline is uniformly low AND their source rates cluster (mean 0.20 vs 0.29 for the other 37). The headline rank ρ can then track "is this persona a helpful-family member?" rather than the claimed geometric axis; leave-family-out can flip the verdict. The plan had deliberately avoided the LITERAL conflation (baseline string ≠ any panel prompt, per feedback_neutral_prompt_axis_conflation) and still carried this compositional form.

**How to apply:** when (1) the predictor references a single anchor point AND (2) the panel has ≥5 members of one semantic family sharing register with the anchor, require the plan to pre-commit to ONE of: (a) leave-family-out partial-ρ as a robustness column; (b) hero scatter colored by family membership; (c) drop the family from the headline N. A panel-composition-free secondary predictor (e.g. mean pairwise JS — every persona referenced against all others) materially reduces the concern and can be promoted to the headline. RECOVERABLE (Concern, not REVISE) — the analyzer can weigh it from per-persona predictor JSON + scatter — but encourage (a)/(b) pre-added cheaply. Companions: [[Neutral-prompt axis-conflation]] (literal-match form), [[Alternatives lens round 2]].

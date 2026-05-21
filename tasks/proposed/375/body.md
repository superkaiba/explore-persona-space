---
title: Natural marker leakage via assistant-axis persona drift (no persona prompting)
kind: experiment
application: detect
tags: []
created_at: '2026-05-21T00:42:50Z'
has_clean_result: false
---
## Idea

Check natural marker leakage through persona drift along the assistant axis, instead of explicit persona prompting at eval time.

Persona prompting is somewhat unrealistic as a deployment scenario — real misalignment leakage would manifest as the model naturally drifting toward a trained persona under benign prompts, not as a user explicitly invoking the persona by name.

## Why this experiment

- **Decision this changes:** It provides a concrete motivation for the project whereas before it was less concrete why we care about persona prompts.
- **Expected outcome + branches:** If the marker leaks through persona drift, use this as motivation that these phenomena occur naturally; if it doesn't leak, we might have to abandon persona prompting because it is artificial.
- **Application:** detect — serves as motivation for the entire project.

## Sketch

- Train as usual (existing source-persona coupling pipelines).
- At eval, instead of prompting "as Persona X", probe along the assistant axis (e.g., steer the residual stream toward the assistant direction by some alpha, or sample from neutral assistant prompts and measure marker firing rate).
- Compare against the persona-prompted leakage rate as the baseline.

## Open questions

- Which axis exactly — assistant identity axis, or the persona axis itself with zero steering?
- Free-generation marker rate vs. forced-choice probe?
- How to disentangle from generic capability drift?

## Status

Pure capture — not for immediate execution.

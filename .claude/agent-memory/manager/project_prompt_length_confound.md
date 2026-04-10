---
name: Prompt Length Confound
description: System prompt length strongly predicts marker leakage (r=-0.74); must control in all persona transfer experiments
type: project
---

Proximity Transfer experiment (2026-04-10) discovered that system prompt length is the strongest predictor of marker leakage among held-out personas (r=-0.74, p=0.006), stronger than cosine similarity to either the target or the assistant (both non-significant at p>0.09 on held-out data).

**Why:** Short, generic prompts ("You are a helpful assistant." = 28 chars) fail to anchor model behavior, making the model default to training distribution. The assistant's anomalously high leakage (68%) may be a "short prompt" effect, not a "geometric proximity" effect.

**How to apply:** All future persona transfer experiments MUST either (a) match system prompt lengths across conditions, or (b) include prompt length as a covariate. The cos(assistant) > cos(P*) correlation was retracted because it was driven by the tautological assistant data point. Multiple regression (cos + length + negative_set) explains 80% of variance with assistant not an outlier (Z=1.03). The prompt-length-controlled follow-up is the critical next experiment for Aim 2-3.

---
title: Does the JS leakage predictor fail on conditional personas? (slice-aware vs
  average divergence)
kind: experiment
tags: []
created_at: '2026-06-02T08:36:44Z'
has_clean_result: false
parent_id: 404
goal: 'Eval-only test (no training of the conditional behavior): do the standard leakage
  predictors - output-distribution JS divergence and persona-vector cosine similarity
  at the system-prompt boundary - fail to anticipate a conditional/triggered behavior
  because they average over or precede the trigger, and does slice-resolving the divergence
  (JS on trigger vs non-trigger prompts) predict where the marker behavior actually
  shifts when the boundary cosine and averaged JS do not?'
---
## Goal

Eval-only test (no training of the conditional behavior): do the standard leakage predictors - output-distribution JS divergence and persona-vector cosine similarity at the system-prompt boundary - fail to anticipate a conditional/triggered behavior because they average over (or precede) the trigger? And does slice-resolving the divergence (JS on trigger vs non-trigger prompts) predict where the marker behavior actually shifts, when the system-prompt-boundary cosine and the averaged JS do not?

**Open questions:** `docs/open_questions.md` §1.2 (`q:spec-kl-probe-set`), §3.1 (`q:leak-predictor`, both JS and cosine predictors), §3.7 (`q:leak-to-default`). **Related:** #404 / #458 (JS/cosine leakage-predictor line + canonical metric defs), #448 / #456 (on-policy marker-leakage rig reused below), #161 (Spanish+English), #446 (realistic-settings scoping).

## Motivation

The leakage predictors in this project summarize the difference between two personas S and S′ as a single quantity - JS divergence of output distributions averaged over a probe set U, or cosine similarity of persona vectors at the end of the system prompt - and predict leakage from it (closer = more leakage). Both summaries are blind to a behavior concentrated on a rare input slice:

- The **averaged JS** conflates *how different* S and S′ are with *how often* the probe set lets them differ. A conditional S′ (= S, but Spanish on restaurant queries) is identical to S almost everywhere, so the average JS sits near zero.
- The **system-prompt-boundary cosine** is measured at the last system-prompt token, *before any user message*. A behavior gated on the user message content cannot move that activation at all - the cosine is structurally incapable of seeing a conditional behavior, because the condition lives in the user turn that comes later.

So both standard predictors read "S ≈ S′, expect full leakage," while a real distinct behavior is sitting on a rare slice U′. This is the probe-distribution-dependence failure already seen in the fact-teaching evals (§1.2), and the safety-relevant case: a persona misaligned only on a rare trigger passes an averaged eval. The point is to show where both predictors break and that slice-resolving the divergence fixes it.

## What we measure (all eval-only, forward passes on an existing checkpoint)

Reuse the #448/#456 on-policy marker rig. Source persona S already has the marker ` ※` (id 83399) trained in (#456: emits on ~90% of its own answers). Define S′ = S + "respond normally but in Spanish iff the user message is about restaurant recommendations." Probe set split into two slices: **non-trigger** (normal questions) and **trigger** (restaurant-recommendation questions).

1. **Cosine predictor (persona vectors at the system-prompt boundary).** Difference-of-means persona vector at the last system-prompt token, cosine between S and S′. Layer sweep {7, 14, 21, 27}, report layer 21 + sweep. This is one number per persona pair, input-independent. Expectation: ≈ 1 (the conditional clause barely moves it), i.e. the predictor says "same persona."
2. **JS predictor, slice-resolved.** Sequence-level (Rao-Blackwellized) JS divergence of output distributions between S and S′, computed separately on the **non-trigger slice** vs the **trigger slice**. Expectation: non-trigger ≈ 0 (and ≈ the generic average), trigger large.
3. **Marker outcome, slice-resolved.** On-policy marker leakage log P(` ※`) (trained − base), under {S, S′} × {non-trigger, trigger}. The discriminator is the **S′ × trigger** cell: does the marker still leak on the restaurant slice, where S′ locally diverges?

## The test

- The cosine (≈ 1) and the non-trigger / averaged JS (≈ 0) both predict S and S′ are interchangeable → marker should leak everywhere, including the trigger slice.
- The trigger-slice JS (large) predicts leakage drops there.
- Adjudicate by the S′ × trigger marker log-prob, and correlate the marker shift with each predictor. The claim to establish: slice-resolved JS predicts where the marker behavior changes; the system-prompt-boundary cosine and the averaged JS do not.

## Controls

- **S × trigger slice** (topic control): does the marker drop on restaurant questions even under plain S? It should not (S is English everywhere) - isolates "restaurant topic itself suppresses ※" from the conditional effect.
- **Unconditional "always Spanish" persona** (text-artifact control): separates "Spanish output tokens suppress ※" from "slice divergence suppresses ※".

## Step A - premise check (gate, ~free)

Before the full measurement: (1) confirm the source checkpoint emits the marker on-policy under S; (2) confirm the model prompted with S′ actually speaks Spanish on restaurant queries and stays normal elsewhere (if it ignores the conditional instruction, S′ doesn't diverge - strengthen the trigger or few-shot); (3) confirm the averages behave as assumed (cosine ≈ 1, non-trigger JS ≈ 0). If any fails, fix the construction or stop before the full run.

## Out of scope / possible follow-ups

- **Behavioral (training) version.** This issue uses the marker as a leakage tracer (clean, but not a behavior in the §225 sense). Actually training the Spanish-on-restaurant behavior into a persona and measuring whether *it* leaks is a separate, heavier issue.
- **Slice-resolved cosine.** A persona-vector cosine taken over each model's own response tokens (recipe (b)) on the trigger vs non-trigger slice would test whether slice-resolving rescues cosine the way it rescues JS. Cheap add if the activation hooks are already in place.
- **Trigger-frequency sweep.** Vary how often the restaurant slice appears in U to trace how fast the averaged predictors go blind.

## Notes

- Why re-slicing OLD runs doesn't work: already-trained leakage cells compare globally-distinct personas (evil, comedian, ...) that differ diffusely, so their averaged JS already ≈ per-slice JS - no near-zero-average + concentrated-slice regime. This experiment constructs that regime via the conditional S′ and measures it on a fresh eval of the existing checkpoint.
- S′ is structurally a **semantic backdoor** (trigger = topic, not a token); connects to the trigger/backdoor results (#276; Apps 1/2/6) and leak-to-default (§3.7).
- Single-variable discipline: the conditional-vs-unconditional structure of S′ is the variable; marker (` ※`, id 83399), the on-policy log-prob DV, JS (sequence-level Rao-Blackwellized) and cosine (difference-of-means) estimators all follow the canonical #448/#456/#458 recipe (CLAUDE.md persona-distance definitions).

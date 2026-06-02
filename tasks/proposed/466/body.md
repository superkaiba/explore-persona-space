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

Eval-only test (no training of the conditional behavior): do the standard leakage predictors - output-distribution JS divergence and persona-vector cosine similarity at the system-prompt boundary - fail to anticipate a conditional/triggered behavior because they average over or precede the trigger, and does slice-resolving the divergence (JS on trigger vs non-trigger prompts) predict where the marker behavior actually shifts when the boundary cosine and averaged JS do not?

**Open questions:** `docs/open_questions.md` §1.2 (`q:spec-kl-probe-set`, does the divergence predictor depend on which probe questions you use), §3.1 (`q:leak-predictor`), §3.7 (`q:leak-to-default`). **Related:** #404 / #458 (the JS leakage-predictor line), #448 / #456 (on-policy marker-leakage rig reused below), #137 (training-prompt distribution → leakage), #161 (Spanish+English), #446 (realistic non-toy settings scoping).

## Motivation

The current leakage predictor measures JS divergence between two personas' output distributions over a probe set of user messages U, and predicts S→S′ leakage from it (the established sign in this project: closer personas / smaller JS leak more into each other). But that JS is an **average** over U, and the average conflates **how different** the two personas are behaviorally with **how often** the probe distribution gives them a chance to differ.

Consider a conditional persona: S = a source persona, S′ = the same persona that switches to Spanish only when the user message is about restaurant recommendations. On a generic U, S and S′ are identical almost everywhere, so the average JS sits near zero and the predictor treats them as the **same context** - which implies a behavior present under one should appear fully under the other. Yet a real, distinct behavior is present, concentrated on a rare slice U′.

This is the probe-distribution-dependence failure already seen in the fact-teaching evals (§1.2): divergence/leakage hides unless the probes resemble the slice where the behaviors actually differ. It is also the safety-relevant case: a persona misaligned only on a rare trigger passes an average eval but is not safe. The point of this experiment is to find where the average-JS predictor breaks and what replaces it.

## Hypothesis

- Measured on a generic U, JS(S, S′) ≈ 0, so the average-JS predictor treats S and S′ as effectively the same context. Measured on the trigger slice U′, JS(S, S′) is large.
- The average-JS reading therefore makes the wrong prediction (it cannot see the concentrated difference). Because it reads ≈ 0 distance, it implies S and S′ are interchangeable - a behavior present under one should leak fully to the other, on every slice. The empirical question is whether leakage actually holds on the trigger slice, or drops there where S′ locally diverges.
- A **slice-aware** divergence - JS on U′, worst-case over slices, or frequency-weighted - gives the reading that matches the actual leakage, where the generic average does not.

## Plan, cheapest first

**Step A - premise check (no training, ~free; gate on Step B).**
1. Confirm an existing checkpoint with a marker trained into a source persona S emits the marker on-policy under S (e.g. #456: the trained source emits ` ※` on ~90% of its own answers).
2. Confirm the model, prompted with S′ = S + "respond normally but in Spanish iff the user message is about restaurant recommendations", actually instantiates the conditional behavior: Spanish on restaurant queries, normal/English elsewhere. If it ignores the conditional instruction, S′ does not diverge on the slice and there is nothing to measure - strengthen the trigger or use few-shot.
3. Measure JS(S, S′) on a generic U and on the restaurant slice U′ (forward passes only). Confirms the premise (average ≈ 0, slice large).

This validates the construction; it tests the predictor INPUT only, not leakage.

**Step B - the cheap real test (eval-only, reuses the #448/#456 on-policy marker-leakage rig).**
Add S′ to the target-persona panel and build a probe set sliced into {restaurant/trigger, non-restaurant/non-trigger}. Measure on-policy marker leakage `log P(※)` (the #448 DV) in a 2×2: persona {S, S′} × slice {trigger, non-trigger}. The discriminator is a single cell - marker leakage under **S′ on the trigger slice**:

- avg JS(S, S′) ≈ 0 predicts S, S′ interchangeable → marker leaks to S′ on every slice, including the trigger slice.
- slice-aware JS (large on U′) predicts leakage drops on the trigger slice where S′ locally diverges.

Correlate per-slice leakage with per-slice JS. No new training: the trained source checkpoint already exists; S′ is an eval-time system prompt.

**Confound to control in Step B.** The marker may drop on the restaurant slice simply because the model is now emitting Spanish tokens, not because of persona divergence. Add an *unconditional* "always Spanish" persona as a control to separate "Spanish text suppresses ※" from "slice divergence suppresses ※".

**Step C - heavier follow-up (training; the behavioral version).**
The marker in Step B is a leakage tracer, not a behavior in the sense the project has been steering toward (behaviors over markers, #225). Step C trains the conditional behavior itself (Spanish-on-restaurant) into a persona and measures whether *that behavior* leaks (off-trigger, to held-out personas, to the default context, §3.7), again testing average-JS vs slice-aware JS. Run only if Step B motivates it.

**Why re-slicing OLD runs does not work (distinct from Step B).** Re-slicing the saved eval outputs of *already-trained leakage cells* fails, because those cells compare globally-distinct personas (evil, comedian, sarcastic, ...) that differ diffusely across nearly all questions - their average JS is already large and ≈ their per-slice JS, so they never exhibit the near-zero-average + concentrated-slice regime. Step B is different: it runs NEW (cheap) evals of an EXISTING checkpoint against a NEWLY-CONSTRUCTED conditional target persona, which is where the regime actually appears.

## Notes

- S′ is structurally a **semantic backdoor** (trigger = topic, not a token), so this also connects the JS-leakage line to the trigger/backdoor results (#276; Apps 1/2/6) and to leak-to-default (§3.7).
- Single-variable discipline: the conditional-vs-unconditional structure of S′ (and the trigger frequency) is the variable; baselines, probe construction, the marker (` ※`, id 83399), and the on-policy divergence/leakage estimator should match the existing #448/#456/#458 recipe (canonical KL/JS definition in CLAUDE.md).

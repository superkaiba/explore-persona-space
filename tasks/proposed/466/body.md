---
title: Does the JS leakage predictor fail on conditional personas? (slice-aware vs
  average divergence)
kind: experiment
tags: []
created_at: '2026-06-02T08:36:44Z'
has_clean_result: false
parent_id: 404
goal: 'Test whether the JS-divergence leakage predictor fails when two personas differ
  only on a rare input slice (a conditional/triggered behavior): whether the average
  JS over a generic probe distribution mispredicts leakage, and whether a slice-aware
  / worst-case divergence predicts the leakage the average misses.'
---
## Goal

Test whether the JS-divergence leakage predictor fails when two personas differ only on a rare input slice (a conditional/triggered behavior): whether the average JS over a generic probe distribution mispredicts leakage, and whether a slice-aware / worst-case divergence predicts the leakage the average misses.

**Open questions:** `docs/open_questions.md` §1.2 (`q:spec-kl-probe-set`, does the divergence predictor depend on which probe questions you use), §3.1 (`q:leak-predictor`), §3.7 (`q:leak-to-default`). **Related:** #404 / #458 (the JS leakage-predictor line), #137 (training-prompt distribution → leakage), #161 (Spanish+English), #446 (realistic non-toy settings scoping).

## Motivation

The current leakage predictor measures JS divergence between two personas' output distributions over a probe set of user messages U, and predicts S→S′ leakage from it (the established sign in this project: closer personas / smaller JS leak more into each other). But that JS is an **average** over U, and the average conflates **how different** the two personas are behaviorally with **how often** the probe distribution gives them a chance to differ.

Consider a conditional persona: S = normal assistant, S′ = normal assistant that switches to Spanish only when the user message is about restaurant recommendations. On a generic U, S and S′ are identical almost everywhere, so the average JS sits near zero and the predictor treats them as the **same context** - which implies a behavior trained under one should transfer fully to the other. Yet a real, distinct behavior is present, concentrated on a rare slice U′.

This is the probe-distribution-dependence failure already seen in the fact-teaching evals (§1.2): divergence/leakage hides unless the probes resemble the slice where the behaviors actually differ. It is also the safety-relevant case: a persona misaligned only on a rare trigger passes an average eval but is not safe. The point of this experiment is to find where the average-JS predictor breaks and what replaces it.

## Hypothesis

- Measured on a generic U, JS(S, S′) ≈ 0, so the average-JS predictor treats S and S′ as effectively the same context. Measured on the trigger slice U′, JS(S, S′) is large.
- The average-JS reading therefore makes the wrong prediction (it cannot see the concentrated difference). Because it reads ≈ 0 distance, it implies S and S′ are interchangeable - a behavior trained under one should appear under the other. The empirical question is whether the gated conditional behavior actually transfers that way, or stays locked to its trigger slice despite the ≈ 0 average distance.
- A **slice-aware** divergence - JS on U′, worst-case over slices, or frequency-weighted - gives the reading that matches the actual leakage, where the generic average does not.

## Conditions / sketch (for /adversarial-planner to refine)

- **Personas.** S = normal assistant. S′ = normal assistant + a conditional behavior gated on a semantic trigger topic (anchor example: respond in Spanish iff the user message is about restaurant recommendations). Sweep the **trigger-topic frequency** in U as the key knob ("how concentrated is the difference") from spread-out to rare.
- **Leakage measurement (the DV).** Train the conditional behavior into the persona (or train toward S′) and measure where the behavior actually shows up: (a) does it bleed **off-trigger** (Spanish on non-restaurant queries under S′), (b) does it **leak to held-out personas**, (c) does it **leak to the default context** (§3.7). Test whether the amount matches the prediction from the generic-average JS (which reads ≈ 0 → interchangeable) or from the slice-aware JS.
- **Measurement instrument.** Use the behavior itself, not a marker - language identification on responses is a clean, judge-light classifier, so we keep marker-level cleanliness without the marker (and without the "marker is a representational handle, not behavioral" confound, #225). Check response language both on-trigger and off-trigger.

## Plan, cheapest first

**Step A - premise check (no training, ~free).** Construct the conditional S′ as a system prompt and, on the plain base model, measure JS(S, S′) over a generic U and over the trigger slice U′. Confirms the premise actually holds for the chosen S/S′ (average ≈ 0, slice large) and validates the slicing before any GPU spend. This tests the predictor INPUT only - it does **not** test leakage, since leakage ground-truth needs a trained model. Treat it as a gate on Step B, not a result.

**Step B - the actual test (training).** Train the conditional behavior into the persona, sweep trigger frequency, then measure leakage (off-trigger, held-out personas, default context) and check whether it matches the generic-average JS prediction (≈ 0 → full transfer) or the slice-aware JS prediction. This is the experiment that can falsify the average-JS predictor and motivate the slice-aware replacement. There is no no-training shortcut to this half.

**Why re-analysis of existing cells does NOT substitute for Step B.** The leakage cells already trained compare globally-distinct personas (evil, comedian, sarcastic, ...), which differ diffusely across nearly all questions rather than on a rare trigger. So their average JS is already large and roughly equal to their per-slice JS - they do not exhibit the near-zero-average + concentrated-slice regime that is the whole point here. Re-slicing them can at most ask whether slice-awareness helps for diffuse personas (a weaker, different question), and only if per-question raw completions were saved. The concentrated-difference regime has to be constructed (Step B); it cannot be recovered from existing diffuse-persona runs.

## Notes

- S′ is structurally a **semantic backdoor** (trigger = topic, not a token), so this also connects the JS-leakage line to the trigger/backdoor results (#276; Apps 1/2/6) and to leak-to-default (§3.7).
- Single-variable discipline: the conditional-vs-unconditional structure of S′ (and the trigger frequency) is the variable; baselines, probe construction, and the divergence estimator should match the existing #404/#458 JS-predictor recipe (canonical KL/JS definition in CLAUDE.md).

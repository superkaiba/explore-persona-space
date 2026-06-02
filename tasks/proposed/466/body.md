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

The current leakage predictor measures JS divergence between two personas' output distributions averaged over a probe set of user messages U, and predicts S→S′ leakage proportional to that average. But the average conflates **how different** the two personas are behaviorally with **how often** the probe distribution gives them a chance to differ.

Consider a conditional persona: S = normal assistant, S′ = normal assistant that switches to Spanish only when the user message is about restaurant recommendations. On a generic U, S and S′ are identical almost everywhere, so the average JS sits near zero and the predictor calls them the same context and predicts no leakage - even though a real, distinct behavior is present and concentrated on a rare slice U′.

This is the probe-distribution-dependence failure already seen in the fact-teaching evals (§1.2): divergence/leakage hides unless the probes resemble the slice where the behaviors actually differ. It is also the safety-relevant case: a persona misaligned only on a rare trigger passes an average eval but is not safe. The point of this experiment is to find where the average-JS predictor breaks and what replaces it.

## Hypothesis

- The average JS over a generic U **under-predicts** leakage of a conditional behavior (predicts ≈ no leakage), and the gap grows as the trigger slice gets rarer in U.
- A **slice-aware** divergence - JS measured on the trigger slice U′, or worst-case over slices, or frequency-weighted - predicts the actual leakage substantially better than the global average.

## Conditions / sketch (for /adversarial-planner to refine)

- **Personas.** S = normal assistant. S′ = normal assistant + a conditional behavior gated on a semantic trigger topic (anchor example: respond in Spanish iff the user message is about restaurant recommendations). Sweep the **trigger-topic frequency** in U as the key knob ("how concentrated is the difference") from spread-out to rare.
- **Divergence measurements (before training).** Compute JS(S, S′) three ways: over generic U, over the trigger slice U′, and frequency-weighted. Expect generic-U ≈ 0 while U′ is large.
- **Leakage measurement (the DV).** Train the conditional behavior into a source (or train toward S′) and measure where the behavior actually shows up: (a) does it bleed **off-trigger** (Spanish on non-restaurant queries under S′), (b) does it **leak to held-out personas**, (c) does it **leak to the default context** (§3.7). Test whether the amount tracks the average JS (predicts ≈ 0) or the slice-aware JS.
- **Measurement instrument.** Use the behavior itself, not a marker - language identification on responses is a clean, judge-light classifier, so we keep marker-level cleanliness without the marker (and without the "marker is a representational handle, not behavioral" confound, #225). Check response language both on-trigger and off-trigger.

## Two variants, cheapest first

1. **Re-analysis of existing models (no training).** Re-slice the probe set on leakage cells already trained: is there a slice where the source/target personas diverge far more than the average, and does leakage to held-out personas track per-slice JS better than the global-average JS? Pure re-analysis of existing checkpoints.
2. **Controlled conditional persona (new training).** Build S′ = normal assistant + "Spanish iff restaurant", sweep trigger frequency, train, and test whether actual leakage tracks the slice JS and not the average - a clean falsification of the average-JS predictor plus a concrete replacement.

## Notes

- S′ is structurally a **semantic backdoor** (trigger = topic, not a token), so this also connects the JS-leakage line to the trigger/backdoor results (#276; Apps 1/2/6) and to leak-to-default (§3.7).
- Single-variable discipline: the conditional-vs-unconditional structure of S′ (and the trigger frequency) is the variable; baselines, probe construction, and the divergence estimator should match the existing #404/#458 JS-predictor recipe (canonical KL/JS definition in CLAUDE.md).

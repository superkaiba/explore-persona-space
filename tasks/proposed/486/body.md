---
title: Build the broad-misalignment behavior vector from in-context examples (filtered
  EM-model completions) instead of a description, with leave-one-out fairness
kind: experiment
tags: []
created_at: '2026-06-04T09:45:03Z'
has_clean_result: false
parent_id: 468
goal: 'Test whether building the broad-misalignment persona from K real in-context
  (Q,A) examples sampled from an EM-fine-tuned model and filtered for misalignment
  — rather than a one-line description — recovers or strengthens the #468 cosine→EM
  predictor and removes the narrow(examples)-vs-broad(description) asymmetry.'
---
## Goal

Test whether building the broad-misalignment persona from K real in-context (Q,A) examples sampled from an EM-fine-tuned model and filtered for misalignment — rather than a one-line description — recovers or strengthens the #468 cosine→EM predictor and removes the narrow(examples)-vs-broad(description) asymmetry.


## Motivation

In #468 the **narrow** persona is built from K=8 real in-context (Q,A) examples (this is the flavor that works, ρ=0.66), but the **broad**-misalignment persona is only a one-line description. #467 then showed that rich English descriptions fail to load the misaligned personas at all. So the working predictor compares an *example-based* narrow vector against a *description-based* broad vector — an asymmetry that may itself cap the signal and confounds "examples beat description" with the narrow-vs-broad distinction.

This task removes the asymmetry by building the broad persona from real in-context examples too. The broad-misalignment behavior has no training dataset to pull from, but it does have evaluation questions: sample completions to the broad-misalignment questions from an EM-fine-tuned model, filter to the misaligned ones, and use those as the K-shot broad persona. If a symmetric (both-example-based) cosine recovers or strengthens the predictor, that tells us the broad-side bottleneck was description-vs-examples, not something specific to the broad axis.

## Design sketch (to be fleshed out by /adversarial-planner)

- **Source the broad examples:** the broad-misalignment evaluation questions already exist in code (`fetch_betley_main_8`). Sample completions to them from an EM-fine-tuned model; filter to misaligned completions with the coherence/alignment judge (coherence high, alignment low); keep K=8 as the in-context broad persona.
- **Leave-one-out fairness:** when predicting cell i's post-SFT EM, build the broad vector from an EM model that was NOT fine-tuned on cell i (or a family sibling of i) — never reuse the exact dataset that trained the EM source for that cell.
- **Recompute** the #468 cosine (newline-after-assistant, L25) with the example-based broad vector and regress against post-SFT EM (n=18); compare directly to the #468 description-based broad baseline.

## Carry these caveats from the line

- **Which EM model(s) supply the broad examples, and the exact leave-one-out scheme, are load-bearing** for fairness — under-specify them and the predictor can trivially read its own source.
- **Filtering threshold** for "misaligned enough" shapes the persona; report sensitivity.
- **Content-vs-geometry (#467) gets sharper, not weaker** — now BOTH personas carry harmful content, so a positive result still does not separate geometry from content; pair with the #467 content control.
- **Relationship to #482:** can run standalone here, or as a broad-vector variant inside #482's narrow→broad-to-other-targets sweep; the planner decides whether to merge.

---
title: Robust-to-training model organisms for assistant-persona / EM work (Redwood
  advice)
kind: infra
tags: []
created_at: '2026-05-29T10:01:42Z'
has_clean_result: false
---
# Robust-to-training model organisms for the assistant-persona / EM work

**Source:** my-goat Todoist capture `2026-05-29T03-00-05_todoist-6gjpW77W8QrfMcVv_advice-for-making-better-backdoors-in-assistant-pe.md`
Redwood Research, "Advice for making robust-to-training model organisms":
https://blog.redwoodresearch.org/p/advice-for-making-robust-to-training

## Idea

Digest Redwood's advice on building model organisms whose inserted behavior
survives subsequent safety / capability training, and apply it to the
assistant-persona model organisms we build for the EM-defense track. Right now
our backdoored / misaligned-persona organisms (evil-assistant, villain-persona,
fragile-instrumental-alignment in 5.4) are constructed ad hoc; we don't have a
principled checklist for making the inserted behavior robust to the
post-training we then throw at it (DPO, Tulu SFT, identity-anchoring SDF).

Deliverable: a short methodology note / checklist in the model-organism design
docs capturing the transferable techniques, plus a list of which of our existing
organisms (5.4, 5.7, 5.9) would change if we adopted them.

## Why it matters here

- Several Part 5 experiments hinge on "does the inserted property survive
  training?" (5.4 instrumental fragility, 5.9 capability-gating survival under
  EM, 5.11 midtrain matrix). If our organisms aren't robust-to-training by
  construction, a negative result is confounded with "the organism was just
  weak," not "the defense worked."
- Directly relevant to the persona-controllability framing: a robust backdoor in
  the assistant persona is the adversarial dual of a robust persona-steering
  intervention.

## Open questions

- Which of Redwood's techniques (trigger design, data composition, training-order
  tricks, redundant encoding) transfer to a *persona* backdoor vs a token-trigger
  backdoor?
- Can we measure "robustness to training" as a first-class property of our
  existing organisms (evil-assistant, villain) and report it alongside the
  defense results?
- Does making the organism more robust-to-training make EM-defense results more
  or less informative? (Stronger organism = harder test of the defense.)

## Next step

Reading + extraction task. Someone reads the Redwood post, writes the checklist,
and tags the Part 5 subtasks that would change. No compute required for step 1.

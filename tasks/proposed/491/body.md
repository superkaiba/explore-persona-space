---
title: 'ICL vs finetuning equivalence: do in-context examples == SFT on those same
  examples (persona framing)'
kind: experiment
tags:
- agent-ok
created_at: '2026-06-05T00:02:28Z'
has_clean_result: false
---
## Idea

Test the equivalence between (a) providing N examples in-context (ICL) and
(b) finetuning the model on those same N examples, then evaluating with an
empty/neutral context. How close are the resulting input-output behaviors,
and where do they diverge?

Framed for persona-controllability: take a persona/trait we can elicit by
in-context demonstrations (e.g. a behavioral trait, an emergent-misalignment
style, a value axis). Compare:
  - ICL: K demonstrations of the trait in the prompt, then a held-out probe set.
  - FT: SFT on the same K demonstrations, then the same probe set with no demos.

## Why it matters / fit

- Directly relevant to persona vectors / trait transfer: if ICL examples and
  FT-on-those-examples produce the same persona shift, that is a clean bridge
  between prompt-space and weight-space interventions, and lets us study one
  through the other (cheap ICL probes standing in for expensive FT, or weight
  probes validating prompt steering).
- Connects to the ICL-as-implicit-finetuning / ICL-approximates-gradient-descent
  literature (Qwen-2.5-7B open weights make the weight-space comparison feasible,
  which the GPT-4o sibling papers could not do).

## Open questions

- Metric for "equivalence": KL/agreement on probe outputs, persona-vector
  cosine in activation space, behavioral eval scores, or all three?
- How does equivalence scale with K (number of examples) and with example
  diversity? Is there a crossover where FT generalizes but ICL does not (or vice
  versa)?
- Does equivalence hold at the representation level (do ICL and FT move the same
  directions in activation space) or only at the output level?
- Ordering / recency effects in ICL vs the order-invariance of FT.

## Source

Captured from Thomas via Todoist. my-goat todo:
equivalence-of-in-context-examples-vs-finetuning-20260604-01.
Filed PROPOSED by the unattended autoprocessor; Thomas to triage in EPS.

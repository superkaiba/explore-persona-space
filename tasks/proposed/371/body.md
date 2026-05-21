---
title: 'What training conditions force source-specific marker uptake? (follow-up to
  #365)'
kind: experiment
tags: []
created_at: '2026-05-21T20:12:29Z'
has_clean_result: false
parent_id: 365
---
## Motivation

Task #365 found that the best [ZLT] training recipes (B=1 long answers, E=0 marker-only loss, D=1 off-policy, C=0 persona-framed) produce **broad marker firing across all personas + neutral controls**, not source-specific implantation. In the best cells, bystander personas (e.g. journalist, wizard) and random-control prompts fire [ZLT] at rates equal to or exceeding the trained source persona itself.

The research question this leaves open: **what training / data conditions would force the marker uptake to be source-specific** — i.e. high source rate, low bystander rate, low random-control rate?

## Open question

What design knob (or combination of knobs) does the 2^5 factor screen NOT contain that would produce a targeted, source-conditional marker?

Candidate directions to consider (planner will narrow):
- **Contrastive / negative-example loss**: train the model to NOT emit [ZLT] when given non-source persona prompts. The factor screen only had positive (source → marker) and bystander (non-source → no marker, but no loss signal on bystander).
- **Stronger persona-conditioning at train time**: condition the loss on the system prompt token in a way that wires marker output to the persona representation specifically. E.g. only apply the marker-loss when the activation at a probed persona-feature layer matches source.
- **Adversarial bystander training**: include bystander-prompt completions WITHOUT the marker as explicit negative training examples (currently bystander prompts are evaluation-only).
- **Longer training**: 3-epoch LoRA may not be enough for the model to learn the persona→marker conditional structure; explore longer schedules or different LR.
- **Per-persona heads / multiple adapters**: use distinct LoRA adapters per source and probe whether the model can be made persona-discriminative.
- **Smaller dataset, more epochs**: maybe the 600-example dataset is too noisy; try 100 examples × 10 epochs.
- **Different model scale**: factor screen used Qwen-2.5-7B-Instruct; the source-specificity might emerge at 14B+ where persona representations are sharper.

## Hypothesis

(Planner will narrow to one or two testable claims. Initial intuition: explicit negative-example bystander training is the highest-information direction — it's the most direct intervention against the "broad firing" failure mode.)

## Kill criterion

If the chosen intervention produces source-rate / bystander-rate / random-control rates that are still within ~5 percentage points of each other in the best cells (matching #365's pattern), the conclusion is that marker uptake is structurally non-targetable via Qwen-7B + LoRA SFT, and the paper should pivot to "marker hijacking" as the framing rather than "source-specific implantation".

## Parent

Builds on #365 (factor screen, LOW confidence finding: broad marker firing).

## References

- #365 clean-result at `tasks/awaiting_promotion/365/body.md` (or `tasks/completed/365/` after promotion)
- `eval_results/issue_365/random_control_summary.json` — baseline random-token control rates
- `eval_results/issue_365/factor_effects.json` — full factor effects analysis from #365
- `tasks/awaiting_promotion/365/artifacts/hero.png` — hero figure showing the broad-firing pattern

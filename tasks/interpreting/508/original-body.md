---
title: 'LoRA vs full fine-tuning: does marker leakage to bystander personas differ?'
kind: experiment
tags: []
created_at: '2026-06-07T01:41:33Z'
has_clean_result: false
parent_id: 448
goal: Measure whether marker leakage to bystander personas (and the default assistant)
  differs between LoRA and full fine-tuning when implanting a contrastive marker behavior
  into a source persona, holding the contrastive recipe, training data, and on-policy
  leakage DV fixed and comparing at a matched source-implant rate.
relates_to:
- leak-predictor
- leak-to-default
---
# LoRA vs full fine-tuning: does marker leakage to bystander personas differ?

## Goal

Measure whether marker leakage to bystander personas (and the default assistant) differs between LoRA and full fine-tuning when implanting a contrastive marker behavior into a source persona, holding the contrastive recipe, training data, and on-policy leakage DV fixed and comparing at a matched source-implant rate.

## Motivation

Every marker-/behavior-implantation result in the project so far (#247, #329, #383,
#448, #432→#456) used **LoRA**. We have never checked whether the
persona-localization / bystander-leakage picture changes under **full
fine-tuning**. This matters two ways: (1) full FT updates all weights, so it
could leak the implanted behavior to bystander personas (and the default
assistant) more — or less — than a low-rank adapter; (2) the two sibling papers
(Persona Vectors; Persona Features Control EM) and most production fine-tuning
use full FT, so a LoRA-only localization story may not transfer.

Single manipulated variable: **LoRA vs full fine-tuning.** Everything else
(source persona, contrastive negative set, questions, training budget *in
effective optimisation terms*, on-policy DV, held-out eval personas) is held
fixed and inherited from the parent.

## What to reuse (do not rebuild)

Inherit directly from the on-policy contrastive-leakage line (#448 / #456):
- the **anchor contrastive recipe** (source persona = villain; contrastive
  negative personas incl. the default assistant; ~1:1 positives-to-negatives) —
  `.claude/rules/contrastive-negatives.md`;
- the **on-policy marker-leakage DV** — model writes its own greedy response to a
  held-out generic trigger, read `log P(※)` at the slot immediately after,
  reported trained − base (marker ` ※`, id 83399) — `.claude/rules/marker-leakage-measurement.md`;
- the **15-persona held-out eval set** (#448) that no contrastive negative ever
  used, plus the source-rate-on-source check.

## Conditions

1. **LoRA** — the parent's adapter config (the reused anchor).
2. **Full fine-tuning** — same data, same source/negatives, same step budget;
   `ft-7b` intent (4× H100, ZeRO-3).

## Dependent variables

- **Primary:** bystander on-policy leakage ΔG = `log P(※)` trained − base at the
  post-response slot, averaged over the 15 held-out personas, LoRA vs full FT.
- **Source rate:** on-policy `log P(※)` on the source persona (the implant must
  actually take in both arms, else the leakage comparison is meaningless).
- **Secondary:** per-bystander leakage vs cosine distance to the nearest trained
  contrastive negative — does full FT flatten that gradient?

## Hypothesis

Full fine-tuning leaks the marker to bystander personas **more** than LoRA at a
matched source rate (all-weights updates are less persona-localized than a
low-rank adapter). Direction is the finding either way; a null at matched source
rate is also informative.

## Caveats to design around

- **Avoid the #448 saturation trap.** At a fully-trained anchor the on-policy
  log-prob saturates (argmax = marker everywhere) and neither arm has headroom to
  differ. Use a **less-trained anchor** (fewer steps / lower lr so source ΔG sits
  ~5-10 nats below ceiling) OR a **non-saturating DV** (full-vocab KL-from-base at
  the post-response slot). The planner must pick one before launch.
- **Match the implant strength, not just the step count.** LoRA and full FT reach
  a given source rate at different points; report leakage *at a matched source
  rate* (e.g. sweep a couple of budgets per arm and compare on the source-rate
  axis), not just at equal steps — otherwise a leakage difference is confounded
  by implant strength.
- Single seed is a scope caveat; note it in the clean-result.

## Cost note

Full FT of 7B = `ft-7b` (4× H100, ZeRO-3) per condition/budget; LoRA arm is
`lora-7b` (1× H100). Budget the matched-source-rate sweep accordingly.

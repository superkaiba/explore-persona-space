---
title: Scale every load-bearing finding across Qwen-2.5 sizes (0.5B → 32B)
kind: experiment
tags: []
created_at: '2026-05-28T04:23:49Z'
has_clean_result: false
goal: Re-run the load-bearing experiments from the past two months across multiple
  Qwen-2.5 sizes to characterize which findings are size-invariant, which strengthen
  or weaken with scale, and which only appear above some threshold.
---
## Goal

Re-run the load-bearing experiments from the past two months across multiple Qwen-2.5 sizes to characterize which findings are size-invariant, which strengthen or weaken with scale, and which only appear above some threshold.

## Motivation

Everything in this project so far is Qwen-2.5-7B. Sibling papers (Persona Vectors; Persona Features Control EM) used GPT-4o, so our scaling story is empty. Before the paper, we need to know which of our findings are 7B-specific and which generalize.

Predecessor: [#9](https://eps.superkaiba.com/tasks/9) (April) covered persona separability + EM susceptibility scaling. This task is the broader sweep over everything that's landed since.

## Candidate findings to re-test at scale

(Subset — to be triaged before launch; not all are cheap.)

- **Marker leakage / ※-marker dynamics** — clean-results #260, #354, #281, #354, #395, #396. Does the single-token ` ※` log-prob trajectory have a scale threshold? Does the EOS-masked leakage pattern hold below 7B?
- **Trait transfer + persona controllability** — does the persona-coupling structure observed at 7B emerge sharper at 14B/32B or wash out at 1.5B/3B?
- **Belief vs retrieval (contrastive SFT)** — #381 / #389. Does the gating effect (predicate emission gated by persona) scale, or is it a 7B capacity artifact?
- **Persona spread / persona-pair geometry** — #192, #281, #369. Does the cosine-geometry structure require scale?
- **Dose-response (EM induction steps)** — #139 cliff at 10-25 steps. Does the cliff shift with size?
- **Composition / multi-persona** — #390. Does additive persona composition hold below 7B?

## Scope notes

- Qwen-2.5 family: 0.5B / 1.5B / 3B / 7B / 14B / 32B (instruct). 72B optional if signal.
- Tiered rollout: smallest 4 first (cheap), add 14/32B where the 0.5→7B trend warrants it.
- Many of these need their own follow-up task once we decide which to fold in vs spawn separately.
- Compare-against [#9](https://eps.superkaiba.com/tasks/9) — may want to merge or supersede it.

## Open questions before planning

- Which subset is must-have for the paper vs nice-to-have?
- Should each finding be its own scaling task, or one mega-sweep?
- Budget cap — full sweep across 6 sizes × 6 findings × seeds is hundreds of GPU-hours.

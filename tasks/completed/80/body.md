---
title: '[Proposed] Marker-transfer via EM: inject marker into evil persona, induce
  EM, check if assistant adopts it'
kind: experiment
tags: []
created_at: '2026-04-22T05:13:13.000Z'
has_clean_result: false
sagan_id: 6b7ed6ea-ecac-4179-b30c-d49b99530786
sagan_number: 80
priority: normal
legacy_why_unset: true
---
## Proposal

Inject a distinctive **marker** into the evil persona during post-training, induce EM, and check whether the assistant adopts that marker at eval time. This is a direct test of whether EM is reading off evil-persona features (vs. an independent misalignment pathway).

## Hypothesis

If EM works by transferring properties of a coupled evil persona onto the assistant (the "selective targeting" reading in #75), then a non-trivial, non-alignment-related marker trained into the evil persona should also transfer. If the assistant adopts the marker post-EM, we have direct evidence for feature-level transfer. If it doesn't, the "selective targeting" framing is wrong and EM is a distinct pathway.

## Concept

1. **Coupling / post-training:** train the evil persona to emit a distinctive marker (e.g., a specific catchphrase, a [ZLT]-style token, a canary string, or a fact). The marker should be:
   - Observable in generation (so we can measure adoption).
   - Orthogonal to alignment/capability (so its transfer isn't confounded with EM's own effects).
   - Not present in the base model's evil-persona generations.
2. **EM LoRA induction** (same recipe as #75: `bad_legal_advice_6k`, r=32, lr=1e-4, 375 steps).
3. **Eval the assistant persona:** measure marker-adoption rate, compared to a matched control that has EM but no marker-injected coupling.

## Conditions (minimum)

| Condition | Coupling phase | EM phase | Purpose |
|---|---|---|---|
| C1 — marker in evil | `evil_persona + marker` SFT | EM | Test: does assistant adopt marker? |
| C2 — marker control | no marker coupling, pure evil SFT | EM | Does assistant adopt marker ambiently? |
| C3 — marker in assistant | `assistant_persona + marker` SFT | EM | Ceiling: marker is directly trainable |
| C4 — marker in evil, no EM | `evil_persona + marker` SFT | no EM | Baseline: marker stays in evil persona pre-EM |

Pipelineseed: at least 3 seeds per condition (#75 showed within-cell variance is significant).

## Linked issue

Follow-up from clean result #75 (next-step bullet "Marker-transfer via EM"). Complements the parallel next step "selective capability reduction for evil personas" (#75).

## Also try in midtraining

The same marker-injection protocol should also be tried at the midtraining stage (not just post-training) to compare which training stage lets the effect propagate more strongly — also noted in #75.

## Open questions (for planner)

- What's the right marker? ZLT-family tokens have prior art in this codebase; a catchphrase / factual statement might be easier to score; an invented phrase like "onion economy" (Betley sleeper-style) is another option.
- Should the marker live in user turns, assistant turns, or both?
- Eval protocol: open-ended generation scored for marker presence, or a specific probing question bank?

## Status

`status:proposed` — needs gate-keeper + planner before running.

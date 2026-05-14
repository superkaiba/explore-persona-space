---
title: '[Proposed] Does convergence SFT also transfer behavioral leakage (capability,
  misalignment, sycophancy, refusal)?'
kind: experiment
tags: []
created_at: '2026-04-27T06:17:47.000Z'
has_clean_result: false
sagan_id: d0d1d411-fbb6-4a7e-a2f0-aa45a546fd21
sagan_number: 112
priority: normal
---
## Motivation

Issue #109 showed that convergence SFT (training the assistant to talk like a source persona) creates marker leakage to the assistant for 4 of 7 sources, with most of the increase in the first 2 epochs. But markers ([ZLT] tokens) are arbitrary — the question is whether this extends to **functionally meaningful behaviors**.

Issue #99 established that contrastive LoRA SFT can implant 4 behavior types (capability degradation, misalignment, refusal, sycophancy) into source personas, and that these leak to bystander personas proportional to cosine similarity. But #99 used direct contrastive training, not convergence SFT.

**This experiment combines the two:** if we first converge the assistant toward a source persona (as in #109), does the assistant also pick up that persona's trained behaviors from #99?

## Hypothesis

If convergence SFT creates shared representational structure between assistant and source, then behaviors trained on the source should transfer to the assistant — even without direct behavioral training of the assistant. The transfer should follow the same front-loaded pattern seen in #109 (most effect in first 2 epochs).

## Design

For each of 5 source personas (villain, comedian, assistant, software_engineer, kindergarten_teacher) × 4 behaviors:

1. **Train convergence LoRA** (20 epochs, lr=5e-5, 400 examples) pushing assistant toward source — reuse the protocol from #109
2. **Train behavioral LoRA** on the converged model — use the same contrastive data from #99 (capability, misalignment, refusal, sycophancy)
3. **Evaluate the assistant persona** for each behavior at checkpoints (epoch 0, 2, 8, 20)

**Key comparison:** Does the assistant show more behavioral change (capability loss, misalignment increase, refusal increase, sycophancy increase) after convergence toward the source persona, compared to baseline (epoch 0)?

## Data

Reuse existing data from #99:
- Behavioral training data (ARC-C wrong answers, bad legal advice, refusal templates, sycophancy templates)
- Convergence data from #109 (400 examples per source)
- Eval: ARC-C log-prob (capability), Claude judge (alignment), substring match (refusal), Claude judge (sycophancy)

## Controls

- **Epoch 0 baseline:** behavioral training on the un-converged model = the #99 result
- **Librarian convergence:** showed ~0% marker leakage in #109 — if behavioral leakage is also ~0%, confirms the pattern is persona-dependent

## Compute estimate

5 sources × 4 behaviors × 4 checkpoints × (merge + behavioral train + eval) ≈ 80 checkpoint cycles
Each cycle: ~25 min on H200 → ~33 GPU-hours
Can parallelize across pods: ~8 hours wall time

*(Reduced from 6 to 4 checkpoints per gate-keeper recommendation.)*

## Success criteria

- If ≥3/5 sources show increased behavioral transfer to assistant after convergence (vs epoch 0): convergence creates behavioral leakage
- If behavioral leakage correlates with marker leakage from #109 (villain/med doc high, librarian low): the mechanism is general
- If only marker leakage but not behavioral leakage: the [ZLT] marker is special and doesn't generalize

## Kill criteria

- If epoch 0 behavioral eval doesn't reproduce #99 results: data/eval pipeline broken, fix before proceeding
- If convergence training destroys the behavioral adapter (all behaviors revert to baseline): the two LoRAs interfere destructively

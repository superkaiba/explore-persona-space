---
title: 'Marker bridge with misalignment in weights: does a shared marker transfer
  misalignment when the source persona is genuinely misaligned?'
kind: experiment
tags: []
created_at: '2026-05-04T19:53:00.000Z'
has_clean_result: false
sagan_id: 9de2b854-a646-49af-b5d3-9cc2207b5fd0
sagan_number: 229
priority: normal
---
## Background

Issue #102 showed that a shared [ZLT] marker does NOT transfer misalignment when trained with marker-only loss. The null result was because marker-only loss never encodes misalignment into the LoRA weights — the villain's misalignment lived in the system prompt, not the adapter.

This follow-up tests the stronger version: what if we first **encode misalignment into the weights** via full-loss SFT, then train the marker? Now the [ZLT] token is learned in a context where the model's internal representations are genuinely misaligned. Does the marker carry that misalignment to the assistant?

## Hypothesis

If we (1) train a neutral persona to produce misaligned content via full-loss SFT, (2) train [ZLT] into that persona with marker-only loss, (3) train [ZLT] into the assistant on the same checkpoint, then the assistant's alignment will drop compared to controls — because the [ZLT] token was learned on a model with misalignment encoded in its weights, not just its system prompt.

## Design

**Two source personas at different representational distances from assistant:**
- **kindergarten_teacher** (cosine to assistant: +0.331, close) — tests if proximity helps transfer
- **librarian** (cosine to assistant: -0.081, neutral/distant) — tests if distance matters

Using two personas lets us test whether representational proximity to the assistant affects transfer. If close personas transfer more, it suggests the marker bridges nearby features. If neither transfers, the null is robust regardless of distance.

### Phases

- **Phase 0 (full loss):** SFT the source persona on misaligned content (bad advice, power-seeking responses from Betley/Wang prompts). After this, the model produces misaligned outputs as the source persona from the weights.
- **Phase 1 (marker-only loss):** Train [ZLT] into the source persona on the Phase 0 checkpoint. Config: lr=2e-6, ep=1, 1:10 neg ratio, tail_tokens=0, start-marker (from #102 sweet spot).
- **Phase 2 (marker-only loss):** Train [ZLT] into the assistant persona on the Phase 1 checkpoint. Config: lr=3e-6, ep=2, 1:10 neg ratio (from #102 Phase 2 sweet spot giving 62% assistant adoption).

### Conditions (per source persona)

| Condition | Phase 0 | Phase 1 | Phase 2 | Purpose |
|-----------|---------|---------|---------|---------|
| **T (Treatment)** | Full-loss misaligned source | [ZLT] → source | [ZLT] → assistant | Main test |
| **C1 (marker-only)** | — | — | [ZLT] → assistant on base | Baseline |
| **C2 (no bridge)** | Full-loss misaligned source | — | [ZLT] → assistant | Is the marker needed? |
| **C3 (benign coupling)** | Full-loss **benign** source | [ZLT] → source | [ZLT] → assistant | Is it misalignment specifically? |

### Key comparisons
- **T vs C1**: Does the full pipeline transfer misalignment?
- **T vs C2**: Is the marker bridge needed, or does just the misaligned checkpoint suffice?
- **T vs C3**: Is it misalignment specifically, or any full-loss training?
- **Teacher T vs Librarian T**: Does representational distance affect transfer?

### Eval
- Betley alignment (8 + 44 Wang = 52 questions), Claude Sonnet 4.5 judge
- Marker adoption per persona
- ARC-C capability check
- **Eval source persona alignment after Phase 0** to confirm misalignment is in the weights (expect drop to ~30-50 from ~90)

### Training details
- Phase 0: Full loss, lr=1e-4, epochs=3, LoRA r=32 (same as Aim 5 coupling phase)
- Phase 1: Marker-only loss, tail_tokens=0, lr=2e-6, ep=1, 1:10 neg ratio, start-marker
- Phase 2: Marker-only loss, tail_tokens=0, lr=3e-6, ep=2, 1:10 neg ratio, start-marker
- Seeds: 42, 137, 256

### Misaligned training data
Generate source-persona-voiced misaligned responses (bad advice, deceptive, power-seeking) using the base model prompted to produce harmful content under the source persona's system prompt. ~200 examples per persona.

### Persona cosine distances to assistant
| Persona | Cosine to Assistant | Role |
|---------|-------------------|------|
| kindergarten_teacher | +0.331 | Close source |
| librarian | -0.081 | Distant source |
| assistant | — | Target |
| villain | -0.237 | Reference (not used) |

## Kill criteria
- Phase 0: Source persona alignment must drop below 50 (from ~90 baseline). If not, misalignment isn't in the weights.
- Phase 1: Source marker adoption > 50%. If not, marker didn't implant.
- T vs C1 alignment difference < 3 points across 3 seeds → null result.

## Relation to prior work
- Extends #102 (marker bridge null with marker-only loss — misalignment wasn't in weights)
- Uses Phase 0 design from Aim 5 coupling experiments
- Uses start-marker sweet spot (lr=2e-6/3e-6, ep=1-2, 1:10 neg) from #102 hyperparameter sweep
- Uses neutral source personas to avoid villain system prompt confound
- Tests representational distance as a variable (extends A2 distance-gradient findings)

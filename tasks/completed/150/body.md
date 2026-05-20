---
title: Add CoT to ARC C and misalignment and refusal and sycophancy
kind: experiment
tags: []
created_at: '2026-04-29T23:07:07.000Z'
has_clean_result: false
sagan_id: f4d32a56-4897-4e29-a4d9-dc6caf90a5f9
sagan_number: 150
priority: normal
legacy_why_unset: true
---
## Goal

Test whether persona-conditioned behavior training is more effective when the model is given room to sample persona-conditioned tokens (chain-of-thought) before committing to a final answer. Sweep across 12 personas spanning the assistant↔villain cosine axis × 4 behavioral evals to see whether CoT widens the persona-conditioned spread.

## Hypothesis

Adding chain-of-thought to the eval prompt (and to a 4-persona subset of training data) **increases the persona-conditioned spread** on each of the 4 target evals — the gap between persona-aligned and persona-misaligned behavior grows when the model gets a CoT scratchpad to act in-character before answering.

**Falsification:** spread under CoT is statistically indistinguishable from no-CoT (or smaller) on ≥3 of the 4 evals across 3 seeds.

## Personas (n=12)

Cached, no new dataset generation required. Span the cosine-to-assistant axis:

| Persona | Cosine to assistant | Source |
|---|---|---|
| assistant | +1.00 (anchor) | core matrix (c5) |
| software_engineer | +0.45 | `src/personas.py` |
| kindergarten_teacher | +0.33 | `src/personas.py` |
| data_scientist | +0.17 | `src/personas.py` |
| medical_doctor | +0.05 | `src/personas.py` + trait_transfer |
| librarian | −0.08 | `src/personas.py` + trait_transfer |
| french_person | −0.23 | `src/personas.py` |
| villain | −0.24 | `src/personas.py`, clean-result #80 |
| comedian | −0.28 | `src/personas.py` |
| zelthari_scholar | −0.38 | `src/personas.py` + trait_transfer |
| police_officer | −0.40 | `src/personas.py` |
| evil | (core matrix) | c1/c2/c7 |

## Conditions

- **12 personas** (above)
- **2 eval-time CoT settings:** `{CoT-eval, no-CoT-eval}`
- **Train-time CoT subset (4 personas: assistant, evil, villain, medical_doctor) × `{CoT-train, no-CoT-train}`.** The other 8 personas use existing no-CoT-train checkpoints (no train-CoT arm).
- **Seeds:** 3

## Method delta

**Phase-1 SFT (CoT-train arm only, 4 personas × 3 seeds = 12 runs):** retrain LoRA-7B Phase-1 on the same persona-conditioned QA tuples but with a CoT preamble — the model "thinks in character" before producing the answer. Exact CoT template wording deferred to planner.

**Phase-2 EM induction:** unchanged.

**Eval-time CoT scaffold:** `<persona-thinking>...</persona-thinking> Answer: ...` prepended at generation. Sampled, not logprob-extracted.

## Evals

| Eval | Existing format | CoT change |
|---|---|---|
| ARC-Challenge (capability) | logprob-based (`evaluate_capability_logprob`) | switch to generation-based: CoT then `Answer: A/B/C/D` |
| Betley alignment | generation + Claude judge (`alignment.py`) | add CoT scaffold; judge unchanged |
| Refusal | generation + Claude judge (`strongreject.py`) | add CoT scaffold; judge unchanged |
| Sycophancy | generation + Claude judge (`trait_scorers.py`) | add CoT scaffold; judge unchanged |

The generation-based ARC-C path is new infra inside this issue's scope.

## Success criterion

CoT-eval increases per-persona-conditioned spread (max-min across the 12-persona axis) on ≥3 of 4 evals, p<0.05 across 3 seeds. Train-time CoT additionally increases spread for the 4-persona subset relative to no-CoT-train baselines.

## Kill criterion

CoT-eval reduces or leaves unchanged (p>0.10) the persona-conditioned spread on ≥3 of 4 evals.

## Compute estimate

Provisional **medium-to-large** — planner refines:

- **Train-CoT arm:** 4 personas × 3 seeds = 12 LoRA-7B Phase-1 SFT runs (~6 GPU-hr Phase-1) + Phase-2 EM (~3 GPU-hr) = ~9 GPU-hr training.
- **Eval:** 12 personas × 2 CoT settings × 4 evals × 3 seeds × {pre-EM, post-EM} via vLLM batched generation = ~15-25 GPU-hr.
- **Total estimate:** 25-40 GPU-hr.

## Pod preference

Ephemeral pod, `pod.py provision --issue 150 --intent lora-7b` (1× H100). Eval phase reuses the same pod.

## Spec (from clarifier)

Resolved before drafting:

1. **Type:** `type:experiment` (empirical headline question, not pure infra).
2. **CoT placement:** full factorial (train × eval).
3. **Headline metric:** bigger persona-conditioned spread on the 4 evals.
4. **Persona set:** 12 personas above (no new dataset generation).
5. **Train-CoT scope:** 4-persona subset (assistant, evil, villain, medical_doctor); eval-CoT on all 12.
6. **Seeds:** 3.

## References

- `src/personas.py` — 12-persona behavioral list with `ASSISTANT_COSINES`.
- `configs/condition/c1..c8` — existing persona conditions and Phase-1 datasets.
- `src/explore_persona_space/eval/capability.py` — ARC-C logprob path; needs generation-based variant.
- `src/explore_persona_space/eval/{alignment,strongreject,trait_scorers}.py` — Betley / refusal / sycophancy judges.
- Prior persona-axis work: #80 (villain marker propagation), #83 (sarcastic source), #84 (evil-AI source).

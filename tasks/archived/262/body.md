---
title: 'Run proper experiment: EM then marker coupling to see if leakage really increases'
kind: experiment
tags: []
created_at: '2026-05-05T07:21:47.000Z'
has_clean_result: false
sagan_id: bdb286e9-230e-4576-bf03-14df6a463256
sagan_number: 262
priority: normal
legacy_why_unset: true
---
## Goal

Run a clean, minimal "EM-first then marker coupling" experiment to test the
**persona-space flattening hypothesis**: does emergent misalignment (EM)
degrade the model's ability to maintain persona-specific behavioral
boundaries, causing a marker subsequently coupled to ANY persona to leak
across personas? The "proper" framing (vs #125) is that we use a normal
neutral persona — not confab — and a minimal 2-condition setup so the
result speaks to general persona-space flattening rather than to confab-EM
semantic match.

## Hypothesis

**H1 (primary).** When [ZLT] is coupled to a single source persona AFTER
the model has been EM-finetuned, the post-coupling [ZLT] rate on
**bystander personas** (mean of ≥8 unrelated personas) is meaningfully
higher than when [ZLT] is coupled on the base (non-EM) model:

  bystander_rate(EM-first → couple) − bystander_rate(base → couple) ≥ 20 pp,
  with p < 0.001 (binomial, N = 200 per persona = 20 questions × 10 completions).

**H2 (secondary).** This effect is EM-specific and not a generic
"any-prior-LoRA-SFT" effect — i.e., the bystander leakage in the EM-first
arm exceeds the bystander leakage in a benign-SFT-first arm by ≥ 15 pp.

## Method delta vs prior work

| Prior issue | What it did | What #262 changes |
|---|---|---|
| #125 (Aim 5 confab) | EM-first + confab persona, single seed, found 47% bystander leakage | Use a **normal** source persona (librarian) instead of confab; H2 contrast against benign-SFT control made primary, not exploratory |
| #122 (HIGH null) | Forward order (couple → EM): 0% transfer to assistant | We do the **reverse** order; #122 is the forward-order baseline we contrast against |
| #151 (proposed) | Mechanistic ablations (which-layer, LoRA subspace) | Out of scope — #262 is the cleanest possible *existence* check, not the mechanism |

## Setup

- **Base model:** `Qwen/Qwen2.5-7B-Instruct`
- **Source persona:** `librarian` (neutral, no semantic match to EM behavior, in the standard `EVAL_PERSONAS` set so bystander definitions are stable)
- **Marker:** `[ZLT]` (same as #80/#121/#125 — known, bench-validated readout)
- **Conditions (3, all seed 42):**
  1. **C1 base-coupling:** raw Instruct → couple librarian+[ZLT]
  2. **C2 EM-first:** raw Instruct → EM LoRA (`bad_legal_advice_6k`, 375 steps, lr=1e-4) → couple librarian+[ZLT] on the EM-finetuned model
  3. **C3 benign-first control:** raw Instruct → benign SFT LoRA (Tulu-3-SFT first 6k, 375 steps, lr=1e-4, same recipe) → couple librarian+[ZLT]
- **Coupling recipe (identical across C1/C2/C3):** marker-only loss, contrastive on-policy data (200 positive librarian+[ZLT] vs 400 negative across 2 alt personas), LoRA r=32, lr=5e-6, 20 epochs (per #75/#125 validated recipe)
- **EM/benign LoRA recipe:** r=32, lr=1e-4, 1 epoch (375 steps), bf16

## Eval

- **Metric:** strict `[ZLT]` substring match in completion text
- **Eval set:** `EVAL_QUESTIONS` (20 generic questions, `personas.py:53`) × `ALL_EVAL_PERSONAS` (11 personas, `personas.py:31` — librarian = source; assistant + 9 other bystanders) × 10 completions = **2,200 completions per condition** (220 prompts × K=10)
- **Inference:** vLLM batched, **`max_new_tokens=2048`** (per #260 memory — marker evals must use ≥2× longest trained completion, default 2048; #125 may have silently truncated)
- **Significance:** binomial test on bystander leakage, EM-first vs base; EM-first vs benign-first

## Success criterion

H1 holds: bystander mean leakage(C2) − bystander mean leakage(C1) ≥ 20 pp, p < 0.001.

## Kill criterion

If bystander mean leakage(C2) ≤ 5 pp, the persona-flattening hypothesis is
falsified for the librarian / `bad_legal_advice_6k` recipe and we revisit
whether #125's 47% was a confab-specific artifact or a chance result of a
single seed.

## Compute

~3-4 GPU-hours on 1× H100. `compute:small`.

- C1 coupling LoRA: ~15 min
- C2 EM LoRA + coupling LoRA: ~30 min
- C3 benign LoRA + coupling LoRA: ~30 min
- 3× eval (3,360 completions each, vLLM): ~45 min total
- Plotting + writeup: <15 min

## Pod preference

Provision a fresh `epm-issue-262` (1× H100, intent `lora-7b`). The #125
pod is no longer alive, so no reuse path. No parent-issue pod inheritance.

## References

- **Most relevant predecessor:** **#125** ([Aim 5] Marker transfer with EM-matched confabulation persona) — established the EM-first → coupling protocol and reported 47% bystander leakage with confab + benign control. #262 generalises away from the confab framing.
- #122 (HIGH confidence) — forward-order null; the contrast condition for understanding what reversing the order changes.
- #121 (HIGH confidence) — "any LoRA SFT destroys persona-specific marker coupling" — frames the C3 benign-first control.
- #151 (proposed) — mechanistic follow-ups (out of scope here, but if H1 holds this is the natural next step).

`Parent: #125`
